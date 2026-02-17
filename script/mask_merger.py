"""Merge per-clip mask polygons into unified per-tag masks via rasterization.

Reads Stage 4 output (*_masks.json), rasterizes all clip polygons onto a shared
canvas at SAM3 modelscale pixel density, then uses cv2.findContours to extract
clean overlap-free boundaries.  Each contour group (outer + holes) is
pre-triangulated via mapbox_earcut so the downstream Blender script can create
mesh directly without relying on Blender's 2D curve fill.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("sam3_pipeline.mask_merger")


def _read_geotiff_bounds(geotiff_path: str) -> Dict[str, Any]:
    """Read only metadata (bounds, size) from GeoTIFF without loading pixels."""
    import rasterio

    with rasterio.open(geotiff_path) as ds:
        return {
            "width": ds.width,
            "height": ds.height,
            "bounds": {
                "left": ds.bounds.left,
                "bottom": ds.bounds.bottom,
                "right": ds.bounds.right,
                "top": ds.bounds.top,
            },
        }


def _compute_canvas_size(
    geotiff_meta: Dict[str, Any],
    mask_dir: str,
) -> Tuple[int, int, float]:
    """Compute merge canvas size to match SAM3 modelscale pixel density.

    Finds a sample mask JSON to determine the original-to-modelscale ratio,
    then scales the full GeoTIFF accordingly.

    Returns (canvas_width, canvas_height, scale_factor).
    """
    sample_meta = None
    for entry in os.listdir(mask_dir):
        tag_path = os.path.join(mask_dir, entry)
        if not os.path.isdir(tag_path):
            continue
        for f in sorted(os.listdir(tag_path)):
            if f.endswith("_masks.json"):
                with open(os.path.join(tag_path, f), "r", encoding="utf-8") as fh:
                    sample = json.load(fh)
                    sample_meta = sample.get("meta", {})
                break
        if sample_meta:
            break

    if sample_meta is None:
        raise RuntimeError(f"No *_masks.json found in {mask_dir}")

    img_size = sample_meta.get("image_size", {})
    model_size = sample_meta.get("model_scale_size", {})

    orig_w = img_size.get("width", 1)
    model_w = model_size.get("width", 1)

    scale_factor = orig_w / model_w  # typically ~6.05

    geo_w = geotiff_meta["width"]
    geo_h = geotiff_meta["height"]

    canvas_w = int(geo_w / scale_factor)
    canvas_h = int(geo_h / scale_factor)

    logger.info(
        "Canvas size: GeoTIFF %dx%d, scale_factor=%.2f -> merge canvas %dx%d",
        geo_w, geo_h, scale_factor, canvas_w, canvas_h,
    )
    return canvas_w, canvas_h, scale_factor


def _geo_to_canvas(
    lon: float, lat: float,
    bounds: Dict[str, float],
    canvas_w: int, canvas_h: int,
) -> Tuple[float, float]:
    """Convert WGS84 lon/lat to canvas pixel coordinates."""
    left = bounds["left"]
    right = bounds["right"]
    top = bounds["top"]
    bottom = bounds["bottom"]

    cx = (lon - left) / (right - left) * canvas_w
    cy = (top - lat) / (top - bottom) * canvas_h  # Y axis flipped
    return cx, cy


def _canvas_to_geo(
    cx: float, cy: float,
    bounds: Dict[str, float],
    canvas_w: int, canvas_h: int,
) -> Tuple[float, float]:
    """Convert canvas pixel coordinates back to WGS84 lon/lat."""
    left = bounds["left"]
    right = bounds["right"]
    top = bounds["top"]
    bottom = bounds["bottom"]

    lon = left + cx / canvas_w * (right - left)
    lat = top - cy / canvas_h * (top - bottom)
    return lon, lat


def _poly_geo_to_canvas(
    geo_xy: List[List[float]],
    bounds: Dict[str, float],
    canvas_w: int, canvas_h: int,
) -> np.ndarray:
    """Convert a polygon's geo_xy to canvas pixel coords (int32 for fillPoly)."""
    pts = []
    for pt in geo_xy:
        if len(pt) < 2:
            continue
        cx, cy = _geo_to_canvas(pt[0], pt[1], bounds, canvas_w, canvas_h)
        pts.append([int(round(cx)), int(round(cy))])
    return np.array(pts, dtype=np.int32)


# ---------------------------------------------------------------------------
# Earcut triangulation
# ---------------------------------------------------------------------------

def _triangulate_group_earcut(
    outer_geo: List[List[float]],
    holes_geo: List[List[List[float]]],
) -> Optional[List[List[int]]]:
    """Triangulate one outer polygon + holes using mapbox_earcut.

    Args:
        outer_geo: Outer contour as [[lon, lat], ...].
        holes_geo: List of hole contours, each [[lon, lat], ...].

    Returns:
        List of triangle faces as [[i0, i1, i2], ...] indexing into the
        concatenated vertex list (outer + all holes in order), or None on
        failure.
    """
    import mapbox_earcut

    # Build (N, 2) coordinate array and cumulative ring end indices for earcut
    all_coords: List[List[float]] = []
    ring_end_indices: List[int] = []

    for pt in outer_geo:
        all_coords.append([pt[0], pt[1]])
    ring_end_indices.append(len(all_coords))

    for hole in holes_geo:
        for pt in hole:
            all_coords.append([pt[0], pt[1]])
        ring_end_indices.append(len(all_coords))

    coords_np = np.array(all_coords, dtype=np.float64)  # shape (N, 2)
    rings_np = np.array(ring_end_indices, dtype=np.uint32)

    try:
        tri_indices = mapbox_earcut.triangulate_float64(coords_np, rings_np)
    except Exception as e:
        logger.warning("earcut triangulation failed: %s", e)
        return None

    if len(tri_indices) == 0:
        return None

    faces = []
    for i in range(0, len(tri_indices), 3):
        faces.append([int(tri_indices[i]), int(tri_indices[i + 1]), int(tri_indices[i + 2])])
    return faces


# ---------------------------------------------------------------------------
# Main merge
# ---------------------------------------------------------------------------

def merge_clip_masks(
    geotiff_path: str,
    mask_dir: str,
    tags: List[str],
    simplify_epsilon: float = 2.0,
    min_contour_area: int = 100,
    preview_dir: Optional[str] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Merge per-clip mask polygons into unified per-tag masks.

    For each tag, rasterizes all clip polygons onto a single canvas (at SAM3
    modelscale pixel density), using per-clip OR strategy to handle overlapping
    regions correctly.  Then extracts clean contours via cv2.findContours.

    Uses ``RETR_CCOMP`` hierarchy to preserve the parent-child relationship
    between outer contours and their holes.  Each contour group is
    pre-triangulated via earcut so the Blender script can create mesh directly.

    Args:
        geotiff_path: Path to the GeoTIFF for bounds/size metadata.
        mask_dir: Stage 4 output directory (output/04_mask_on_clips).
        tags: List of tag names to process (e.g. ["road", "grass", ...]).
        simplify_epsilon: approxPolyDP epsilon in canvas pixels.
        min_contour_area: Minimum contour area in canvas pixels to keep.
        preview_dir: Optional directory to save merge preview PNGs.

    Returns:
        Dict mapping tag -> list of contour groups.
        Each group dict has:
          - ``"include"``: ``[one_polygon]`` — the outer contour geo_xy
          - ``"exclude"``: ``[hole_polys...]`` — hole contour geo_xy list
          - ``"vertices"``: ``[[lon,lat], ...]`` — all vertices (outer + holes concatenated)
          - ``"faces"``: ``[[i0,i1,i2], ...]`` — triangle indices into vertices
    """
    geo_meta = _read_geotiff_bounds(geotiff_path)
    canvas_w, canvas_h, _scale = _compute_canvas_size(geo_meta, mask_dir)
    bounds = geo_meta["bounds"]

    if preview_dir:
        os.makedirs(preview_dir, exist_ok=True)

    results: Dict[str, List[Dict[str, Any]]] = {}

    for tag in tags:
        tag_dir = os.path.join(mask_dir, tag)
        if not os.path.isdir(tag_dir):
            logger.warning("Tag directory not found: %s, skipping", tag_dir)
            continue

        clip_files = sorted([
            os.path.join(tag_dir, f)
            for f in os.listdir(tag_dir)
            if f.endswith("_masks.json")
        ])

        if not clip_files:
            logger.warning("No mask files for tag '%s', skipping", tag)
            continue

        logger.info("Tag '%s': merging %d clips ...", tag, len(clip_files))

        # Full canvas: OR accumulator across all clips
        full_canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

        clips_processed = 0
        for clip_file in clip_files:
            try:
                with open(clip_file, "r", encoding="utf-8") as f:
                    clip_data = json.load(f)
            except Exception as e:
                logger.error("Failed to read %s: %s", clip_file, e)
                continue

            masks = clip_data.get("masks", [])
            if not masks:
                continue

            # Per-clip canvas: include first, then exclude, then OR into full
            clip_canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

            for mask in masks:
                polys = mask.get("polygons", {})

                # Draw include polygons (white=255)
                for inc_poly in polys.get("include", []):
                    geo_xy = inc_poly.get("geo_xy", [])
                    if len(geo_xy) < 3:
                        continue
                    pts = _poly_geo_to_canvas(geo_xy, bounds, canvas_w, canvas_h)
                    if len(pts) >= 3:
                        cv2.fillPoly(clip_canvas, [pts], 255)

                # Erase exclude polygons (black=0)
                for exc_poly in polys.get("exclude", []):
                    geo_xy = exc_poly.get("geo_xy", [])
                    if len(geo_xy) < 3:
                        continue
                    pts = _poly_geo_to_canvas(geo_xy, bounds, canvas_w, canvas_h)
                    if len(pts) >= 3:
                        cv2.fillPoly(clip_canvas, [pts], 0)

            # OR merge: any clip that marks a pixel as valid keeps it
            full_canvas = np.maximum(full_canvas, clip_canvas)
            clips_processed += 1

        logger.info("Tag '%s': %d clips rasterized, extracting contours ...", tag, clips_processed)

        # Extract contours with 2-level hierarchy (RETR_CCOMP)
        contours, hierarchy = cv2.findContours(
            full_canvas, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1
        )

        # Build grouped results preserving hierarchy + pre-triangulate
        groups: List[Dict[str, Any]] = []
        outer_to_group: Dict[int, int] = {}

        total_include = 0
        total_exclude = 0
        total_faces = 0

        if contours and hierarchy is not None:
            hier = hierarchy[0]  # shape (N, 4): [next, prev, first_child, parent]

            # First pass: simplify all contours and convert to geo_xy
            geo_contours: Dict[int, List[List[float]]] = {}
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area < min_contour_area:
                    continue
                approx = cv2.approxPolyDP(contour, simplify_epsilon, closed=True)
                if len(approx) < 3:
                    continue
                geo_pts: List[List[float]] = []
                for pt in approx:
                    cx, cy = float(pt[0][0]), float(pt[0][1])
                    lon, lat = _canvas_to_geo(cx, cy, bounds, canvas_w, canvas_h)
                    geo_pts.append([lon, lat])
                geo_contours[i] = geo_pts

            # Second pass: build groups from outer contours
            for i in geo_contours:
                if hier[i][3] == -1:  # outer contour (no parent)
                    gidx = len(groups)
                    outer_to_group[i] = gidx
                    groups.append({
                        "include": [geo_contours[i]],
                        "exclude": [],
                        "vertices": None,
                        "faces": None,
                    })
                    total_include += 1

            # Third pass: attach holes to their parent outer contour
            for i in geo_contours:
                parent = hier[i][3]
                if parent >= 0 and parent in outer_to_group:
                    gidx = outer_to_group[parent]
                    groups[gidx]["exclude"].append(geo_contours[i])
                    total_exclude += 1

            # Fourth pass: earcut triangulation for each group
            for group in groups:
                outer = group["include"][0]
                holes = group["exclude"]

                faces = _triangulate_group_earcut(outer, holes)
                if faces is not None:
                    # Build concatenated vertex list: outer + holes in order
                    all_verts: List[List[float]] = list(outer)
                    for hole in holes:
                        all_verts.extend(hole)
                    group["vertices"] = all_verts
                    group["faces"] = faces
                    total_faces += len(faces)

        results[tag] = groups

        logger.info(
            "Tag '%s': %d groups (%d include, %d exclude, %d triangles) after merge",
            tag, len(groups), total_include, total_exclude, total_faces,
        )

        # Save preview image
        if preview_dir:
            preview_path = os.path.join(preview_dir, f"{tag}_merged.png")
            cv2.imwrite(preview_path, full_canvas)
            logger.info("  Preview saved: %s", preview_path)

    return results
