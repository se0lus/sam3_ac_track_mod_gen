"""Merge per-clip mask polygons into unified per-tag masks via rasterization.

Reads Stage 4 output (*_masks.json), rasterizes all clip polygons onto a shared
canvas at SAM3 modelscale pixel density, then uses cv2.findContours to extract
clean overlap-free boundaries.  Each contour group (outer + holes) is
pre-triangulated via mapbox_earcut so the downstream Blender script can create
mesh directly without relying on Blender's 2D curve fill.

Supports priority compositing for surface tags: all surface tags are painted on
a single canvas ordered by priority (low→high), eliminating inter-tag gaps.
Stage 2 full-map masks and Stage 2a layout masks can supplement clip-level data.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("sam3_pipeline.mask_merger")

# Colors for composite preview image (BGR for cv2)
TAG_COLORS_BGR = {
    "sand":  (100, 200, 200),  # yellow-ish
    "grass": (0, 200, 0),      # green
    "road2": (180, 180, 180),  # light grey
    "road":  (100, 100, 100),  # dark grey
    "kerb":  (0, 0, 255),      # red
}


def _pixel_size_m(
    bounds: Dict[str, float],
    canvas_w: int,
    canvas_h: int,
) -> float:
    """Compute average pixel size in metres, handling geographic & projected CRS.

    Geographic CRS (WGS84): bounds in degrees → convert via lat/lon scale.
    Projected CRS (UTM etc.): bounds already in metres → direct division.
    """
    import math

    left, right = bounds["left"], bounds["right"]
    bottom, top = bounds["bottom"], bounds["top"]
    geo_w = right - left
    geo_h = top - bottom

    if -180 <= left <= 180 and -180 <= right <= 180 and -90 <= bottom <= 90 and -90 <= top <= 90:
        # Geographic (degrees)
        lat_mid = (top + bottom) / 2.0
        m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat_mid))
        m_per_deg_lat = 111_320.0
        px = geo_w * m_per_deg_lon / canvas_w
        py = geo_h * m_per_deg_lat / canvas_h
    else:
        # Projected (metres)
        px = geo_w / canvas_w
        py = geo_h / canvas_h

    return (px + py) / 2.0


def _read_geotiff_bounds(geotiff_path: str) -> Dict[str, Any]:
    """Read metadata (bounds, size) from GeoTIFF without loading pixels.

    Returns dict with:
        width, height: Pixel dimensions.
        bounds: Native CRS bounds (for pixel↔native coord mapping).
        bounds_wgs84: WGS84 (EPSG:4326) bounds (for geo_xy output to Blender).
            Same as bounds when the GeoTIFF is already in geographic CRS.
            Includes ``corners`` sub-dict for bilinear interpolation when the
            source CRS has grid convergence (e.g. UTM).
    """
    import rasterio
    from rasterio.warp import transform_bounds, transform as warp_transform

    with rasterio.open(geotiff_path) as ds:
        native = {
            "left": ds.bounds.left,
            "bottom": ds.bounds.bottom,
            "right": ds.bounds.right,
            "top": ds.bounds.top,
        }

        # Convert to WGS84 if the GeoTIFF uses a projected CRS
        if ds.crs and not ds.crs.is_geographic:
            l84, b84, r84, t84 = transform_bounds(
                ds.crs, "EPSG:4326",
                ds.bounds.left, ds.bounds.bottom,
                ds.bounds.right, ds.bounds.top,
            )
            wgs84 = {"left": l84, "bottom": b84, "right": r84, "top": t84}
            logger.info(
                "GeoTIFF CRS %s → WGS84: [%.6f, %.6f, %.6f, %.6f]",
                ds.crs, l84, b84, r84, t84,
            )
            # Compute exact WGS84 corners for bilinear interpolation.
            # Order: TL, TR, BL, BR (native coords: left/top, right/top,
            # left/bottom, right/bottom).
            xs = [ds.bounds.left, ds.bounds.right, ds.bounds.left, ds.bounds.right]
            ys = [ds.bounds.top, ds.bounds.top, ds.bounds.bottom, ds.bounds.bottom]
            c_lons, c_lats = warp_transform(ds.crs, "EPSG:4326", xs, ys)
            wgs84["corners"] = {
                "tl": [c_lons[0], c_lats[0]],
                "tr": [c_lons[1], c_lats[1]],
                "bl": [c_lons[2], c_lats[2]],
                "br": [c_lons[3], c_lats[3]],
            }
        else:
            wgs84 = dict(native)

        return {
            "width": ds.width,
            "height": ds.height,
            "bounds": native,
            "bounds_wgs84": wgs84,
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
    """Convert WGS84 lon/lat to canvas pixel coordinates.

    When *bounds* contains a ``corners`` dict (tl/tr/bl/br as [lon, lat]),
    uses inverse bilinear interpolation for accurate mapping with UTM grid
    convergence.  Falls back to the simplified rectangle formula otherwise.
    """
    corners = bounds.get("corners")
    if corners:
        u, v = _inverse_bilinear(lon, lat, corners)
        return u * canvas_w, v * canvas_h

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
    """Convert canvas pixel coordinates back to WGS84 lon/lat.

    When *bounds* contains a ``corners`` dict (tl/tr/bl/br as [lon, lat]),
    uses bilinear interpolation for accurate mapping with UTM grid convergence.
    Falls back to the simplified rectangle formula otherwise.
    """
    corners = bounds.get("corners")
    if corners:
        u = cx / canvas_w
        v = cy / canvas_h
        return _forward_bilinear(u, v, corners)

    left = bounds["left"]
    right = bounds["right"]
    top = bounds["top"]
    bottom = bounds["bottom"]

    lon = left + cx / canvas_w * (right - left)
    lat = top - cy / canvas_h * (top - bottom)
    return lon, lat


def _forward_bilinear(
    u: float, v: float,
    corners: Dict[str, List[float]],
) -> Tuple[float, float]:
    """Bilinear interpolation: (u, v) in [0,1]² → (lon, lat).

    corners: {"tl": [lon,lat], "tr": [lon,lat], "bl": [lon,lat], "br": [lon,lat]}
    u=0,v=0 → TL;  u=1,v=0 → TR;  u=0,v=1 → BL;  u=1,v=1 → BR.
    """
    tl, tr, bl, br = corners["tl"], corners["tr"], corners["bl"], corners["br"]
    lon = (1 - u) * (1 - v) * tl[0] + u * (1 - v) * tr[0] + \
          (1 - u) * v * bl[0] + u * v * br[0]
    lat = (1 - u) * (1 - v) * tl[1] + u * (1 - v) * tr[1] + \
          (1 - u) * v * bl[1] + u * v * br[1]
    return lon, lat


def _inverse_bilinear(
    lon: float, lat: float,
    corners: Dict[str, List[float]],
) -> Tuple[float, float]:
    """Inverse bilinear: (lon, lat) → (u, v) in [0,1]².

    Solves the bilinear system analytically (quadratic in v, then u).
    """
    tl, tr, bl, br = corners["tl"], corners["tr"], corners["bl"], corners["br"]

    # Coefficients for: lon = tl + u*e1 + v*e2 + u*v*e3
    e1 = tr[0] - tl[0]
    e2 = bl[0] - tl[0]
    e3 = tl[0] - tr[0] - bl[0] + br[0]
    f1 = tr[1] - tl[1]
    f2 = bl[1] - tl[1]
    f3 = tl[1] - tr[1] - bl[1] + br[1]

    p = lon - tl[0]
    q = lat - tl[1]

    # Quadratic in v: A*v² + B*v + C = 0
    A = f2 * e3 - e2 * f3
    B = p * f3 - q * e3 - e2 * f1 + f2 * e1
    C = p * f1 - q * e1

    if abs(A) < 1e-18:
        # Nearly linear (typical for small grid convergence)
        v = -C / B if abs(B) > 1e-18 else 0.0
    else:
        disc = B * B - 4.0 * A * C
        if disc < 0:
            disc = 0.0
        sqrt_disc = disc ** 0.5
        v1 = (-B + sqrt_disc) / (2.0 * A)
        v2 = (-B - sqrt_disc) / (2.0 * A)
        # Pick the root in [0, 1] (or closest)
        v = v1 if abs(v1 - 0.5) <= abs(v2 - 0.5) else v2

    denom = e1 + v * e3
    if abs(denom) < 1e-18:
        u = (q - v * f2) / (f1 + v * f3) if abs(f1 + v * f3) > 1e-18 else 0.0
    else:
        u = (p - v * e2) / denom

    return u, v


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
# Stage 2 / Stage 2a mask loading
# ---------------------------------------------------------------------------

def _load_fullmap_mask(
    mask_path: str, canvas_w: int, canvas_h: int,
) -> Optional[np.ndarray]:
    """Load a Stage 2 full-map mask PNG and resize to the merge canvas.

    Stage 2 masks are at modelscale (~1008x998) covering the same geographic
    extent as the merge canvas, so a simple resize with INTER_NEAREST aligns
    them pixel-for-pixel.

    Returns binary mask (0/255) at canvas size, or None if file doesn't exist.
    """
    if not os.path.isfile(mask_path):
        logger.debug("Full-map mask not found: %s", mask_path)
        return None

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        logger.warning("Failed to read full-map mask: %s", mask_path)
        return None

    resized = cv2.resize(mask, (canvas_w, canvas_h), interpolation=cv2.INTER_NEAREST)
    # Ensure binary
    _, binary = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)

    logger.info(
        "  Loaded full-map mask %s: %dx%d -> %dx%d, coverage=%.1f%%",
        os.path.basename(mask_path), mask.shape[1], mask.shape[0],
        canvas_w, canvas_h,
        np.count_nonzero(binary) / binary.size * 100,
    )
    return binary


def _load_layout_masks(
    layouts_json_path: str, canvas_w: int, canvas_h: int,
) -> Optional[np.ndarray]:
    """Load all Stage 2a layout mask PNGs and return their union, resized.

    Reads layouts.json to find mask_file entries, loads each, and ORs them
    together to produce a single union mask covering all layout variants.

    Returns binary mask (0/255) at canvas size, or None if unavailable.
    """
    if not os.path.isfile(layouts_json_path):
        logger.debug("layouts.json not found: %s", layouts_json_path)
        return None

    try:
        with open(layouts_json_path, "r", encoding="utf-8") as f:
            layouts_data = json.load(f)
    except Exception as e:
        logger.warning("Failed to read layouts.json: %s", e)
        return None

    layouts = layouts_data.get("layouts", [])
    if not layouts:
        return None

    layouts_dir = os.path.dirname(layouts_json_path)
    union_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    loaded = 0

    for layout in layouts:
        mask_file = layout.get("mask_file", "")
        if not mask_file:
            continue
        mask_path = os.path.join(layouts_dir, mask_file)
        if not os.path.isfile(mask_path):
            logger.debug("Layout mask not found: %s", mask_path)
            continue

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        resized = cv2.resize(mask, (canvas_w, canvas_h), interpolation=cv2.INTER_NEAREST)
        _, binary = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
        union_mask = np.maximum(union_mask, binary)
        loaded += 1
        logger.info("  Loaded layout mask: %s (%s)", mask_file, layout.get("name", "?"))

    if loaded == 0:
        return None

    logger.info(
        "  Layout masks union: %d layouts, coverage=%.1f%%",
        loaded, np.count_nonzero(union_mask) / union_mask.size * 100,
    )
    return union_mask


def _rasterize_tag_clips(
    tag: str,
    mask_dir: str,
    bounds: Dict[str, float],
    canvas_w: int, canvas_h: int,
) -> np.ndarray:
    """Rasterize all clip polygons for a single tag onto a canvas.

    Reads *_masks.json files from mask_dir/tag/, draws include polygons in
    white and exclude polygons in black (per-clip), then ORs all clips together.

    Returns binary mask (0/255) at canvas size.
    """
    tag_dir = os.path.join(mask_dir, tag)
    canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

    if not os.path.isdir(tag_dir):
        return canvas

    clip_files = sorted([
        os.path.join(tag_dir, f)
        for f in os.listdir(tag_dir)
        if f.endswith("_masks.json")
    ])

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

        clip_canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

        for mask in masks:
            polys = mask.get("polygons", {})

            for inc_poly in polys.get("include", []):
                geo_xy = inc_poly.get("geo_xy", [])
                if len(geo_xy) < 3:
                    continue
                pts = _poly_geo_to_canvas(geo_xy, bounds, canvas_w, canvas_h)
                if len(pts) >= 3:
                    cv2.fillPoly(clip_canvas, [pts], 255)

            for exc_poly in polys.get("exclude", []):
                geo_xy = exc_poly.get("geo_xy", [])
                if len(geo_xy) < 3:
                    continue
                pts = _poly_geo_to_canvas(geo_xy, bounds, canvas_w, canvas_h)
                if len(pts) >= 3:
                    cv2.fillPoly(clip_canvas, [pts], 0)

        canvas = np.maximum(canvas, clip_canvas)
        clips_processed += 1

    if clips_processed > 0:
        logger.info(
            "  Tag '%s': %d clips rasterized, coverage=%.1f%%",
            tag, clips_processed,
            np.count_nonzero(canvas) / canvas.size * 100,
        )

    return canvas


def _absorb_narrow_kerb_into_road(
    composited: Dict[str, np.ndarray],
    bounds: Dict[str, float],
    canvas_w: int,
    canvas_h: int,
    max_width_m: float = 0.30,
    adjacency_m: float = 0.20,
) -> None:
    """Reclassify narrow kerb protrusions adjacent to road as road (in-place).

    SAM3 sometimes misidentifies thin white lane markings as kerb.  These
    false kerbs are narrow (< *max_width_m*) strips connected to or near the
    road surface.  A narrow strip may be attached to a thick kerb body — the
    per-component max width would be dominated by the thick part, so we use
    **morphological opening** to isolate narrow protrusions:

    1. Opening with a circular kernel removes features narrower than
       *max_width_m*.
    2. Over-dilating the opened result by 2 px covers thick-kerb edges so
       they are not mistaken for narrow features.
    3. ``narrow = kerb AND NOT over_dilated_opened`` gives truly narrow
       protrusions (white lines, thin artifacts).
    4. Among those, connected components near road (within *adjacency_m*)
       are reclassified as road.
    """
    if "kerb" not in composited or "road" not in composited:
        return

    kerb = composited["kerb"]
    road = composited["road"]

    if np.count_nonzero(kerb) == 0:
        return

    # --- pixel size in metres ------------------------------------------------
    pixel_size_m = _pixel_size_m(bounds, canvas_w, canvas_h)

    # Opening kernel radius in pixels — features narrower than 2*radius vanish
    radius_px = max(1, int(round(max_width_m / pixel_size_m / 2)))
    if radius_px < 1:
        return

    logger.debug(
        "  Narrow-kerb: pixel_size=%.4f m/px, opening radius=%d px "
        "(≈%.0f cm width)",
        pixel_size_m, radius_px, radius_px * 2 * pixel_size_m * 100,
    )

    # --- Step 1: morphological opening removes narrow features ---------------
    ksize = 2 * radius_px + 1
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    opened = cv2.morphologyEx(kerb, cv2.MORPH_OPEN, open_kernel)

    # --- Step 2: over-dilate to cover thick-kerb edge pixels -----------------
    expand_kernel = np.ones((5, 5), dtype=np.uint8)
    opened_expanded = cv2.dilate(opened, expand_kernel, iterations=1)

    # --- Step 3: narrow features = original minus expanded opened ------------
    narrow = cv2.bitwise_and(kerb, cv2.bitwise_not(opened_expanded))
    n_narrow = np.count_nonzero(narrow)
    if n_narrow == 0:
        logger.debug("  Narrow kerb → road: no narrow features found")
        return

    # --- Step 4: filter to road-adjacent components --------------------------
    adj_px = max(1, int(round(adjacency_m / pixel_size_m)))
    adj_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * adj_px + 1, 2 * adj_px + 1),
    )
    road_dilated = cv2.dilate(road, adj_kernel, iterations=1)

    # Label narrow features and keep only components near road
    num_labels, labels = cv2.connectedComponents(narrow, connectivity=8)
    adjacent_mask = cv2.bitwise_and(narrow, road_dilated)
    adjacent_labels = set(np.unique(labels[adjacent_mask > 0])) - {0}

    merged_components = 0
    merged_pixels = 0

    for lbl in adjacent_labels:
        mask_lbl = labels == lbl
        npx = int(np.count_nonzero(mask_lbl))
        road[mask_lbl] = 255
        kerb[mask_lbl] = 0
        merged_components += 1
        merged_pixels += npx

    if merged_components > 0:
        logger.info(
            "  Narrow kerb → road: merged %d component(s), %d px "
            "(opening radius=%d px ≈ %.0f cm, adjacency=%d px ≈ %.0f cm)",
            merged_components, merged_pixels,
            radius_px, radius_px * 2 * pixel_size_m * 100,
            adj_px, adjacency_m * 100,
        )
    else:
        logger.debug(
            "  Narrow kerb → road: %d narrow px found but none near road",
            n_narrow,
        )


def _absorb_orphan_kerb(
    composited: Dict[str, np.ndarray],
    adjacency_px: int = 2,
) -> None:
    """Absorb kerb components not adjacent to road into their largest neighbor.

    Real kerbs are always adjacent to the road surface.  Kerb components that
    do not touch road (within *adjacency_px*) are false positives — typically
    white markings on grass/concrete or SAM3 artifacts.  Each such orphan is
    reclassified to whichever neighboring surface tag shares the longest
    contact perimeter with it.

    Modifies *composited* masks in-place.
    """
    if "kerb" not in composited or "road" not in composited:
        return

    kerb = composited["kerb"]
    road = composited["road"]

    if np.count_nonzero(kerb) == 0:
        return

    # Which kerb pixels are adjacent to road?
    adj_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * adjacency_px + 1, 2 * adjacency_px + 1),
    )
    road_dilated = cv2.dilate(road, adj_kernel, iterations=1)

    num_labels, labels = cv2.connectedComponents(kerb, connectivity=8)
    road_touch = cv2.bitwise_and(kerb, road_dilated)
    road_labels = set(np.unique(labels[road_touch > 0])) - {0}

    # Orphan labels = components that do NOT touch road
    orphan_labels = set(range(1, num_labels)) - road_labels
    if not orphan_labels:
        logger.debug("  Orphan kerb: all %d component(s) touch road", num_labels - 1)
        return

    # Candidate neighbor tags (everything in composited except kerb itself)
    neighbor_tags = [t for t in composited if t != "kerb"]

    contact_kernel = np.ones((3, 3), dtype=np.uint8)
    absorbed_by: Dict[str, int] = {}   # tag -> pixel count
    absorbed_components = 0
    deleted_components = 0
    deleted_pixels = 0

    for lbl in orphan_labels:
        mask_lbl = (labels == lbl).astype(np.uint8) * 255
        npx = int(np.count_nonzero(mask_lbl))
        if npx == 0:
            continue

        # Dilate component by 1px to find contact perimeter with neighbors
        dilated = cv2.dilate(mask_lbl, contact_kernel, iterations=1)
        border = cv2.subtract(dilated, mask_lbl)  # 1px ring around component

        best_tag = None
        best_contact = 0
        for tag in neighbor_tags:
            contact = int(np.count_nonzero(
                cv2.bitwise_and(border, composited[tag])
            ))
            if contact > best_contact:
                best_contact = contact
                best_tag = tag

        if best_tag is not None:
            composited[best_tag][mask_lbl > 0] = 255
            kerb[mask_lbl > 0] = 0
            absorbed_by[best_tag] = absorbed_by.get(best_tag, 0) + npx
            absorbed_components += 1
        else:
            # No neighbor at all — isolated in empty space, just delete
            kerb[mask_lbl > 0] = 0
            deleted_components += 1
            deleted_pixels += npx

    parts = []
    if absorbed_components > 0:
        detail = ", ".join(f"{t}: {n} px" for t, n in sorted(absorbed_by.items()))
        parts.append(f"absorbed {absorbed_components} → {{{detail}}}")
    if deleted_components > 0:
        parts.append(f"deleted {deleted_components} isolated ({deleted_pixels} px)")
    if parts:
        logger.info("  Orphan kerb: %s", "; ".join(parts))
    else:
        logger.debug("  Orphan kerb: all %d component(s) touch road",
                      num_labels - 1)


def _close_road_gaps(
    composited: Dict[str, np.ndarray],
    bounds: Dict[str, float],
    canvas_w: int,
    canvas_h: int,
    gap_close_m: float = 0.20,
) -> None:
    """Fill narrow unassigned strips between road and neighboring tags (in-place).

    SAM3 often leaves thin lane markings unclassified, creating a gap between
    road and adjacent tags (grass, kerb, sand, road2).  This function finds
    unassigned pixels where ``dist_to_road + dist_to_any_other_tag ≤ gap_px``
    and fills them as road.

    Only unassigned pixels are claimed — existing tag assignments are never
    overwritten.
    """
    if "road" not in composited or gap_close_m <= 0:
        return

    road = composited["road"]
    if np.count_nonzero(road) == 0:
        return

    pixel_size_m = _pixel_size_m(bounds, canvas_w, canvas_h)
    gap_px = gap_close_m / pixel_size_m

    # Build union of all assigned pixels (any tag)
    assigned = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    for mask in composited.values():
        assigned = cv2.bitwise_or(assigned, mask)

    # Unassigned = pixels not covered by any tag
    unassigned = cv2.bitwise_not(assigned)
    if np.count_nonzero(unassigned) == 0:
        logger.debug("  Road gap close: no unassigned pixels")
        return

    # Union of all non-road tags
    others = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    for tag, mask in composited.items():
        if tag != "road":
            others = cv2.bitwise_or(others, mask)

    if np.count_nonzero(others) == 0:
        logger.debug("  Road gap close: no non-road tags to bridge to")
        return

    # Distance transforms (L2, from foreground to background)
    # cv2.distanceTransform needs binary input where 0=foreground for distance
    dist_from_road = cv2.distanceTransform(
        cv2.bitwise_not(road), cv2.DIST_L2, cv2.DIST_MASK_PRECISE,
    )
    dist_from_others = cv2.distanceTransform(
        cv2.bitwise_not(others), cv2.DIST_L2, cv2.DIST_MASK_PRECISE,
    )

    # Fill condition: unassigned AND (dist_road + dist_others ≤ gap_px)
    gap_sum = dist_from_road + dist_from_others
    fill_mask = (unassigned > 0) & (gap_sum <= gap_px)

    n_filled = np.count_nonzero(fill_mask)
    if n_filled == 0:
        logger.debug(
            "  Road gap close: no qualifying gaps (threshold=%.1f px ≈ %.1f cm)",
            gap_px, gap_close_m * 100,
        )
        return

    road[fill_mask] = 255

    logger.info(
        "  Road gap close: filled %d px (threshold=%.1f px ≈ %.1f cm)",
        n_filled, gap_px, gap_close_m * 100,
    )


def _composite_surface_tags(
    clip_masks: Dict[str, np.ndarray],
    fullmap_masks: Dict[str, Optional[np.ndarray]],
    layout_mask: Optional[np.ndarray],
    priority_config: List[Dict[str, Any]],
    canvas_h: int, canvas_w: int,
) -> Dict[str, np.ndarray]:
    """Priority compositing: paint all surface tags on one canvas, low→high.

    Each pixel belongs to exactly one tag (the highest-priority one that covers
    it). This eliminates inter-tag gaps and overlaps.

    Args:
        clip_masks: tag -> binary clip mask (0/255).
        fullmap_masks: tag -> binary full-map mask (0/255) or None.
        layout_mask: Union of all layout masks (0/255) or None.
        priority_config: List of dicts ordered low→high priority, each with:
            - "tag": tag name
            - "clip": bool (use clip data)
            - "stage2_mask": str or None (single Stage 2 mask filename)
            - "stage2_masks": list of str or None (multiple Stage 2 masks to union)
            - "stage2a": bool (use layout mask for this tag)
        canvas_h, canvas_w: Canvas dimensions.

    Returns:
        Dict mapping tag -> binary mask (0/255) extracted from the composite.
    """
    # Assign tag IDs: 1-based, ordered by priority
    tag_ids = {}
    for idx, cfg in enumerate(priority_config):
        tag_ids[cfg["tag"]] = idx + 1  # 1, 2, 3, ...

    # Composite canvas: 0 = unassigned, N = tag_id
    composite = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

    # Paint low→high priority: each overwrites previous
    for cfg in priority_config:
        tag = cfg["tag"]
        tag_id = tag_ids[tag]

        # Union of all sources for this tag
        tag_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

        # Clip-level data
        if cfg.get("clip", False) and tag in clip_masks:
            tag_mask = np.maximum(tag_mask, clip_masks[tag])

        # Stage 2 full-map mask (single)
        fm = fullmap_masks.get(tag)
        if fm is not None:
            tag_mask = np.maximum(tag_mask, fm)

        # Stage 2a layout mask
        if cfg.get("stage2a", False) and layout_mask is not None:
            tag_mask = np.maximum(tag_mask, layout_mask)

        # Paint: anywhere this tag has coverage, set to this tag_id
        composite[tag_mask > 0] = tag_id

        coverage = np.count_nonzero(tag_mask) / tag_mask.size * 100
        logger.info(
            "  Composite: tag '%s' (priority %d) raw_coverage=%.1f%%",
            tag, tag_id, coverage,
        )

    # Extract per-tag binary masks from composite
    result: Dict[str, np.ndarray] = {}
    for cfg in priority_config:
        tag = cfg["tag"]
        tag_id = tag_ids[tag]
        binary = np.where(composite == tag_id, 255, 0).astype(np.uint8)
        final_coverage = np.count_nonzero(binary) / binary.size * 100
        logger.info(
            "  Composite result: tag '%s' final_coverage=%.1f%%",
            tag, final_coverage,
        )
        result[tag] = binary

    return result


# ---------------------------------------------------------------------------
# Contour extraction + triangulation (shared by both composited and independent)
# ---------------------------------------------------------------------------

def extract_contours_and_triangulate(
    binary_mask: np.ndarray,
    tag: str,
    bounds: Dict[str, float],
    canvas_w: int, canvas_h: int,
    simplify_epsilon: float,
    min_contour_area: int,
) -> List[Dict[str, Any]]:
    """Extract contours from a binary mask and triangulate each group.

    Returns list of contour groups, each with include/exclude/vertices/faces.
    """
    contours, hierarchy = cv2.findContours(
        binary_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1
    )

    groups: List[Dict[str, Any]] = []
    outer_to_group: Dict[int, int] = {}
    total_include = 0
    total_exclude = 0
    total_faces = 0

    if contours and hierarchy is not None:
        hier = hierarchy[0]

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
                all_verts: List[List[float]] = list(outer)
                for hole in holes:
                    all_verts.extend(hole)
                group["vertices"] = all_verts
                group["faces"] = faces
                total_faces += len(faces)

    logger.info(
        "Tag '%s': %d groups (%d include, %d exclude, %d triangles) after merge",
        tag, len(groups), total_include, total_exclude, total_faces,
    )

    return groups


# ---------------------------------------------------------------------------
# Main merge
# ---------------------------------------------------------------------------

def _resolve_mask_path(
    filename: str,
    manual_dir: Optional[str],
    fallback_dir: str,
) -> str:
    """Return path for a mask file, preferring 5a manual dir over Stage 2."""
    if manual_dir:
        p = os.path.join(manual_dir, filename)
        if os.path.isfile(p):
            return p
    return os.path.join(fallback_dir, filename)


def merge_clip_masks(
    geotiff_path: str,
    mask_dir: str,
    tags: List[str],
    fullmap_mask_dir: Optional[str] = None,
    layout_mask_dir: Optional[str] = None,
    manual_surface_mask_dir: Optional[str] = None,
    composite_priority: Optional[List[Dict[str, Any]]] = None,
    simplify_epsilon: float = 2.0,
    min_contour_area: int = 100,
    preview_dir: Optional[str] = None,
    road_gap_close_m: float = 0.20,
    kerb_narrow_max_width_m: float = 0.30,
    kerb_narrow_adjacency_m: float = 0.20,
) -> Dict[str, List[Dict[str, Any]]]:
    """Merge per-clip mask polygons into unified per-tag masks.

    Supports two modes:
    - **Independent tags** (listed in ``tags``): each processed separately
      using clip-level rasterization only (original behavior).
    - **Composited surface tags** (listed in ``composite_priority``): all
      painted on a single canvas by priority (low→high) to eliminate gaps.
      Stage 2 full-map masks and Stage 2a layout masks supplement clip data.

    Args:
        geotiff_path: Path to the GeoTIFF for bounds/size metadata.
        mask_dir: Stage 4 output directory (output/04_mask_on_clips).
        tags: List of independent tag names (e.g. ["trees", "building", "water"]).
        fullmap_mask_dir: Stage 2 output directory with *_mask.png files.
        manual_surface_mask_dir: Stage 5a manual surface masks directory.
            When provided, masks here take priority over fullmap_mask_dir.
        layout_mask_dir: Stage 2a output directory with layouts.json.
        composite_priority: List of dicts ordered low→high priority for
            surface tag compositing. Each dict has:
            - "tag": tag name
            - "clip": bool (use clip polygons from Stage 4)
            - "stage2_mask": optional str (filename in fullmap_mask_dir)
            - "stage2_masks": optional list[str] (filenames to union)
            - "stage2a": optional bool (use layout masks)
        simplify_epsilon: approxPolyDP epsilon in canvas pixels.
        min_contour_area: Minimum contour area in canvas pixels to keep.
        preview_dir: Optional directory to save merge preview PNGs.

    Returns:
        Dict mapping tag -> list of contour groups (both composited and
        independent tags combined).
    """
    geo_meta = _read_geotiff_bounds(geotiff_path)
    canvas_w, canvas_h, _scale = _compute_canvas_size(geo_meta, mask_dir)
    bounds = geo_meta["bounds"]
    bounds_wgs84 = geo_meta["bounds_wgs84"]

    if preview_dir:
        os.makedirs(preview_dir, exist_ok=True)

    results: Dict[str, List[Dict[str, Any]]] = {}

    # -----------------------------------------------------------------------
    # Part 1: Priority compositing for surface tags
    # -----------------------------------------------------------------------
    if composite_priority:
        logger.info("=== Priority compositing for %d surface tags ===",
                     len(composite_priority))

        # 1a. Rasterize clip polygons for each composite tag that uses clips
        clip_masks: Dict[str, np.ndarray] = {}
        for cfg in composite_priority:
            tag = cfg["tag"]
            if cfg.get("clip", False):
                clip_masks[tag] = _rasterize_tag_clips(
                    tag, mask_dir, bounds, canvas_w, canvas_h,
                )

        # 1b. Load full-map masks (5a manual > Stage 2 priority)
        fullmap_masks: Dict[str, Optional[np.ndarray]] = {}
        if fullmap_mask_dir:
            for cfg in composite_priority:
                tag = cfg["tag"]

                # For tags with stage2_masks (union of multiple files),
                # check 5a for a single pre-merged mask first
                s2_files = cfg.get("stage2_masks")
                if s2_files and manual_surface_mask_dir:
                    manual_path = os.path.join(
                        manual_surface_mask_dir, f"{tag}_mask.png"
                    )
                    if os.path.isfile(manual_path):
                        m = _load_fullmap_mask(manual_path, canvas_w, canvas_h)
                        if m is not None:
                            fullmap_masks[tag] = m
                            continue  # skip Stage 2 union

                # Single mask file (5a > Stage 2)
                s2_file = cfg.get("stage2_mask")
                if s2_file:
                    path = _resolve_mask_path(
                        f"{tag}_mask.png", manual_surface_mask_dir,
                        fullmap_mask_dir,
                    )
                    # Fall back to Stage 2 original filename if 5a not found
                    if not os.path.isfile(path):
                        path = os.path.join(fullmap_mask_dir, s2_file)
                    fullmap_masks[tag] = _load_fullmap_mask(path, canvas_w, canvas_h)

                # Multiple mask files (union) — only if not already loaded above
                if s2_files and tag not in fullmap_masks:
                    union = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
                    for fname in s2_files:
                        path = os.path.join(fullmap_mask_dir, fname)
                        m = _load_fullmap_mask(path, canvas_w, canvas_h)
                        if m is not None:
                            union = np.maximum(union, m)
                    if np.count_nonzero(union) > 0:
                        fullmap_masks[tag] = union

        # 1c. Load Stage 2a layout masks
        layout_mask: Optional[np.ndarray] = None
        if layout_mask_dir:
            layouts_json = os.path.join(layout_mask_dir, "layouts.json")
            layout_mask = _load_layout_masks(layouts_json, canvas_w, canvas_h)

        # 1d. Composite
        composited = _composite_surface_tags(
            clip_masks=clip_masks,
            fullmap_masks=fullmap_masks,
            layout_mask=layout_mask,
            priority_config=composite_priority,
            canvas_h=canvas_h,
            canvas_w=canvas_w,
        )

        # 1d-post1. Absorb narrow kerb into road
        _absorb_narrow_kerb_into_road(composited, bounds, canvas_w, canvas_h,
                                      max_width_m=kerb_narrow_max_width_m,
                                      adjacency_m=kerb_narrow_adjacency_m)

        # 1d-post2. Absorb orphan kerb (not adjacent to road) into neighbors
        _absorb_orphan_kerb(composited)

        # 1d-post3. Close small gaps in road edges (after kerb absorption)
        _close_road_gaps(composited, bounds, canvas_w, canvas_h,
                         gap_close_m=road_gap_close_m)

        # 1e. Save composite label map for downstream gap-filling (Stage 8)
        if preview_dir:
            # Rebuild label map from post-processed binary masks
            tag_ids = {cfg["tag"]: idx + 1 for idx, cfg in enumerate(composite_priority)}
            composite_labels = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
            for cfg in composite_priority:
                tag = cfg["tag"]
                composite_labels[composited[tag] > 0] = tag_ids[tag]

            parent_dir = os.path.dirname(preview_dir)
            labels_path = os.path.join(parent_dir, "composite_labels.npy")
            np.save(labels_path, composite_labels)

            meta = {
                "canvas_w": canvas_w,
                "canvas_h": canvas_h,
                "bounds": bounds,
                "tag_ids": tag_ids,
            }
            meta_path = os.path.join(parent_dir, "composite_meta.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

            logger.info(
                "  Saved composite_labels.npy (%dx%d) and composite_meta.json to %s",
                canvas_w, canvas_h, parent_dir,
            )

        # 1f. Extract contours + triangulate for each composited tag
        #     Use WGS84 bounds so geo_xy output is in lon/lat for Blender.
        for cfg in composite_priority:
            tag = cfg["tag"]
            binary = composited[tag]
            groups = extract_contours_and_triangulate(
                binary, tag, bounds_wgs84, canvas_w, canvas_h,
                simplify_epsilon, min_contour_area,
            )
            results[tag] = groups

            # Per-tag preview
            if preview_dir:
                preview_path = os.path.join(preview_dir, f"{tag}_merged.png")
                cv2.imwrite(preview_path, binary)
                logger.info("  Preview saved: %s", preview_path)

        # 1g. Multi-color composite preview
        if preview_dir:
            color_preview = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
            for cfg in composite_priority:
                tag = cfg["tag"]
                binary = composited[tag]
                color = TAG_COLORS_BGR.get(tag, (128, 128, 128))
                color_preview[binary > 0] = color
            preview_path = os.path.join(preview_dir, "composite_surface.png")
            cv2.imwrite(preview_path, color_preview)
            logger.info("  Composite preview saved: %s", preview_path)

    # -----------------------------------------------------------------------
    # Part 2: Independent tags (original per-tag logic)
    # -----------------------------------------------------------------------
    for tag in tags:
        logger.info("=== Independent merge for tag '%s' ===", tag)
        full_canvas = _rasterize_tag_clips(
            tag, mask_dir, bounds, canvas_w, canvas_h,
        )

        if np.count_nonzero(full_canvas) == 0:
            logger.warning("Tag '%s': no pixels after rasterization, skipping", tag)
            continue

        groups = extract_contours_and_triangulate(
            full_canvas, tag, bounds_wgs84, canvas_w, canvas_h,
            simplify_epsilon, min_contour_area,
        )
        results[tag] = groups

        if preview_dir:
            preview_path = os.path.join(preview_dir, f"{tag}_merged.png")
            cv2.imwrite(preview_path, full_canvas)
            logger.info("  Preview saved: %s", preview_path)

    return results
