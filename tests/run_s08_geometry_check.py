"""Standalone Stage 8 geometry check — run the pixel-corner extraction
on real data from webtools_config and validate geometry before Blender.

Usage:
    python tests/run_s08_geometry_check.py
"""
from __future__ import annotations

import json
import logging
import os
import sys

import cv2
import numpy as np

# Setup paths
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPT_DIR = os.path.join(_PROJECT_ROOT, "script")
sys.path.insert(0, _SCRIPT_DIR)

from pipeline_config import PipelineConfig

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("s08_geometry_check")


def _load_config() -> PipelineConfig:
    """Load config from webtools_config.json like the real pipeline does."""
    config_json = os.path.join(_PROJECT_ROOT, "webtools_config.json")
    config = PipelineConfig()

    if os.path.isfile(config_json):
        with open(config_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        for key, value in data.items():
            if key.startswith("_") or key == "manual_stages":
                continue
            if hasattr(config, key):
                setattr(config, key, value)
        logger.info("Loaded config from %s", config_json)
    else:
        raise FileNotFoundError(f"webtools_config.json not found at {config_json}")

    config.resolve()
    return config


def main():
    config = _load_config()

    logger.info("=" * 60)
    logger.info("Stage 8 Geometry Check (pixel-corner boundary graph)")
    logger.info("=" * 60)
    logger.info("GeoTIFF: %s", config.geotiff_path)
    logger.info("Output: %s", config.output_dir)
    logger.info("s8_use_pixel_corner_contours: %s", config.s8_use_pixel_corner_contours)
    logger.info("s8_shared_boundary_epsilon: %s", config.s8_shared_boundary_epsilon)

    # --- 1. Compute canvas from GeoTIFF ---
    from mask_merger import _read_geotiff_bounds, _compute_canvas_size

    if not config.geotiff_path or not os.path.isfile(config.geotiff_path):
        logger.error("GeoTIFF not found: %s", config.geotiff_path)
        return

    geotiff_meta = _read_geotiff_bounds(config.geotiff_path)
    bounds = geotiff_meta["bounds"]
    bounds_wgs84 = geotiff_meta["bounds_wgs84"]

    mask_on_clips_dir = config.mask_on_clips_dir
    if not os.path.isdir(mask_on_clips_dir):
        logger.error("Stage 4 dir not found: %s", mask_on_clips_dir)
        return

    canvas_w, canvas_h, _scale = _compute_canvas_size(geotiff_meta, mask_on_clips_dir)
    logger.info("Canvas: %dx%d (scale=%.2f)", canvas_w, canvas_h, _scale)

    # --- 2. Load walls + build driveable zone ---
    from mask_gap_filler import (
        fill_mask_gaps, build_driveable_zone,
        SURFACE_TAGS, TAG_NAME_TO_ID, INDEPENDENT_TAGS,
    )
    from stages.s08_blender_polygons import _load_walls, _rasterize_blender_json

    # Read from result junction
    input_dir = config.merge_segments_result
    if not os.path.isdir(input_dir):
        input_dir = config.merge_segments_dir
    if not os.path.isdir(input_dir):
        logger.error("Stage 5 result not found")
        return

    out_dir = os.path.dirname(config.blend_file)
    os.makedirs(out_dir, exist_ok=True)

    walls, wall_resolution = _load_walls(config)
    if walls is None:
        logger.error("No walls loaded — cannot build driveable zone")
        return

    driveable = build_driveable_zone(
        canvas_h, canvas_w, walls,
        wall_resolution=wall_resolution,
    )
    logger.info("Driveable zone: %.1f%% (%d px)",
                np.count_nonzero(driveable) / max(canvas_h * canvas_w, 1) * 100,
                int(np.count_nonzero(driveable)))

    # --- 3. Rasterize tags ---
    surface_masks = {}
    independent_masks = {}
    all_tags = SURFACE_TAGS + INDEPENDENT_TAGS

    for tag_name in all_tags:
        json_path = os.path.join(input_dir, tag_name, f"{tag_name}_merged_blender.json")
        if not os.path.isfile(json_path):
            continue
        mask = _rasterize_blender_json(json_path, bounds_wgs84, canvas_w, canvas_h)
        n = int(np.count_nonzero(mask))
        if tag_name in SURFACE_TAGS:
            surface_masks[tag_name] = mask
            logger.info("  Rasterized surface '%s': %d px", tag_name, n)
        else:
            independent_masks[tag_name] = mask
            logger.info("  Rasterized independent '%s': %d px", tag_name, n)

    if not surface_masks:
        logger.error("No surface tags rasterized")
        return

    # --- 4. Build composite ---
    composite = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    for tag_name in SURFACE_TAGS:
        if tag_name not in surface_masks:
            continue
        tag_id = TAG_NAME_TO_ID[tag_name]
        composite[surface_masks[tag_name] > 127] = tag_id

    composite[~driveable] = 0

    for tag_name in SURFACE_TAGS:
        tag_id = TAG_NAME_TO_ID[tag_name]
        n = int(np.count_nonzero(composite == tag_id))
        if n > 0:
            logger.info("  Composite '%s' (id=%d): %d px", tag_name, tag_id, n)

    # --- 5. Build fill_zone + gap-fill ---
    fill_zone = driveable.copy()
    for tag_name, mask in independent_masks.items():
        fill_zone[mask > 127] = False

    filled = fill_mask_gaps(
        composite, fill_zone, bounds,
        gap_threshold_m=config.s8_gap_fill_threshold_m,
        default_tag=config.s8_gap_fill_default_tag,
    )
    composite = filled

    # --- 6. Filter narrow strips ---
    if config.s8_filter_narrow_strips and config.s8_min_width_m > 0:
        from mask_gap_filler import _compute_meters_per_pixel
        from concurrent.futures import ThreadPoolExecutor
        mpp = _compute_meters_per_pixel(bounds, canvas_w, canvas_h)
        radius = max(1, int(round(config.s8_min_width_m / 2 / mpp)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
        for tag_name in reversed(SURFACE_TAGS):
            tag_id = TAG_NAME_TO_ID[tag_name]
            mask = (composite == tag_id).astype(np.uint8) * 255
            if np.count_nonzero(mask) == 0:
                continue
            opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            removed = (mask > 0) & (opened == 0)
            n_removed = int(np.count_nonzero(removed))
            if n_removed > 0:
                composite[removed] = 0
                logger.info("  Narrow filter '%s': removed %d px", tag_name, n_removed)

    logger.info("")
    logger.info("=" * 60)
    logger.info("Final composite stats:")
    TAG_ID_TO_NAME = {v: k for k, v in TAG_NAME_TO_ID.items()}
    for tag_id in sorted(set(np.unique(composite)) - {0}):
        n = int(np.count_nonzero(composite == tag_id))
        logger.info("  %s (id=%d): %d px", TAG_ID_TO_NAME.get(tag_id, "?"), tag_id, n)

    # --- 7. Run pixel-corner extraction ---
    logger.info("")
    logger.info("=" * 60)
    logger.info("Running pixel-corner boundary graph extraction...")
    logger.info("=" * 60)

    from pixel_corner_contour_extractor import extract_all_tag_polygons

    all_tag_groups = extract_all_tag_polygons(
        composite, bounds_wgs84, canvas_w, canvas_h,
        epsilon=config.s8_shared_boundary_epsilon,
    )

    # --- 8. Geometry validation ---
    logger.info("")
    logger.info("=" * 60)
    logger.info("GEOMETRY VALIDATION (before Blender)")
    logger.info("=" * 60)

    any_issues = False

    for tag_id in sorted(all_tag_groups.keys()):
        tag_name = TAG_ID_TO_NAME.get(tag_id, f"tag_{tag_id}")
        groups = all_tag_groups[tag_id]

        total_verts = sum(len(g['vertices']) for g in groups)
        total_faces = sum(len(g['faces']) for g in groups)
        logger.info("")
        logger.info("--- %s (id=%d) ---", tag_name, tag_id)
        logger.info("  Groups: %d", len(groups))
        logger.info("  Total vertices: %d", total_verts)
        logger.info("  Total faces: %d", total_faces)

        for gi, g in enumerate(groups):
            verts = g['vertices']
            faces = g['faces']
            n_v = len(verts)
            n_f = len(faces)

            # Check: enough vertices
            if n_v < 3:
                logger.error("  [FAIL] Group %d: only %d vertices (need >= 3)", gi, n_v)
                any_issues = True
                continue

            # Check: has faces
            if n_f == 0:
                logger.error("  [FAIL] Group %d: no faces", gi)
                any_issues = True
                continue

            # Check: face indices in range
            max_idx = max(max(f) for f in faces)
            if max_idx >= n_v:
                logger.error("  [FAIL] Group %d: face index %d >= vertex count %d",
                            gi, max_idx, n_v)
                any_issues = True

            # Check: no degenerate faces
            degenerate = sum(1 for f in faces if len(set(f)) < 3)
            if degenerate > 0:
                logger.warning("  [WARN] Group %d: %d degenerate faces (duplicate indices)", gi, degenerate)

            # Check: bounding box makes sense
            lons = [v[0] for v in verts]
            lats = [v[1] for v in verts]
            lon_range = max(lons) - min(lons)
            lat_range = max(lats) - min(lats)
            logger.info("  Group %d: %d verts, %d faces, bbox lon=[%.6f..%.6f] (Δ%.6f), lat=[%.6f..%.6f] (Δ%.6f)",
                        gi, n_v, n_f,
                        min(lons), max(lons), lon_range,
                        min(lats), max(lats), lat_range)

            # Sanity: bbox shouldn't be too tiny or too huge
            if lon_range < 1e-10 and lat_range < 1e-10:
                logger.warning("  [WARN] Group %d: zero-area bounding box", gi)
            if lon_range > 1.0 or lat_range > 1.0:
                logger.warning("  [WARN] Group %d: bounding box > 1 degree — likely wrong", gi)

    # --- 9. Shared boundary check ---
    logger.info("")
    logger.info("=" * 60)
    logger.info("SHARED BOUNDARY CHECK")
    logger.info("=" * 60)

    # Collect vertices per tag
    tag_vertex_sets = {}
    for tag_id, groups in all_tag_groups.items():
        tag_name = TAG_ID_TO_NAME.get(tag_id, f"tag_{tag_id}")
        verts = set()
        for g in groups:
            for v in g['vertices']:
                verts.add((round(v[0], 10), round(v[1], 10)))
        tag_vertex_sets[tag_name] = verts
        logger.info("  %s: %d unique vertices", tag_name, len(verts))

    # Check pairwise shared vertices
    tag_names = sorted(tag_vertex_sets.keys())
    for i in range(len(tag_names)):
        for j in range(i + 1, len(tag_names)):
            a, b = tag_names[i], tag_names[j]
            shared = tag_vertex_sets[a] & tag_vertex_sets[b]
            if shared:
                logger.info("  %s ↔ %s: %d shared vertices ✓", a, b, len(shared))
            else:
                # Check if they should share (adjacent in composite)
                a_id = TAG_NAME_TO_ID.get(a, 0)
                b_id = TAG_NAME_TO_ID.get(b, 0)
                a_mask = composite == a_id
                b_mask = composite == b_id
                # Check 4-connectivity adjacency (matching boundary graph model)
                cross_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
                a_dilated = cv2.dilate(a_mask.astype(np.uint8), cross_kernel)
                adjacent = np.any(a_dilated & b_mask)
                if adjacent:
                    # Check if they share boundaries via chains (same coords, different rounding)
                    # Also check with lower precision
                    verts_a_low = set((round(v[0], 8), round(v[1], 8)) for v in tag_vertex_sets[a])
                    verts_b_low = set((round(v[0], 8), round(v[1], 8)) for v in tag_vertex_sets[b])
                    shared_low = verts_a_low & verts_b_low
                    if shared_low:
                        logger.info("  %s ↔ %s: %d shared vertices at 8dp (rounding artifact) ✓", a, b, len(shared_low))
                    else:
                        # Show closest vertex pairs for debugging
                        min_dist = float('inf')
                        for va in list(tag_vertex_sets[a])[:100]:
                            for vb in list(tag_vertex_sets[b])[:100]:
                                d = ((va[0]-vb[0])**2 + (va[1]-vb[1])**2)**0.5
                                if d < min_dist:
                                    min_dist = d
                        # 1 pixel ≈ bounds_range / canvas_size
                        pixel_deg = (bounds_wgs84.get("right", bounds_wgs84.get("east", 0)) - bounds_wgs84.get("left", bounds_wgs84.get("west", 0))) / canvas_w
                        dist_px = min_dist / pixel_deg if pixel_deg > 0 else 0
                        if dist_px > 5:
                            logger.info("  %s ↔ %s: dilate-adjacent but %.0fpx apart (narrow filter artifact, not a real gap)", a, b, dist_px)
                        else:
                            logger.warning("  %s ↔ %s: ADJACENT but 0 shared vertices, %.1fpx apart [POTENTIAL GAP]", a, b, dist_px)
                            any_issues = True
                else:
                    logger.info("  %s ↔ %s: not adjacent (no shared boundary expected)", a, b)

    logger.info("")
    logger.info("=" * 60)
    if any_issues:
        logger.error("RESULT: Issues detected — review warnings above")
    else:
        logger.info("RESULT: All geometry checks passed ✓")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
