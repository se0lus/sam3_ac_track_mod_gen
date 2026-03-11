"""Stage 8: Gap-fill mask voids + Blender polygon mesh generation.

Phase 1 (gap-fill):
  Re-rasterizes surface masks from Stage 5 result merged JSONs onto a
  fresh canvas, applies wall constraints from Stage 6, fills gaps via
  morphological operations, then re-extracts contours for Blender.

  Key design: reads from 05_result junction (which may point to 05 auto
  OR 05a manual-edited data), so manual edits are always respected.

Phase 2 (Blender):
  Runs Blender headless to create polygon meshes from the gap-filled
  (or original) blender JSON files.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np

logger = logging.getLogger("sam3_pipeline.s08")

_script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from pipeline_config import PipelineConfig


def run(config: PipelineConfig) -> None:
    """Execute Stage 8: Gap-fill + Blender polygon generation."""
    logger.info("=== Stage 8: Blender polygon generation ===")

    if not config.blender_exe:
        raise ValueError("blender_exe is required for blender_polygons stage")

    # Read from 05_result junction (points to 05 or 05a)
    input_dir = config.merge_segments_result
    if not os.path.isdir(input_dir):
        input_dir = config.merge_segments_dir  # fallback
    if not os.path.isdir(input_dir):
        raise ValueError(f"Blender clips directory not found: {input_dir}")

    out_dir = os.path.dirname(config.blend_file)
    os.makedirs(out_dir, exist_ok=True)

    # --- Phase 1: Gap filling ---
    if config.s8_gap_fill_enabled:
        gap_filled_dir = _run_gap_fill(config, input_dir, out_dir)
        if gap_filled_dir:
            input_dir = gap_filled_dir
    else:
        logger.info("Gap filling disabled, using original blender clips")

    # --- Phase 2: Blender polygon generation ---
    blender_script = os.path.join(
        _script_dir, "..", "blender_scripts", "blender_create_polygons.py"
    )
    blender_script = os.path.abspath(blender_script)

    cmd = [
        config.blender_exe,
        "--background",
        "--python", blender_script,
        "--",
        "--input", input_dir,
        "--output", config.blend_file,
    ]
    if config.s8_generate_curves:
        cmd.append("--generate-curves")
    logger.info("Running Blender: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    logger.info("Blender polygon generation complete: %s", config.blend_file)


# ---------------------------------------------------------------------------
# Gap-fill pipeline (Phase 1)
# ---------------------------------------------------------------------------

def _run_gap_fill(
    config: PipelineConfig,
    stage5_dir: str,
    out_dir: str,
) -> str | None:
    """Run gap filling by re-rasterizing from blender JSONs.

    1. Compute canvas from GeoTIFF + Stage 4 metadata
    2. Load walls from Stage 6 → build driveable zone
    3. Rasterize each tag's triangulated meshes from geo_xy in blender JSONs
    4. Build composite (priority: sand < grass < road2 < road < kerb)
    5. Build fill_zone = driveable AND NOT independent_tags
    6. Run gap-fill algorithm
    7. Re-extract contours → Blender coords → gap_filled/ directory

    Returns path to gap_filled/ directory, or None on failure.
    """
    from mask_gap_filler import (
        fill_mask_gaps, build_driveable_zone,
        SURFACE_TAGS, TAG_NAME_TO_ID, INDEPENDENT_TAGS,
        _save_debug_colorized,
    )
    from mask_merger import (
        _read_geotiff_bounds, _compute_canvas_size, _geo_to_canvas,
        extract_contours_and_triangulate,
    )
    from geo_sam3_blender_utils import get_tileset_transform, geo_points_to_blender_xyz

    logger.info("--- Phase 1: Gap filling (rasterize from blender JSONs) ---")

    # ---- 1. Compute canvas dimensions ----
    logger.info("[1/8] Computing canvas dimensions...")
    if not config.geotiff_path or not os.path.isfile(config.geotiff_path):
        logger.warning("GeoTIFF not found: %s. Skipping gap-fill.", config.geotiff_path)
        return None

    geotiff_meta = _read_geotiff_bounds(config.geotiff_path)
    bounds = geotiff_meta["bounds"]
    bounds_wgs84 = geotiff_meta["bounds_wgs84"]

    # Canvas size: use Stage 4 mask dir for scale factor
    mask_on_clips_dir = config.mask_on_clips_dir
    if not os.path.isdir(mask_on_clips_dir):
        logger.warning("Stage 4 dir not found: %s. Skipping gap-fill.", mask_on_clips_dir)
        return None

    canvas_w, canvas_h, _scale = _compute_canvas_size(geotiff_meta, mask_on_clips_dir)
    logger.info("Canvas: %dx%d (scale_factor=%.2f)", canvas_w, canvas_h, _scale)

    # ---- 2. Load walls → build driveable zone ----
    logger.info("[2/8] Building driveable zone...")
    walls, wall_resolution = _load_walls(config)
    if walls is None:
        return None

    debug_dir = os.path.join(out_dir, "gap_fill_debug")
    os.makedirs(debug_dir, exist_ok=True)

    driveable = build_driveable_zone(
        canvas_h, canvas_w, walls,
        wall_resolution=wall_resolution,
        debug_dir=debug_dir,
    )
    logger.info(
        "Driveable zone: %.1f%% of canvas (%d px)",
        np.count_nonzero(driveable) / max(canvas_h * canvas_w, 1) * 100,
        int(np.count_nonzero(driveable)),
    )

    # ---- 3. Rasterize all tags from blender JSONs ----
    logger.info("[3/8] Rasterizing tags from blender JSONs...")
    # Surface tags → rasterize onto canvas, will be composited
    # Independent tags → rasterize for fill_zone exclusion, also copied as-is
    surface_masks = {}  # tag -> (H, W) binary uint8 (0/255)
    independent_masks = {}  # tag -> (H, W) binary uint8 (0/255)

    all_tags = SURFACE_TAGS + INDEPENDENT_TAGS

    # 并发光栅化
    from concurrent.futures import ThreadPoolExecutor
    def rasterize_tag(tag_name):
        json_path = os.path.join(stage5_dir, tag_name, f"{tag_name}_merged_blender.json")
        if not os.path.isfile(json_path):
            return None
        mask = _rasterize_blender_json(json_path, bounds_wgs84, canvas_w, canvas_h)
        return (tag_name, mask, int(np.count_nonzero(mask)))

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(rasterize_tag, all_tags))

    for result in results:
        if result is None:
            continue
        tag_name, mask, n_pixels = result
        if tag_name in SURFACE_TAGS:
            surface_masks[tag_name] = mask
            logger.info("  Rasterized surface '%s': %d px", tag_name, n_pixels)
        else:
            independent_masks[tag_name] = mask
            logger.info("  Rasterized independent '%s': %d px", tag_name, n_pixels)

    if not surface_masks:
        logger.warning("No surface tags rasterized. Skipping gap-fill.")
        return None

    # Save rasterized masks as debug
    if debug_dir:
        for tag, mask in {**surface_masks, **independent_masks}.items():
            cv2.imwrite(os.path.join(debug_dir, f"00_raster_{tag}.png"), mask)

    # ---- 4. Build composite (priority: low→high, high overwrites) ----
    logger.info("[4/8] Building composite (priority: sand < grass < road2 < road < kerb)...")
    #    Then clip to driveable zone so surface pixels inside inner walls
    #    (e.g. kerb inside tree/building walls) are removed.
    composite = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    for tag_name in SURFACE_TAGS:
        if tag_name not in surface_masks:
            continue
        tag_id = TAG_NAME_TO_ID[tag_name]
        composite[surface_masks[tag_name] > 127] = tag_id

    # Clip to driveable zone
    outside = ~driveable
    n_clipped = int(np.count_nonzero(composite[outside]))
    composite[outside] = 0
    if n_clipped > 0:
        logger.info("  Clipped %d surface px outside driveable zone", n_clipped)

    for tag_name in SURFACE_TAGS:
        tag_id = TAG_NAME_TO_ID[tag_name]
        n = int(np.count_nonzero(composite == tag_id))
        if n > 0:
            logger.info("  Composite '%s' (id=%d): %d px", tag_name, tag_id, n)

    # ---- 5. Build fill_zone = driveable AND NOT independent ----
    logger.info("[5/8] Building fill zone (driveable - independent tags)...")
    fill_zone = driveable.copy()
    for tag_name, mask in independent_masks.items():
        n_before = int(np.count_nonzero(fill_zone))
        fill_zone[mask > 127] = False
        n_removed = n_before - int(np.count_nonzero(fill_zone))
        if n_removed > 0:
            logger.info("  Fill zone: removed %d px for '%s'", n_removed, tag_name)

    logger.info(
        "  Fill zone: %.1f%% of canvas (%d px)",
        np.count_nonzero(fill_zone) / max(canvas_h * canvas_w, 1) * 100,
        int(np.count_nonzero(fill_zone)),
    )

    if debug_dir:
        cv2.imwrite(
            os.path.join(debug_dir, "01c_fill_zone_final.png"),
            fill_zone.astype(np.uint8) * 255,
        )
        # 保存 gap-fill 前的 composite
        _save_debug_colorized(composite, "02_composite_before_gapfill", debug_dir)

    # ---- 6. Run gap-fill (before narrow strip filter) ----
    logger.info("[6/8] Running gap-fill algorithm...")
    composite_before_filter = composite.copy()  # 保存副本用于预览
    filled = fill_mask_gaps(
        composite,
        fill_zone,
        bounds,
        gap_threshold_m=config.s8_gap_fill_threshold_m,
        default_tag=config.s8_gap_fill_default_tag,
        debug_dir=debug_dir,
    )
    composite = filled  # 用填充后的结果替换 composite

    # ---- 4.5 Filter narrow strips (remove fragments < min_width_m) ----
    logger.info("[7/8] Filtering narrow strips (min_width=%.2fm)...", config.s8_min_width_m if config.s8_filter_narrow_strips else 0)
    if config.s8_filter_narrow_strips and config.s8_min_width_m > 0:
        from mask_gap_filler import _compute_meters_per_pixel
        logger.info("--- Filtering narrow strips (min_width=%.2fm) ---", config.s8_min_width_m)
        mpp = _compute_meters_per_pixel(bounds, canvas_w, canvas_h)
        radius = max(1, int(round(config.s8_min_width_m / 2 / mpp)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
        probe_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        logger.info("  Filter kernel: radius=%dpx (%.4fm/px)", radius, mpp)

        # Phase 1: 处理每个 tag 的窄条（从高到低优先级）
        for tag_name in reversed(SURFACE_TAGS):
            tag_id = TAG_NAME_TO_ID[tag_name]
            mask = (composite == tag_id).astype(np.uint8) * 255
            if np.count_nonzero(mask) == 0:
                continue

            opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            removed = ((mask > 0) & (opened == 0))
            n_removed = int(np.count_nonzero(removed))
            if n_removed == 0:
                if debug_dir:
                    _save_debug_colorized(composite, f"01d_filter_{tag_name}_nochange", debug_dir)
                continue

            # 连通域分析：对每个窄条区域单独处理
            removed_uint8 = removed.astype(np.uint8) * 255
            num_labels, labels = cv2.connectedComponents(removed_uint8)
            logger.info("  '%s': found %d connected narrow regions", tag_name, num_labels - 1)

            # 预计算所有候选 tag 的 dilated mask（避免重复计算）
            dilated_masks = {}
            for candidate_name in SURFACE_TAGS:
                candidate_id = TAG_NAME_TO_ID[candidate_name]
                if candidate_id == tag_id:
                    continue
                candidate_mask = (composite == candidate_id).astype(np.uint8) * 255
                if np.count_nonzero(candidate_mask) > 0:
                    dilated_masks[candidate_id] = cv2.dilate(candidate_mask, probe_kernel, iterations=1)

            # 并发处理每个连通区域
            def process_region(region_id):
                region_mask = (labels == region_id)
                region_pixels = int(np.count_nonzero(region_mask))
                adjacent_tags = []
                for candidate_id, dilated_mask in dilated_masks.items():
                    if np.any(region_mask & (dilated_mask > 0)):
                        adjacent_tags.append(candidate_id)
                if len(adjacent_tags) > 0:
                    best_fill_id = int(max(adjacent_tags))
                    fill_tag = [k for k, v in TAG_NAME_TO_ID.items() if v == best_fill_id][0]
                    return (region_id, best_fill_id, region_pixels, fill_tag)
                return None

            with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
                results = list(executor.map(process_region, range(1, num_labels)))

            # 应用填充结果（重新计算 region_mask 以节省内存）
            for result in results:
                if result is not None:
                    region_id, best_fill_id, region_pixels, fill_tag = result
                    region_mask = (labels == region_id)
                    composite[region_mask] = best_fill_id
                    logger.info("    Region %d/%d: %dpx filled with '%s'",
                               region_id, num_labels-1, region_pixels, fill_tag)

            # Debug: 保存处理后的结果
            if debug_dir:
                debug_composite = np.zeros_like(composite)
                debug_composite[removed] = tag_id
                dilated_removed = cv2.dilate(removed_uint8, probe_kernel, iterations=1)
                neighbor_ring = (dilated_removed > 0) & ~removed
                for cid in range(1, 6):
                    candidate_mask = (composite == cid)
                    debug_composite[neighbor_ring & candidate_mask] = cid
                _save_debug_colorized(debug_composite, f"06_filter_{tag_name}_detail", debug_dir)

            # Debug: 每处理完一个 tag 保存一次
            if debug_dir:
                _save_debug_colorized(composite, f"06_filter_{tag_name}", debug_dir)

        if debug_dir:
            _save_debug_colorized(composite, "07_after_narrow_filter", debug_dir)

    # ---- 7. Save preview (before gap-fill vs after narrow filter) ----
    _save_gap_fill_preview(composite_before_filter, composite, out_dir)

    # ---- 7.5. Extract shared boundaries (if enabled) ----
    boundary_lib = None
    if config.s8_use_shared_boundaries:
        logger.info("[7.5/8] Extracting shared boundaries...")
        from shared_boundary_extractor import (
            extract_all_shared_boundaries,
            visualize_shared_boundaries,
            generate_boundary_report
        )
        boundary_lib = extract_all_shared_boundaries(
            composite, bounds_wgs84, canvas_w, canvas_h,
            simplify_epsilon=config.s8_shared_boundary_epsilon,
            min_chain_length=3,
            logger=logger,
        )
        logger.info("  Extracted %d shared boundaries", len(boundary_lib.boundaries))

        # Visualize and report boundaries
        if debug_dir:
            vis_path = os.path.join(debug_dir, "08_shared_boundaries.png")
            visualize_shared_boundaries(composite, boundary_lib, vis_path)
            logger.info("  Saved boundary visualization: %s", vis_path)

            report_path = os.path.join(debug_dir, "08_shared_boundaries_report.txt")
            generate_boundary_report(boundary_lib, report_path)
            logger.info("  Saved boundary report: %s", report_path)

    # ---- 8. Re-extract contours + triangulate → Blender coords ----
    logger.info("[8/8] Re-extracting contours and converting to Blender coordinates...")
    gap_filled_dir = os.path.join(out_dir, "gap_filled")
    if os.path.isdir(gap_filled_dir):
        shutil.rmtree(gap_filled_dir)
    os.makedirs(gap_filled_dir, exist_ok=True)

    # Get tileset transform for Blender coordinate conversion
    if not config.tiles_dir:
        logger.warning("tiles_dir not set, cannot convert to Blender coords.")
        return None

    sample_geo = _find_sample_geo(stage5_dir)
    tf_info = get_tileset_transform(
        config.tiles_dir, sample_geo_xy=sample_geo, frame_mode="auto"
    )

    # Re-extract surface tags from filled composite
    def extract_tag(tag_name):
        tag_id = TAG_NAME_TO_ID[tag_name]
        binary = (composite == tag_id).astype(np.uint8) * 255
        n_pixels = int(np.count_nonzero(binary))
        if n_pixels == 0:
            return (tag_name, 0)
        _write_tag_blender_json(
            binary, tag_name, bounds_wgs84, canvas_w, canvas_h,
            tf_info, gap_filled_dir, boundary_lib,
        )
        return (tag_name, n_pixels)

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(extract_tag, SURFACE_TAGS))

    for tag_name, n_pixels in results:
        if n_pixels == 0:
            logger.info("  Tag '%s': no pixels, skipping", tag_name)
        else:
            logger.info("  Tag '%s': extracted %d pixels", tag_name, n_pixels)

    # Copy independent tag JSONs from Stage 5 as-is
    for entry in os.listdir(stage5_dir):
        entry_path = os.path.join(stage5_dir, entry)
        if not os.path.isdir(entry_path):
            continue
        if entry in SURFACE_TAGS or entry == "merge_preview":
            continue
        has_blender = any(
            f.endswith("_blender.json")
            for f in os.listdir(entry_path)
            if os.path.isfile(os.path.join(entry_path, f))
        )
        if has_blender:
            dst = os.path.join(gap_filled_dir, entry)
            shutil.copytree(entry_path, dst, dirs_exist_ok=True)
            logger.info("  Copied independent tag '%s' from Stage 5", entry)

    logger.info("Gap-filled JSONs written to %s", gap_filled_dir)
    return gap_filled_dir


# ---------------------------------------------------------------------------
# Rasterization helpers
# ---------------------------------------------------------------------------

def _rasterize_blender_json(
    json_path: str,
    bounds: dict,
    canvas_w: int,
    canvas_h: int,
) -> np.ndarray:
    """Rasterize a blender JSON's triangulated meshes onto a canvas.

    Reads geo_xy vertices and faces from each mesh_group, converts to
    canvas pixel coordinates, and fills triangles via cv2.fillPoly.

    Returns (H, W) uint8 binary mask (0 or 255).
    """
    from mask_merger import _geo_to_canvas

    mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for mg in data.get("mesh_groups", []):
        geo_xy = mg.get("geo_xy", [])
        faces = mg.get("faces", [])
        if not geo_xy or not faces:
            continue

        # Convert all geo_xy to canvas pixels
        canvas_pts = []
        for pt in geo_xy:
            if len(pt) < 2:
                canvas_pts.append((0, 0))
                continue
            cx, cy = _geo_to_canvas(pt[0], pt[1], bounds, canvas_w, canvas_h)
            canvas_pts.append((cx, cy))

        # Fill each triangle
        for face in faces:
            if len(face) < 3:
                continue
            tri = np.array([
                [int(round(canvas_pts[face[0]][0])), int(round(canvas_pts[face[0]][1]))],
                [int(round(canvas_pts[face[1]][0])), int(round(canvas_pts[face[1]][1]))],
                [int(round(canvas_pts[face[2]][0])), int(round(canvas_pts[face[2]][1]))],
            ], dtype=np.int32)
            cv2.fillPoly(mask, [tri], 255)

    return mask


def _write_tag_blender_json(
    binary: np.ndarray,
    tag_name: str,
    bounds: dict,
    canvas_w: int,
    canvas_h: int,
    tf_info,
    gap_filled_dir: str,
    boundary_lib=None,
) -> None:
    """Extract contours from binary mask, triangulate, convert to Blender
    coords, and write *_merged_blender.json."""
    from mask_merger import extract_contours_and_triangulate
    from geo_sam3_blender_utils import geo_points_to_blender_xyz

    groups = extract_contours_and_triangulate(
        binary, tag_name, bounds, canvas_w, canvas_h,
        simplify_epsilon=2.0,
        min_contour_area=100,
        shared_boundaries=boundary_lib,
    )

    mesh_groups = []
    total_verts = 0
    total_faces = 0

    for group_idx, group in enumerate(groups):
        verts_geo = group.get("vertices")
        faces = group.get("faces")
        if verts_geo is None or faces is None:
            continue

        points_xyz = geo_points_to_blender_xyz(
            verts_geo, tf_info, z_mode="zero"
        )
        if len(points_xyz) < 3:
            continue

        mesh_groups.append({
            "group_index": group_idx,
            "tag": tag_name,
            "points_xyz": points_xyz,
            "faces": faces,
            "geo_xy": verts_geo,
        })
        total_verts += len(points_xyz)
        total_faces += len(faces)

    result = {
        "origin": {
            "ecef": list(tf_info.origin_ecef),
            "lonlat": [tf_info.origin_lon, tf_info.origin_lat],
            "h": tf_info.origin_h,
            "source": tf_info.origin_src,
        },
        "frame": {
            "mode": tf_info.effective_mode,
            "tileset_transform_source": tf_info.tf_source,
        },
        "source_tag": tag_name,
        "mesh_groups": mesh_groups,
    }

    tag_dir = os.path.join(gap_filled_dir, tag_name)
    os.makedirs(tag_dir, exist_ok=True)
    out_path = os.path.join(tag_dir, f"{tag_name}_merged_blender.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info(
        "  Gap-filled %s: %d groups, %d verts, %d faces",
        tag_name, len(mesh_groups), total_verts, total_faces,
    )


# ---------------------------------------------------------------------------
# Wall / data loading helpers
# ---------------------------------------------------------------------------

def _load_walls(config: PipelineConfig):
    """Load walls from Stage 6 result and detect wall resolution.

    Returns (walls_list, wall_resolution) or (None, None) on failure.
    """
    walls_result_dir = config.walls_result_dir
    if not os.path.isdir(walls_result_dir):
        walls_result_dir = config.stage_dir("ai_walls")
    walls_json_path = os.path.join(walls_result_dir, "walls.json")
    if not os.path.isfile(walls_json_path):
        logger.warning("walls.json not found at %s. Skipping gap-fill.", walls_json_path)
        return None, None

    with open(walls_json_path, "r", encoding="utf-8") as f:
        walls_data = json.load(f)
    walls = walls_data.get("walls", [])
    logger.info("Loaded %d walls from %s", len(walls), walls_json_path)

    wall_resolution = _detect_wall_resolution(config)
    if wall_resolution:
        logger.info(
            "Wall resolution (Stage 2 modelscale): %dx%d",
            wall_resolution[0], wall_resolution[1],
        )

    return walls, wall_resolution


def _detect_wall_resolution(config: PipelineConfig):
    """Detect the pixel resolution at which walls were generated.

    Walls are generated from Stage 2 modelscale masks. Returns
    (width, height) of the modelscale image, or None if not found.
    """
    mask_dir = config.mask_full_map_result
    if not os.path.isdir(mask_dir):
        mask_dir = config.mask_full_map_dir
    if not os.path.isdir(mask_dir):
        return None

    for f in os.listdir(mask_dir):
        if f.endswith("_modelscale.png") or f.endswith("_modelscale_original.png"):
            path = os.path.join(mask_dir, f)
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                h, w = img.shape[:2]
                return (w, h)

    for f in os.listdir(mask_dir):
        if f.endswith("_mask.png"):
            path = os.path.join(mask_dir, f)
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                h, w = img.shape[:2]
                return (w, h)

    return None


def _find_sample_geo(stage5_dir: str):
    """Find a sample geo coordinate from existing blender JSONs."""
    for root, _, files in os.walk(stage5_dir):
        for fname in files:
            if fname.endswith("_blender.json"):
                try:
                    with open(os.path.join(root, fname), "r", encoding="utf-8") as f:
                        data = json.load(f)
                    for mg in data.get("mesh_groups", []):
                        geo_xy = mg.get("geo_xy", [])
                        if geo_xy and len(geo_xy) > 0:
                            return (geo_xy[0][0], geo_xy[0][1])
                except Exception:
                    continue
    return None


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------

def _save_gap_fill_preview(
    before: np.ndarray,
    after: np.ndarray,
    out_dir: str,
) -> None:
    """Save a side-by-side before/after preview image."""
    from mask_gap_filler import TAG_NAME_TO_ID, _TAG_COLORS_BGR

    h, w = before.shape[:2]

    def colorize(labels):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        for name, tid in TAG_NAME_TO_ID.items():
            color = _TAG_COLORS_BGR.get(name, (128, 128, 128))
            img[labels == tid] = color
        return img

    before_img = colorize(before)
    after_img = colorize(after)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(before_img, "Before", (10, 30), font, 0.8, (255, 255, 255), 2)
    cv2.putText(after_img, "After", (10, 30), font, 0.8, (255, 255, 255), 2)

    preview = np.hstack([before_img, after_img])
    preview_path = os.path.join(out_dir, "gap_fill_preview.png")
    cv2.imwrite(preview_path, preview)
    logger.info("Gap fill preview saved: %s", preview_path)


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )
    p = argparse.ArgumentParser(description="Stage 8: Blender polygon generation")
    p.add_argument("--blender-exe", default="", help="Path to Blender executable")
    p.add_argument("--geotiff", default="", help="Path to GeoTIFF file")
    p.add_argument("--tiles-dir", default="", help="Directory with tileset.json")
    p.add_argument("--output-dir", default="output", help="Output base directory")
    p.add_argument("--no-gap-fill", action="store_true",
                    help="Skip mask gap filling")
    p.add_argument("--gap-fill-threshold", type=float, default=0.20,
                    help="Gap fill threshold in metres (default: 0.20)")
    p.add_argument("--gap-fill-default-tag", default="road2",
                    help="Default fill tag for remaining voids (default: road2)")
    args = p.parse_args()
    config = PipelineConfig(
        geotiff_path=args.geotiff,
        tiles_dir=args.tiles_dir,
        output_dir=args.output_dir,
    ).resolve()
    if args.blender_exe:
        config.blender_exe = args.blender_exe
    config.s8_gap_fill_enabled = not args.no_gap_fill
    config.s8_gap_fill_threshold_m = args.gap_fill_threshold
    config.s8_gap_fill_default_tag = args.gap_fill_default_tag
    run(config)
