"""Stage 5a: Manual surface mask editing (optional step).

Users refine per-tag surface masks via the web-based surface editor.
This stage initialises the output directory by copying Stage 5's complete
output (merge_preview masks + blender JSON subdirectories) so that the
05_result junction can point here and Stage 6 sees a format-compatible dir.

Editable surface tags: road, grass, sand, kerb, road2.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import sys

import cv2
import numpy as np

logger = logging.getLogger("sam3_pipeline.s05a")

TILE_SIZE = 512

_script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from pipeline_config import PipelineConfig

# Priority compositing order (low → high), same as Stage 5's composite_priority.
# Higher-priority tags overwrite lower: kerb > road > road2 > grass > sand.
SURFACE_TAGS = ["sand", "grass", "road2", "road", "kerb"]

TAG_COLORS = {
    "sand":  "#c8c864",
    "grass": "#00c800",
    "road2": "#b4b4b4",
    "road":  "#666666",
    "kerb":  "#ff0000",
}

TAG_LABELS = {
    "sand":  "砂石",
    "grass": "草地",
    "road2": "次路面",
    "road":  "路面",
    "kerb":  "路缘",
}


def run(config: PipelineConfig) -> None:
    """Initialise the manual surface masks directory.

    Copies all files from Stage 5 output, including:
    - merge_preview/ directory (composited masks)
    - {tag}/{tag}_merged_blender.json subdirectories
    - Creates {tag}_mask.png from merge_preview for editing
    - Writes surface_masks.json metadata
    """
    logger.info("=== Stage 5a: Manual surface mask management ===")

    stage5_dir = config.blender_clips_dir
    out_dir = config.stage_dir("manual_surface_masks")
    masks_dir = os.path.join(out_dir, "masks")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    # Migrate old root-level masks into masks/ subdirectory
    for tag in SURFACE_TAGS:
        old_root = os.path.join(out_dir, f"{tag}_mask.png")
        new_sub = os.path.join(masks_dir, f"{tag}_mask.png")
        if os.path.isfile(old_root) and not os.path.isfile(new_sub):
            shutil.move(old_root, new_sub)
            logger.info("  Migrated %s_mask.png -> masks/", tag)
    old_meta = os.path.join(out_dir, "surface_masks.json")
    new_meta = os.path.join(masks_dir, "surface_masks.json")
    if os.path.isfile(old_meta) and not os.path.isfile(new_meta):
        shutil.move(old_meta, new_meta)
        logger.info("  Migrated surface_masks.json -> masks/")

    if not os.path.isdir(stage5_dir):
        logger.error("Stage 5 output not found: %s. Run stage 5 first.", stage5_dir)
        return

    # Copy merge_preview/ directory
    stage5_preview_dir = os.path.join(stage5_dir, "merge_preview")
    out_preview_dir = os.path.join(out_dir, "merge_preview")
    if os.path.isdir(stage5_preview_dir) and not os.path.isdir(out_preview_dir):
        shutil.copytree(stage5_preview_dir, out_preview_dir)
        logger.info("Copied merge_preview/ from stage 5")

    # Copy all tag subdirectories (contain *_merged_blender.json)
    for entry in os.listdir(stage5_dir):
        src_sub = os.path.join(stage5_dir, entry)
        dst_sub = os.path.join(out_dir, entry)
        if os.path.isdir(src_sub) and entry != "merge_preview":
            if not os.path.isdir(dst_sub):
                shutil.copytree(src_sub, dst_sub)
                logger.info("Copied %s/ from stage 5", entry)

    # Detect canvas dimensions from first available merged mask
    canvas_w, canvas_h = 0, 0
    preview_dir = out_preview_dir if os.path.isdir(out_preview_dir) else stage5_preview_dir
    if os.path.isdir(preview_dir):
        for tag in SURFACE_TAGS:
            src = os.path.join(preview_dir, f"{tag}_merged.png")
            if os.path.isfile(src):
                m = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
                if m is not None:
                    canvas_h, canvas_w = m.shape[:2]
                    break
    if canvas_w == 0:
        logger.error("No merged mask found in preview directory")
        return
    logger.info("Canvas dimensions: %dx%d", canvas_w, canvas_h)

    # Create editable {tag}_mask.png in masks/ subdirectory (skip if exists)
    for tag in SURFACE_TAGS:
        dst = os.path.join(masks_dir, f"{tag}_mask.png")
        if os.path.isfile(dst):
            logger.info("  masks/%s_mask.png already exists, skipping", tag)
            continue
        src = os.path.join(preview_dir, f"{tag}_merged.png")
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            logger.info("  Copied %s_merged.png -> masks/%s_mask.png", tag, tag)
        else:
            logger.warning("  %s_merged.png not found in preview", tag)

    # Create / update surface_masks.json in masks/ subdirectory
    grid_cols = math.ceil(canvas_w / TILE_SIZE)
    grid_rows = math.ceil(canvas_h / TILE_SIZE)
    surface_meta = {
        "image_width": canvas_w,
        "image_height": canvas_h,
        "tile_size": TILE_SIZE,
        "grid_cols": grid_cols,
        "grid_rows": grid_rows,
        "geotiff_path": config.geotiff_path,
        "tags": [
            {
                "tag": tag,
                "color": TAG_COLORS[tag],
                "label": TAG_LABELS[tag],
                "mask_file": f"{tag}_mask.png",
            }
            for tag in SURFACE_TAGS
        ],
    }
    surface_json = os.path.join(masks_dir, "surface_masks.json")
    with open(surface_json, "w", encoding="utf-8") as f:
        json.dump(surface_meta, f, indent=2, ensure_ascii=False)
    logger.info("Wrote masks/surface_masks.json with %d tags", len(SURFACE_TAGS))

    logger.info("Manual surface masks directory ready: %s", out_dir)


def reconvert_masks(config: PipelineConfig) -> None:
    """Re-generate blender JSON from edited mask PNGs.

    Called after the user edits masks in the surface editor and flushes.
    Reads each {tag}_mask.png, extracts contours + triangulates, converts
    to Blender coordinates, and writes {tag}/{tag}_merged_blender.json
    in the same format as Stage 5.
    """
    from mask_merger import extract_contours_and_triangulate
    from geo_sam3_blender_utils import get_tileset_transform, geo_points_to_blender_xyz

    out_dir = config.stage_dir("manual_surface_masks")
    masks_dir = os.path.join(out_dir, "masks")
    if not os.path.isdir(masks_dir):
        logger.warning("5a masks directory not found: %s", masks_dir)
        return

    # Read geo bounds from result_masks.json (via 02_result or direct)
    result_dir = config.mask_full_map_result
    if not os.path.isdir(result_dir):
        result_dir = config.mask_full_map_dir
    masks_json = os.path.join(result_dir, "result_masks.json")
    if not os.path.isfile(masks_json):
        logger.error("result_masks.json not found in %s", result_dir)
        return

    with open(masks_json, "r", encoding="utf-8") as f:
        masks_data = json.load(f)

    meta = masks_data.get("meta", {})
    geo_bounds = meta.get("geo", {}).get("bounds", {})
    bounds = {
        "left": geo_bounds.get("left", 0),
        "right": geo_bounds.get("right", 0),
        "top": geo_bounds.get("top", 0),
        "bottom": geo_bounds.get("bottom", 0),
    }

    # Get tileset transform (need tiles_dir + sample_geo_xy for tileset_local mode)
    # Without sample_geo_xy, get_tileset_transform falls back to ENU mode,
    # which is a different coordinate system than Stage 5's tileset_local.
    sample_geo_xy = None
    if bounds["left"] and bounds["top"]:
        cx = (bounds["left"] + bounds["right"]) / 2
        cy = (bounds["top"] + bounds["bottom"]) / 2
        sample_geo_xy = (cx, cy)

    tf_info = None
    if config.tiles_dir:
        try:
            tf_info = get_tileset_transform(
                config.tiles_dir,
                sample_geo_xy=sample_geo_xy,
                frame_mode="auto",
            )
        except Exception as e:
            logger.error("Failed to get tileset transform: %s", e)
            return
    else:
        logger.error("tiles_dir not configured, cannot reconvert masks")
        return
    logger.info("Tileset transform: mode=%s, source=%s",
                tf_info.effective_mode, tf_info.tf_source)

    # ------------------------------------------------------------------
    # Step 1: Load all raw edited masks
    # ------------------------------------------------------------------
    raw_masks = {}
    canvas_h, canvas_w = 0, 0
    for tag in SURFACE_TAGS:
        mask_path = os.path.join(masks_dir, f"{tag}_mask.png")
        if not os.path.isfile(mask_path):
            continue
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        canvas_h, canvas_w = mask.shape[:2]
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        raw_masks[tag] = binary

    if canvas_w == 0:
        logger.warning("No mask images found in %s", masks_dir)
        return

    # ------------------------------------------------------------------
    # Step 2: Priority compositing — eliminate inter-tag overlaps.
    # SURFACE_TAGS is ordered low→high priority.  Each pixel belongs to
    # the highest-priority tag that covers it (same as Stage 5).
    # ------------------------------------------------------------------
    composite = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    tag_ids = {tag: idx + 1 for idx, tag in enumerate(SURFACE_TAGS)}

    for tag in SURFACE_TAGS:
        if tag not in raw_masks:
            continue
        composite[raw_masks[tag] > 0] = tag_ids[tag]

    composited = {}
    for tag in SURFACE_TAGS:
        if tag not in raw_masks:
            continue
        composited[tag] = np.where(composite == tag_ids[tag], 255, 0).astype(np.uint8)
        raw_pct = np.count_nonzero(raw_masks[tag]) / raw_masks[tag].size * 100
        comp_pct = np.count_nonzero(composited[tag]) / composited[tag].size * 100
        logger.info("  Composite: %s raw=%.1f%% -> composited=%.1f%%",
                     tag, raw_pct, comp_pct)

    # ------------------------------------------------------------------
    # Step 3: Write composited masks back to disk (so mask files are
    # always clean and consistent with what Stage 6 will consume).
    # ------------------------------------------------------------------
    for tag, mask in composited.items():
        cv2.imwrite(os.path.join(masks_dir, f"{tag}_mask.png"), mask)
    logger.info("Wrote %d composited masks back to disk", len(composited))

    # ------------------------------------------------------------------
    # Step 4: Extract contours, triangulate, convert to Blender coords
    # ------------------------------------------------------------------
    for tag, binary in composited.items():
        groups = extract_contours_and_triangulate(
            binary, tag, bounds, canvas_w, canvas_h,
            simplify_epsilon=2.0, min_contour_area=100,
        )

        if not groups:
            logger.info("  %s: no contours found", tag)
            continue

        mesh_groups = []
        total_verts = 0
        total_faces = 0
        for group_idx, group in enumerate(groups):
            verts_geo = group.get("vertices")
            faces = group.get("faces")
            if verts_geo is None or faces is None:
                continue
            points_xyz = geo_points_to_blender_xyz(verts_geo, tf_info, z_mode="zero")
            if len(points_xyz) < 3:
                continue
            mesh_groups.append({
                "group_index": group_idx,
                "tag": tag,
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
            "source_tag": tag,
            "mesh_groups": mesh_groups,
        }

        tag_dir = os.path.join(out_dir, tag)
        os.makedirs(tag_dir, exist_ok=True)
        out_path = os.path.join(tag_dir, f"{tag}_merged_blender.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        logger.info("  %s: %d groups, %d verts, %d faces -> %s",
                     tag, len(mesh_groups), total_verts, total_faces, out_path)

    logger.info("Reconvert complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Stage 5a: Manual surface mask management")
    p.add_argument("--geotiff", required=True, help="Path to GeoTIFF image")
    p.add_argument("--tiles-dir", default="", help="Directory with tileset.json")
    p.add_argument("--output-dir", default="output", help="Output base directory")
    args = p.parse_args()
    config = PipelineConfig(
        geotiff_path=args.geotiff,
        tiles_dir=args.tiles_dir,
        output_dir=args.output_dir,
    ).resolve()
    run(config)
