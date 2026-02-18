"""Stage 7: Generate game objects using hybrid VLM + programmatic approach.

Supports multi-layout mode: if stage 2a track layouts exist, generates
independent centerline + game objects for each layout in a per-layout subdirectory.
Falls back to single-layout behaviour when no layouts are defined.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys

logger = logging.getLogger("sam3_pipeline.s07")

_script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from pipeline_config import PipelineConfig


def run(config: PipelineConfig) -> None:
    """Execute Stage 7: AI game object generation (hybrid).

    If track layouts exist (stage 2a), processes each layout independently.
    Otherwise falls back to the original single-layout flow.
    """
    logger.info("=== Stage 7: AI game object generation (hybrid) ===")

    if not config.geotiff_path:
        raise ValueError("geotiff_path is required for ai_game_objects stage")

    out_dir = config.stage_dir("ai_game_objects")
    os.makedirs(out_dir, exist_ok=True)

    # Read from 02_result junction (points to 02 or 02a)
    result_dir = config.mask_full_map_result
    if not os.path.isdir(result_dir):
        result_dir = config.mask_full_map_dir  # fallback

    # Resolve image path (prefer vlmscale > modelscale > raw geotiff)
    basename = os.path.splitext(os.path.basename(config.geotiff_path))[0]
    vlmscale_img = os.path.join(result_dir, f"{basename}_vlmscale.png")
    modelscale_img = os.path.join(result_dir, f"{basename}_modelscale.png")
    image_path = vlmscale_img if os.path.isfile(vlmscale_img) else \
                 (modelscale_img if os.path.isfile(modelscale_img) else config.geotiff_path)

    # Multi-layout mode — layouts.json lives in 02_result
    layouts_json = os.path.join(result_dir, "layouts.json")
    if os.path.isfile(layouts_json):
        layouts = _load_layouts(layouts_json)
        if layouts:
            logger.info("Multi-layout mode: %d layout(s)", len(layouts))
            layouts_dir = result_dir  # layout mask PNGs are in the same dir
            for layout in layouts:
                _generate_for_layout(config, layout, layouts_dir, out_dir, image_path,
                                     modelscale_img=modelscale_img)
            _merge_all_layouts(out_dir, config.game_objects_json)
            _write_geo_metadata(config, out_dir)
            return

    # Single-layout fallback
    _generate_single(config, out_dir, image_path, modelscale_img=modelscale_img)


def _load_layouts(layouts_json_path: str):
    """Load and return the list of layout definitions."""
    with open(layouts_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("layouts", [])


def _safe_name(name: str) -> str:
    """Convert layout name to a filesystem-safe directory name."""
    return re.sub(r'[^\w\-]', '_', name).strip('_') or "unnamed"


def _generate_for_layout(config, layout, layouts_dir, out_dir, image_path,
                         modelscale_img: str = ""):
    """Generate centerline + game objects for a single layout.

    Uses per-type VLM generation with validation + retry:
    1. VLM generates hotlap, pits, starts, timing_0 (4 separate calls)
    2. Snap TIME_0 to centerline
    3. Generate timing points from TIME_0 in driving order
    """
    import cv2
    import numpy as np
    from ai_game_objects import (
        ValidationMasks, generate_all_vlm_sequential,
    )
    from ai_visualizer import visualize_game_objects
    from road_centerline import (
        extract_centerline, compute_curvature, detect_composite_bends,
        snap_to_centerline, generate_timing_points_from_time0,
    )

    # Compute modelscale size for VLM coordinate scaling
    modelscale_size = None
    if modelscale_img and os.path.isfile(modelscale_img) and image_path != modelscale_img:
        from PIL import Image as PILImage
        with PILImage.open(modelscale_img) as ms_img:
            modelscale_size = ms_img.size  # (w, h)

    name = layout.get("name", "unnamed")
    safe = _safe_name(name)
    direction = layout.get("track_direction", config.track_direction)
    mask_file = layout.get("mask_file", "")

    layout_out = os.path.join(out_dir, safe)
    os.makedirs(layout_out, exist_ok=True)

    logger.info("--- Layout: %s (direction: %s) ---", name, direction)

    # Load layout mask
    mask_path = os.path.join(layouts_dir, mask_file) if mask_file else None
    if not mask_path or not os.path.isfile(mask_path):
        logger.warning("Layout '%s' mask not found: %s, skipping", name, mask_path)
        return

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        logger.warning("Cannot read layout mask: %s", mask_path)
        return

    # Centerline extraction from layout mask
    try:
        centerline = extract_centerline(mask)
        curvature = compute_curvature(centerline) if len(centerline) >= 10 else np.array([])
        bends = detect_composite_bends(centerline, curvature) if len(curvature) > 0 else []
    except Exception as e:
        logger.warning("Centerline extraction failed for '%s': %s", name, e)
        centerline = np.zeros((0, 2), dtype=np.float64)
        curvature = np.array([])
        bends = []

    # Load validation masks (with geo metadata for pixel_size_m)
    result_dir = config.mask_full_map_result
    if not os.path.isdir(result_dir):
        result_dir = config.mask_full_map_dir
    geo_meta_path = os.path.join(result_dir, "result_masks.json")
    masks = None
    if os.path.isfile(geo_meta_path):
        try:
            masks = ValidationMasks.load(mask_path, result_dir, geo_meta_path)
            logger.info("Validation masks loaded (pixel_size_m=%.4f)", masks.pixel_size_m)
        except Exception as e:
            logger.warning("Failed to load validation masks: %s", e)

    pixel_size_m = masks.pixel_size_m if masks else 0.3

    # Sequential VLM generation (4 per-type calls with validation + retry)
    vlm_result = {"hotlap": [], "pits": [], "starts": [], "timing_0_raw": [], "validation": {}}
    try:
        vlm_result = generate_all_vlm_sequential(
            image_path, mask_path, direction, masks,
            api_key=config.gemini_api_key, model_name=config.gemini_model,
            modelscale_size=modelscale_size,
        )
        logger.info("VLM validation: %s", json.dumps(vlm_result.get("validation", {})))
    except Exception as e:
        logger.warning("VLM generation failed for '%s': %s", name, e)

    # Snap hotlap_start to centerline for precise position + heading
    if len(centerline) >= 10 and vlm_result.get("hotlap"):
        hotlap = vlm_result["hotlap"][0]
        vlm_pos = hotlap["position"]
        idx, snapped = snap_to_centerline(vlm_pos, centerline)
        # Compute tangent direction from centerline neighbors
        n = len(centerline)
        fwd = 1 if direction == "CW" else -1  # index advance direction
        i_next = (idx + fwd) % n
        i_prev = (idx - fwd) % n
        tangent = centerline[i_next] - centerline[i_prev]
        norm = float(np.linalg.norm(tangent))
        if norm > 0:
            tangent = tangent / norm
        hotlap["position"] = snapped
        hotlap["orientation_z"] = tangent.tolist()
        logger.info(
            "HOTLAP_START snapped to centerline: VLM %s → [%.1f, %.1f], "
            "heading [%.3f, %.3f] (idx %d)",
            vlm_pos, snapped[0], snapped[1],
            tangent[0], tangent[1], idx,
        )

    # Snap TIME_0 to centerline and generate timing points
    timing_objs = []
    labeled_bends = bends  # fallback
    if len(centerline) >= 10:
        time0_raw = vlm_result.get("timing_0_raw", [])
        if time0_raw:
            time0_pos = time0_raw[0]["position"]
            time0_idx, _ = snap_to_centerline(time0_pos, centerline)
            logger.info("TIME_0 VLM position %s snapped to centerline idx %d", time0_pos, time0_idx)
        else:
            time0_idx = 0
            logger.warning("No TIME_0 from VLM, using centerline index 0")

        try:
            timing_objs, labeled_bends = generate_timing_points_from_time0(
                mask, centerline, curvature, bends, time0_idx,
                direction, pixel_size_m,
            )
        except Exception as e:
            logger.warning("Timing generation from TIME_0 failed: %s", e)

    # Merge all objects
    all_objects = vlm_result["hotlap"] + vlm_result["pits"] + vlm_result["starts"] + timing_objs

    # Save centerline (with labeled bends including turn_label)
    centerline_out = {
        "layout_name": name,
        "centerline": centerline.tolist(),
        "bends": labeled_bends,
        "edited": False,
        "track_direction": direction,
    }
    if len(centerline) >= 10 and vlm_result.get("timing_0_raw"):
        centerline_out["time0_idx"] = time0_idx

    cl_path = os.path.join(layout_out, "centerline.json")
    with open(cl_path, "w", encoding="utf-8") as f:
        json.dump(centerline_out, f, indent=2, ensure_ascii=False)
    logger.info("Centerline saved: %s (%d points, %d bends)",
                cl_path, len(centerline), len(labeled_bends))

    # Save game objects
    game_obj_data = {
        "layout_name": name,
        "track_direction": direction,
        "objects": all_objects,
    }
    if vlm_result.get("validation"):
        game_obj_data["_validation"] = vlm_result["validation"]

    go_path = os.path.join(layout_out, "game_objects.json")
    with open(go_path, "w", encoding="utf-8") as f:
        json.dump(game_obj_data, f, indent=2, ensure_ascii=False)
    logger.info("Game objects saved: %s (%d objects)", go_path, len(all_objects))

    # Preview — use modelscale image (coordinates are in modelscale space)
    preview_image = modelscale_img if modelscale_img and os.path.isfile(modelscale_img) else image_path
    try:
        preview_path = os.path.join(layout_out, "preview.png")
        visualize_game_objects(
            preview_image, game_obj_data, preview_path,
            centerline=centerline.tolist(),
            bends=labeled_bends,
        )
        logger.info("Preview saved: %s", preview_path)
    except Exception as e:
        logger.warning("Preview failed for '%s' (non-critical): %s", name, e)


def _merge_all_layouts(out_dir, game_objects_json_path):
    """Merge all per-layout game objects into the top-level game_objects.json."""
    all_objects = []
    layout_names = []

    for entry in sorted(os.listdir(out_dir)):
        go_path = os.path.join(out_dir, entry, "game_objects.json")
        if os.path.isfile(go_path):
            with open(go_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            layout_name = data.get("layout_name", entry)
            layout_names.append(layout_name)
            for obj in data.get("objects", []):
                obj["_layout"] = layout_name
                all_objects.append(obj)

    merged = {
        "track_direction": "clockwise",
        "layouts": layout_names,
        "objects": all_objects,
    }

    with open(game_objects_json_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    logger.info("Merged game_objects.json: %d objects from %d layouts",
                len(all_objects), len(layout_names))


def _generate_single(config, out_dir, image_path, modelscale_img: str = ""):
    """Original single-layout generation (backward-compatible)."""
    from ai_game_objects import generate_game_objects
    from ai_visualizer import visualize_game_objects

    result_dir = config.mask_full_map_result
    if not os.path.isdir(result_dir):
        result_dir = config.mask_full_map_dir

    merged_mask = os.path.join(result_dir, "merged_mask.png")
    mask_path = merged_mask if os.path.isfile(merged_mask) else None

    road_mask_path = None
    road_mask_candidate = os.path.join(result_dir, "road_mask.png")
    if os.path.isfile(road_mask_candidate):
        road_mask_path = road_mask_candidate
    elif mask_path:
        road_mask_path = mask_path

    objects_data = generate_game_objects(
        image_path=image_path,
        mask_path=mask_path,
        road_mask_path=road_mask_path,
        track_direction=config.track_direction,
        api_key=config.gemini_api_key,
        model_name=config.gemini_model,
    )

    # Extract and save centerline data separately
    centerline_data = objects_data.pop("_centerline_data", None)
    if centerline_data:
        centerline_path = os.path.join(out_dir, "centerline.json")
        with open(centerline_path, "w", encoding="utf-8") as f:
            json.dump(centerline_data, f, indent=2, ensure_ascii=False)
        logger.info("Centerline JSON saved: %s", centerline_path)

    with open(config.game_objects_json, "w", encoding="utf-8") as f:
        json.dump(objects_data, f, indent=2, ensure_ascii=False)
    logger.info("Game objects JSON saved: %s", config.game_objects_json)

    _write_geo_metadata(config, out_dir)

    # Use modelscale image for preview (coordinates are in modelscale space)
    preview_image = modelscale_img if modelscale_img and os.path.isfile(modelscale_img) else image_path
    try:
        centerline_pts = centerline_data.get("centerline") if centerline_data else None
        bends_data = centerline_data.get("bends") if centerline_data else None
        visualize_game_objects(preview_image, objects_data, config.game_objects_preview,
                               centerline=centerline_pts, bends=bends_data)
        logger.info("Game objects preview saved: %s", config.game_objects_preview)
    except Exception as e:
        logger.warning("Game objects preview failed (non-critical): %s", e)


def _write_geo_metadata(config: PipelineConfig, out_dir: str) -> None:
    """Write geo_metadata.json from result_masks.json for the object editor."""
    result_dir = config.mask_full_map_result
    if not os.path.isdir(result_dir):
        result_dir = config.mask_full_map_dir
    masks_json = os.path.join(result_dir, "result_masks.json")
    if not os.path.isfile(masks_json):
        stage6_meta = os.path.join(config.stage_dir("ai_walls"), "geo_metadata.json")
        if os.path.isfile(stage6_meta):
            import shutil
            out_path = os.path.join(out_dir, "geo_metadata.json")
            shutil.copy2(stage6_meta, out_path)
            logger.info("Geo metadata copied from stage 6: %s", out_path)
            return
        logger.warning("No geo metadata source found, skipping geo_metadata.json")
        return

    try:
        with open(masks_json, "r", encoding="utf-8") as f:
            masks_data = json.load(f)

        meta = masks_data.get("meta", {})
        model_scale = meta.get("model_scale_size", {})
        geo_bounds = meta.get("geo", {}).get("bounds", {})

        geo_metadata = {
            "image_width": model_scale.get("width", 0),
            "image_height": model_scale.get("height", 0),
            "bounds": {
                "north": geo_bounds.get("top", 0),
                "south": geo_bounds.get("bottom", 0),
                "east": geo_bounds.get("right", 0),
                "west": geo_bounds.get("left", 0),
            },
        }

        out_path = os.path.join(out_dir, "geo_metadata.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(geo_metadata, f, indent=2, ensure_ascii=False)
        logger.info("Geo metadata saved: %s", out_path)
    except Exception as e:
        logger.warning("geo_metadata.json generation failed (non-critical): %s", e)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Stage 7: AI game object generation")
    p.add_argument("--geotiff", required=True, help="Path to GeoTIFF image")
    p.add_argument("--output-dir", default="output", help="Output base directory")
    p.add_argument("--gemini-api-key", default="", help="Gemini API key")
    args = p.parse_args()
    config = PipelineConfig(geotiff_path=args.geotiff, output_dir=args.output_dir).resolve()
    if args.gemini_api_key:
        config.gemini_api_key = args.gemini_api_key
    run(config)
