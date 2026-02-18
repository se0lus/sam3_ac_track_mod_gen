"""Stage 6: Generate virtual wall data using programmatic mask contours."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

logger = logging.getLogger("sam3_pipeline.s06")

_script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from pipeline_config import PipelineConfig


def run(config: PipelineConfig) -> None:
    """Execute Stage 6: AI wall generation.

    Reads modelscale image from ``config.mask_full_map_dir`` (stage 2 output),
    uses trees/grass masks for precise wall placement,
    writes walls JSON + preview to ``config.stage_dir("ai_walls")``.
    """
    logger.info("=== Stage 6: AI wall generation ===")

    from ai_wall_generator import generate_walls
    from ai_visualizer import visualize_walls

    if not config.geotiff_path:
        raise ValueError("geotiff_path is required for ai_walls stage")

    out_dir = config.stage_dir("ai_walls")
    os.makedirs(out_dir, exist_ok=True)

    # Read from 02_result junction (points to 02 or 02a)
    result_dir = config.mask_full_map_result
    if not os.path.isdir(result_dir):
        result_dir = config.mask_full_map_dir  # fallback

    basename = os.path.splitext(os.path.basename(config.geotiff_path))[0]
    modelscale_img = os.path.join(result_dir, f"{basename}_modelscale.png")
    image_path = modelscale_img if os.path.isfile(modelscale_img) else config.geotiff_path

    merged_mask = os.path.join(result_dir, "merged_mask.png")
    mask_path = merged_mask if os.path.isfile(merged_mask) else None

    # Check for SAM3 masks from result directory
    mask_names = ["trees", "grass", "kerb", "sand", "building", "water", "concrete"]
    mask_paths = {}
    for name in mask_names:
        path = os.path.join(result_dir, f"{name}_mask.png")
        mask_paths[name] = path if os.path.isfile(path) else None
        if mask_paths[name]:
            logger.info("Using %s mask: %s", name, mask_paths[name])
        else:
            logger.info("No %s mask found", name)

    walls_data = generate_walls(
        image_path=image_path,
        mask_image_path=mask_path,
        trees_mask_path=mask_paths["trees"],
        grass_mask_path=mask_paths["grass"],
        kerb_mask_path=mask_paths["kerb"],
        sand_mask_path=mask_paths["sand"],
        building_mask_path=mask_paths["building"],
        water_mask_path=mask_paths["water"],
        concrete_mask_path=mask_paths["concrete"],
        output_dir=out_dir,
    )

    with open(config.walls_json, "w", encoding="utf-8") as f:
        json.dump(walls_data, f, indent=2, ensure_ascii=False)
    logger.info("Wall JSON saved: %s", config.walls_json)

    try:
        visualize_walls(image_path, walls_data, config.walls_preview)
        logger.info("Wall preview saved: %s", config.walls_preview)
    except Exception as e:
        logger.warning("Wall preview failed (non-critical): %s", e)

    # Generate geo_metadata.json for wall editor
    _write_geo_metadata(config, out_dir)


def _write_geo_metadata(config: PipelineConfig, out_dir: str) -> None:
    """Write geo_metadata.json from result_masks.json for the wall editor."""
    result_dir = config.mask_full_map_result
    if not os.path.isdir(result_dir):
        result_dir = config.mask_full_map_dir
    masks_json = os.path.join(result_dir, "result_masks.json")
    if not os.path.isfile(masks_json):
        logger.warning("result_masks.json not found, skipping geo_metadata.json")
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
    p = argparse.ArgumentParser(description="Stage 6: AI wall generation")
    p.add_argument("--geotiff", required=True, help="Path to GeoTIFF image")
    p.add_argument("--output-dir", default="output", help="Output base directory")
    p.add_argument("--gemini-api-key", default="", help="Gemini API key")
    args = p.parse_args()
    config = PipelineConfig(geotiff_path=args.geotiff, output_dir=args.output_dir).resolve()
    if args.gemini_api_key:
        config.gemini_api_key = args.gemini_api_key
    run(config)
