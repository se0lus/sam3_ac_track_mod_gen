"""Stage 8: Generate game objects using Gemini AI."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

logger = logging.getLogger("sam3_pipeline.s08")

_script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from pipeline_config import PipelineConfig


def run(config: PipelineConfig) -> None:
    """Execute Stage 8: AI game object generation.

    Reads modelscale image from ``config.mask_full_map_dir`` (stage 2 output),
    writes game objects JSON + preview to ``config.stage_dir("ai_game_objects")``.
    """
    logger.info("=== Stage 8: AI game object generation ===")

    from ai_game_objects import generate_game_objects
    from ai_visualizer import visualize_game_objects

    if not config.geotiff_path:
        raise ValueError("geotiff_path is required for ai_game_objects stage")

    out_dir = config.stage_dir("ai_game_objects")
    os.makedirs(out_dir, exist_ok=True)

    # Read modelscale image from stage 2 output dir (not source dir)
    basename = os.path.splitext(os.path.basename(config.geotiff_path))[0]
    modelscale_img = os.path.join(config.mask_full_map_dir, f"{basename}_modelscale.png")
    image_path = modelscale_img if os.path.isfile(modelscale_img) else config.geotiff_path

    mask_path = config.mask_image_path if os.path.isfile(config.mask_image_path) else None

    objects_data = generate_game_objects(
        image_path=image_path,
        mask_path=mask_path,
        track_direction=config.track_direction,
        api_key=config.gemini_api_key,
        model_name=config.gemini_model,
    )

    with open(config.game_objects_json, "w", encoding="utf-8") as f:
        json.dump(objects_data, f, indent=2, ensure_ascii=False)
    logger.info("Game objects JSON saved: %s", config.game_objects_json)

    try:
        visualize_game_objects(image_path, objects_data, config.game_objects_preview)
        logger.info("Game objects preview saved: %s", config.game_objects_preview)
    except Exception as e:
        logger.warning("Game objects preview failed (non-critical): %s", e)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Stage 8: AI game object generation")
    p.add_argument("--geotiff", required=True, help="Path to GeoTIFF image")
    p.add_argument("--output-dir", default="output", help="Output base directory")
    p.add_argument("--gemini-api-key", default="", help="Gemini API key")
    args = p.parse_args()
    config = PipelineConfig(geotiff_path=args.geotiff, output_dir=args.output_dir).resolve()
    if args.gemini_api_key:
        config.gemini_api_key = args.gemini_api_key
    run(config)
