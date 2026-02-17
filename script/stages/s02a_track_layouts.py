"""Stage 2a: Track layout mask management (optional manual step).

Users create per-layout binary masks via the web-based layout editor.
This stage validates existing layouts and initialises the output directory.
It does NOT participate in the automatic pipeline — run it manually or
use the layout editor directly.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys

logger = logging.getLogger("sam3_pipeline.s02a")

_script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from pipeline_config import PipelineConfig


def run(config: PipelineConfig) -> None:
    """Initialise / validate the track layouts directory."""
    logger.info("=== Stage 2a: Track layout management ===")

    layouts_dir = config.stage_dir("track_layouts")
    os.makedirs(layouts_dir, exist_ok=True)

    # Copy geo_metadata from stage 2 if not present
    geo_meta_dst = os.path.join(layouts_dir, "geo_metadata.json")
    if not os.path.isfile(geo_meta_dst):
        # Try stage 7, then stage 8 sources
        for src_dir in [config.stage_dir("ai_walls"),
                        config.stage_dir("ai_game_objects")]:
            src = os.path.join(src_dir, "geo_metadata.json")
            if os.path.isfile(src):
                shutil.copy2(src, geo_meta_dst)
                logger.info("Copied geo_metadata from %s", src)
                break

    # Initialise layouts.json if missing
    if not os.path.isfile(config.track_layouts_json):
        initial = {"layouts": []}
        with open(config.track_layouts_json, "w", encoding="utf-8") as f:
            json.dump(initial, f, indent=2, ensure_ascii=False)
        logger.info("Created empty layouts.json at %s", config.track_layouts_json)
    else:
        _validate_layouts(config.track_layouts_json, layouts_dir)

    logger.info("Track layouts directory ready: %s", layouts_dir)


def _validate_layouts(layouts_json_path: str, layouts_dir: str) -> None:
    """Validate layouts.json and check that referenced mask files exist."""
    with open(layouts_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    layouts = data.get("layouts", [])
    logger.info("Found %d layout(s)", len(layouts))

    for i, layout in enumerate(layouts):
        name = layout.get("name", f"unnamed_{i}")
        mask_file = layout.get("mask_file", "")
        if not mask_file:
            logger.warning("Layout '%s' has no mask_file", name)
            continue
        mask_path = os.path.join(layouts_dir, mask_file)
        if not os.path.isfile(mask_path):
            logger.warning("Layout '%s' mask file not found: %s", name, mask_path)
        else:
            logger.info("Layout '%s': mask OK (%s)", name, mask_file)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Stage 2a: Track layout management")
    p.add_argument("--geotiff", required=True, help="Path to GeoTIFF image")
    p.add_argument("--output-dir", default="output", help="Output base directory")
    args = p.parse_args()
    config = PipelineConfig(geotiff_path=args.geotiff, output_dir=args.output_dir).resolve()
    run(config)
