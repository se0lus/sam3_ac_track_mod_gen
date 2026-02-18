"""Stage 2a: Track layout mask management (optional manual step).

Users create per-layout binary masks via the web-based layout editor.
This stage initialises the output directory by copying ALL of Stage 2's
output (masks, images, metadata) so that 02_result junction can point
here and downstream stages see a complete, format-compatible directory.
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
    """Initialise / validate the track layouts directory.

    Copies all files from stage 2 output into 02a, preserving any
    existing user-edited files (layouts.json, layout mask PNGs).
    """
    logger.info("=== Stage 2a: Track layout management ===")

    stage2_dir = config.mask_full_map_dir
    layouts_dir = config.stage_dir("track_layouts")
    os.makedirs(layouts_dir, exist_ok=True)

    if not os.path.isdir(stage2_dir):
        logger.warning("Stage 2 output not found: %s. Run stage 2 first.", stage2_dir)
        return

    # Copy all files from stage 2, but never overwrite existing files
    copied = 0
    skipped = 0
    for fname in os.listdir(stage2_dir):
        src = os.path.join(stage2_dir, fname)
        dst = os.path.join(layouts_dir, fname)
        if not os.path.isfile(src):
            continue
        if os.path.isfile(dst):
            skipped += 1
            continue
        shutil.copy2(src, dst)
        copied += 1
    logger.info("Copied %d files from stage 2, skipped %d existing", copied, skipped)

    # Validate layouts.json if present
    layouts_json = os.path.join(layouts_dir, "layouts.json")
    if os.path.isfile(layouts_json):
        _validate_layouts(layouts_json, layouts_dir)
    else:
        logger.info("No layouts.json yet — use the layout editor to create layouts")

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
