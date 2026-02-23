"""Stage 1: Convert B3DM files to GLB."""
from __future__ import annotations

import argparse
import logging
import os
import sys

logger = logging.getLogger("sam3_pipeline.s01")

# Ensure script/ is on sys.path for sibling imports
_script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from pipeline_config import PipelineConfig
from progress import ProgressTracker


def run(config: PipelineConfig) -> None:
    """Execute Stage 1: B3DM -> GLB conversion."""
    logger.info("=== Stage 1: B3DM -> GLB conversion ===")

    from b3dm_converter import convert_directory

    if not config.tiles_dir:
        logger.warning("No tiles_dir specified, skipping B3DM conversion.")
        return

    os.makedirs(config.glb_dir, exist_ok=True)
    tracker = ProgressTracker(total=1, pct_start=5, pct_end=95)

    def _on_progress(current, total):
        tracker.total = max(1, total)
        tracker.update(current, f"Converting {current}/{total}")

    converted = convert_directory(config.tiles_dir, config.glb_dir,
                                  max_workers=config.max_workers,
                                  on_progress=_on_progress)
    tracker.complete("B3DM conversion complete")
    logger.info("Converted %d B3DM files to GLB in %s", len(converted), config.glb_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Stage 1: B3DM -> GLB conversion")
    p.add_argument("--tiles-dir", required=True, help="Directory with b3dm files and tileset.json")
    p.add_argument("--output-dir", default="output", help="Output base directory")
    args = p.parse_args()
    config = PipelineConfig(tiles_dir=args.tiles_dir, output_dir=args.output_dir).resolve()
    run(config)
