"""Stage 6a: Initialize manual walls directory from stage 6 output.

Copies stage 6 results into 06a_manual_walls/ for manual editing via the
wall editor.  Existing edits in 6a are preserved (never overwritten).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys

logger = logging.getLogger("sam3_pipeline.s06a")

_script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from pipeline_config import PipelineConfig


def _copy_if_missing(src: str, dst: str) -> bool:
    """Copy src to dst only if dst does not already exist."""
    if os.path.isfile(dst):
        return False
    if not os.path.isfile(src):
        return False
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    return True


def run(config: PipelineConfig) -> None:
    """Initialize 06a_manual_walls from stage 6 output."""
    logger.info("=== Stage 6a: Initialize manual walls ===")

    src_dir = config.stage_dir("ai_walls")
    dst_dir = config.stage_dir("manual_walls")
    os.makedirs(dst_dir, exist_ok=True)

    # Always overwrite geo_metadata.json (metadata, not user edits)
    src_geo = os.path.join(src_dir, "geo_metadata.json")
    dst_geo = os.path.join(dst_dir, "geo_metadata.json")
    if os.path.isfile(src_geo):
        shutil.copy2(src_geo, dst_geo)
        logger.info("Copied geo_metadata.json")

    # walls.json: only copy if not already present in 7a (preserve edits)
    src_walls = os.path.join(src_dir, "walls.json")
    dst_walls = os.path.join(dst_dir, "walls.json")
    if _copy_if_missing(src_walls, dst_walls):
        logger.info("Copied walls.json from stage 6")
    elif os.path.isfile(dst_walls):
        logger.info("walls.json already exists in 6a, preserving edits")
    else:
        logger.warning("No walls.json found in stage 6 output")

    logger.info("Stage 6a complete: %s", dst_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Stage 6a: Initialize manual walls")
    p.add_argument("--output-dir", default="output", help="Output base directory")
    args = p.parse_args()
    config = PipelineConfig(output_dir=args.output_dir).resolve()
    run(config)
