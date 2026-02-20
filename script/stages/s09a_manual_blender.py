"""Stage 9a: Initialize manual Blender editing directory from stage 9 output.

Copies final_track.blend + texture/ into 09a_manual_blender/ for manual editing
in Blender.  Existing edits are preserved (never overwritten).
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys

logger = logging.getLogger("sam3_pipeline.s09a")

_script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from pipeline_config import PipelineConfig


def _copy_if_missing(src: str, dst: str) -> bool:
    """Copy src to dst only if dst does not already exist."""
    if os.path.isfile(dst):
        logger.info("  already exists, preserving edits: %s", os.path.basename(dst))
        return False
    if not os.path.isfile(src):
        return False
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _copy_dir_if_missing(src_dir: str, dst_dir: str) -> int:
    """Recursively copy directory contents, skipping files that already exist.

    Returns the number of files copied.
    """
    if not os.path.isdir(src_dir):
        return 0
    copied = 0
    for root, dirs, files in os.walk(src_dir):
        rel = os.path.relpath(root, src_dir)
        dst_root = os.path.join(dst_dir, rel) if rel != "." else dst_dir
        os.makedirs(dst_root, exist_ok=True)
        for f in files:
            src_file = os.path.join(root, f)
            dst_file = os.path.join(dst_root, f)
            if not os.path.isfile(dst_file):
                shutil.copy2(src_file, dst_file)
                copied += 1
            else:
                logger.info("  already exists, preserving edits: %s", os.path.relpath(dst_file, dst_dir))
    return copied


def run(config: PipelineConfig) -> None:
    """Initialize 09a_manual_blender from stage 9 output."""
    logger.info("=== Stage 9a: Initialize manual Blender editing ===")

    src_dir = config.stage_dir("blender_automate")
    dst_dir = config.stage_dir("manual_blender")
    os.makedirs(dst_dir, exist_ok=True)

    # Copy final_track.blend (preserve existing)
    src_blend = os.path.join(src_dir, "final_track.blend")
    dst_blend = os.path.join(dst_dir, "final_track.blend")
    if _copy_if_missing(src_blend, dst_blend):
        logger.info("Copied final_track.blend from stage 9")
    elif not os.path.isfile(src_blend):
        logger.warning("No final_track.blend found in stage 9 output")

    # Copy texture/ directory (preserve existing files)
    src_tex = os.path.join(src_dir, "texture")
    dst_tex = os.path.join(dst_dir, "texture")
    n = _copy_dir_if_missing(src_tex, dst_tex)
    if n > 0:
        logger.info("Copied %d texture files from stage 9", n)
    elif os.path.isdir(dst_tex):
        logger.info("texture/ already exists, preserving edits")
    elif not os.path.isdir(src_tex):
        logger.info("No texture/ directory in stage 9 output (skipped)")

    logger.info("Stage 9a complete: %s", dst_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Stage 9a: Initialize manual Blender editing")
    p.add_argument("--output-dir", default="output", help="Output base directory")
    args = p.parse_args()
    config = PipelineConfig(output_dir=args.output_dir).resolve()
    run(config)
