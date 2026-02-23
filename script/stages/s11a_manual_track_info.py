"""Stage 11a: Initialize manual track info editing from stage 11 output.

Copies cameras.ini (per-layout) into 11a_manual_track_info/ for manual editing.
Existing edits are preserved (never overwritten).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from typing import Dict, List

logger = logging.getLogger("sam3_pipeline.s11a")

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


def _layout_short_name(layout: Dict) -> str:
    """Derive a folder name from the layout name, lowercased."""
    name = layout.get("name", "default")
    return name.lower()


def _find_track_folder(packaging_dir: str) -> str:
    """Find the track folder name inside the packaging directory.

    Stage 11 creates output/{track_folder}/ inside 11_track_packaging/.
    We look for the first directory that contains layout sub-folders.
    """
    if not os.path.isdir(packaging_dir):
        return ""
    for entry in os.listdir(packaging_dir):
        candidate = os.path.join(packaging_dir, entry)
        if os.path.isdir(candidate) and not entry.startswith("."):
            return entry
    return ""


def _load_layouts(config: PipelineConfig) -> List[Dict]:
    """Load layout list from track_layouts_json."""
    layouts_json = config.track_layouts_json
    if os.path.isfile(layouts_json):
        with open(layouts_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        layouts = data.get("layouts", [])
        if layouts:
            return layouts
    return [{"name": "Default", "track_direction": config.track_direction}]


def run(config: PipelineConfig) -> None:
    """Initialize 11a_manual_track_info from stage 11 output."""
    logger.info("=== Stage 11a: Initialize manual track info editing ===")

    packaging_dir = config.packaging_dir  # 11_track_packaging
    dst_dir = config.stage_dir("manual_track_info")  # 11a_manual_track_info
    os.makedirs(dst_dir, exist_ok=True)

    # Find track folder inside packaging dir
    track_folder = _find_track_folder(packaging_dir)
    if not track_folder:
        logger.warning("No track folder found in %s — skipping camera copy", packaging_dir)

    layouts = _load_layouts(config)
    logger.info("Layouts: %s", [l["name"] for l in layouts])

    for layout in layouts:
        short = _layout_short_name(layout)

        # Copy cameras.ini from stage 11 packaging output
        if track_folder:
            src_cameras = os.path.join(
                packaging_dir, track_folder, short, "data", "cameras.ini"
            )
            dst_cameras = os.path.join(dst_dir, short, "cameras.ini")
            if _copy_if_missing(src_cameras, dst_cameras):
                logger.info("Copied cameras.ini for layout '%s'", short)
            elif not os.path.isfile(src_cameras):
                logger.info("No cameras.ini for layout '%s' in stage 11", short)

        # Copy centerline.json (read-only reference for the editor)
        game_objects_result = config.game_objects_result_dir
        src_cl = os.path.join(game_objects_result, layout["name"], "centerline.json")
        if not os.path.isfile(src_cl):
            # Fallback: try direct stage 7 directory
            src_cl = os.path.join(
                config.stage_dir("ai_game_objects"), layout["name"], "centerline.json"
            )
        dst_cl = os.path.join(dst_dir, short, "centerline.json")
        if _copy_if_missing(src_cl, dst_cl):
            logger.info("Copied centerline.json for layout '%s'", short)

    logger.info("Stage 11a complete: %s", dst_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )
    p = argparse.ArgumentParser(description="Stage 11a: Initialize manual track info editing")
    p.add_argument("--output-dir", default="output", help="Output base directory")
    args = p.parse_args()
    config = PipelineConfig(output_dir=args.output_dir).resolve()
    run(config)
