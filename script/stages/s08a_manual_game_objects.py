"""Stage 8a: Initialize manual game objects directory from stage 8 output.

Copies stage 8 results into 08a_manual_game_objects/ for manual editing.
Existing edits in 8a are preserved (never overwritten).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys

logger = logging.getLogger("sam3_pipeline.s08a")

_script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from pipeline_config import PipelineConfig


def _safe_name(name: str) -> str:
    return re.sub(r'[^\w\-]', '_', name).strip('_') or "unnamed"


def _copy_if_exists(src_dir: str, dst_dir: str, filename: str) -> bool:
    src = os.path.join(src_dir, filename)
    if os.path.isfile(src):
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy2(src, os.path.join(dst_dir, filename))
        return True
    return False


def _load_layouts(layouts_json_path: str):
    if not os.path.isfile(layouts_json_path):
        return []
    with open(layouts_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("layouts", [])


def _merge_all_layouts(dst_dir: str) -> None:
    """Merge all per-layout game objects into the top-level game_objects.json."""
    all_objects = []
    layout_names = []

    for entry in sorted(os.listdir(dst_dir)):
        go_path = os.path.join(dst_dir, entry, "game_objects.json")
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

    merged_path = os.path.join(dst_dir, "game_objects.json")
    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    logger.info("Merged game_objects.json: %d objects from %d layouts",
                len(all_objects), len(layout_names))


def run(config: PipelineConfig) -> None:
    """Initialize 08a_manual_game_objects from stage 8 output."""
    logger.info("=== Stage 8a: Initialize manual game objects ===")

    src_dir = config.stage_dir("ai_game_objects")
    dst_dir = config.stage_dir("manual_game_objects")
    os.makedirs(dst_dir, exist_ok=True)

    # Always overwrite geo_metadata.json (metadata, not user edits)
    _copy_if_exists(src_dir, dst_dir, "geo_metadata.json")

    # Per-layout: only copy if layout subdir doesn't exist in 8a yet
    layouts = _load_layouts(config.track_layouts_json)
    if layouts:
        copied = 0
        skipped = 0
        for layout in layouts:
            safe = _safe_name(layout.get("name", "unnamed"))
            src_sub = os.path.join(src_dir, safe)
            dst_sub = os.path.join(dst_dir, safe)

            if os.path.isdir(dst_sub):
                logger.info("Layout '%s' already exists in 8a, preserving edits", safe)
                skipped += 1
                continue

            if not os.path.isdir(src_sub):
                logger.warning("Layout '%s' not found in stage 8 output, skipping", safe)
                continue

            os.makedirs(dst_sub, exist_ok=True)
            for f in ["centerline.json", "game_objects.json"]:
                _copy_if_exists(src_sub, dst_sub, f)
            copied += 1

        logger.info("Copied %d layout(s), preserved %d existing", copied, skipped)
    else:
        # Single-layout fallback
        for f in ["centerline.json", "game_objects.json"]:
            if not os.path.isfile(os.path.join(dst_dir, f)):
                _copy_if_exists(src_dir, dst_dir, f)

    _merge_all_layouts(dst_dir)
    logger.info("Stage 8a complete: %s", dst_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Stage 8a: Initialize manual game objects")
    p.add_argument("--output-dir", default="output", help="Output base directory")
    args = p.parse_args()
    config = PipelineConfig(output_dir=args.output_dir).resolve()
    run(config)
