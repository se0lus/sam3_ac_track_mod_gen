"""Stage 5a: Manual surface mask editing (optional step).

Users refine per-tag surface masks via the web-based surface editor.
This stage initialises the output directory by copying Stage 5's composited
merge-preview masks as defaults (full canvas precision, no resize loss).
It does NOT participate in the automatic pipeline -- run it manually or
use the surface editor directly.

Editable surface tags: road, grass, sand, kerb, road2.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys

import math

import cv2

logger = logging.getLogger("sam3_pipeline.s05a")

TILE_SIZE = 512

_script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from pipeline_config import PipelineConfig

SURFACE_TAGS = ["road", "grass", "sand", "kerb", "road2"]

TAG_COLORS = {
    "road":  "#666666",
    "grass": "#00c800",
    "sand":  "#c8c864",
    "kerb":  "#ff0000",
    "road2": "#b4b4b4",
}

TAG_LABELS = {
    "road":  "路面",
    "grass": "草地",
    "sand":  "砂石",
    "kerb":  "路缘",
    "road2": "次路面",
}


def run(config: PipelineConfig) -> None:
    """Initialise the manual surface masks directory."""
    logger.info("=== Stage 5a: Manual surface mask management ===")

    out_dir = config.stage_dir("manual_surface_masks")
    os.makedirs(out_dir, exist_ok=True)

    # Source: Stage 5 merge_preview (full canvas precision)
    stage5_preview_dir = os.path.join(
        config.blender_clips_dir, "merge_preview",
    )
    if not os.path.isdir(stage5_preview_dir):
        logger.error(
            "Stage 5 merge_preview not found: %s. Run stage 5 first.",
            stage5_preview_dir,
        )
        return

    # Detect canvas dimensions from first available merged mask
    canvas_w, canvas_h = 0, 0
    for tag in SURFACE_TAGS:
        src = os.path.join(stage5_preview_dir, f"{tag}_merged.png")
        if os.path.isfile(src):
            m = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
            if m is not None:
                canvas_h, canvas_w = m.shape[:2]
                break
    if canvas_w == 0:
        logger.error("No merged mask found in %s", stage5_preview_dir)
        return
    logger.info("Canvas dimensions: %dx%d", canvas_w, canvas_h)

    # Copy each tag's composited mask from Stage 5 (only if not already present)
    for tag in SURFACE_TAGS:
        dst = os.path.join(out_dir, f"{tag}_mask.png")
        if os.path.isfile(dst):
            logger.info("  %s_mask.png already exists, skipping", tag)
            continue

        src = os.path.join(stage5_preview_dir, f"{tag}_merged.png")
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            logger.info("  Copied %s_merged.png -> %s_mask.png", tag, tag)
        else:
            logger.warning("  %s_merged.png not found in Stage 5 preview", tag)

    # Create / update surface_masks.json (includes canvas dimensions + tile info)
    grid_cols = math.ceil(canvas_w / TILE_SIZE)
    grid_rows = math.ceil(canvas_h / TILE_SIZE)
    surface_meta = {
        "image_width": canvas_w,
        "image_height": canvas_h,
        "tile_size": TILE_SIZE,
        "grid_cols": grid_cols,
        "grid_rows": grid_rows,
        "geotiff_path": config.geotiff_path,
        "tags": [
            {
                "tag": tag,
                "color": TAG_COLORS[tag],
                "label": TAG_LABELS[tag],
                "mask_file": f"{tag}_mask.png",
            }
            for tag in SURFACE_TAGS
        ],
    }
    surface_json = os.path.join(out_dir, "surface_masks.json")
    with open(surface_json, "w", encoding="utf-8") as f:
        json.dump(surface_meta, f, indent=2, ensure_ascii=False)
    logger.info("Wrote surface_masks.json with %d tags", len(SURFACE_TAGS))

    logger.info("Manual surface masks directory ready: %s", out_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Stage 5a: Manual surface mask management")
    p.add_argument("--geotiff", required=True, help="Path to GeoTIFF image")
    p.add_argument("--output-dir", default="output", help="Output base directory")
    args = p.parse_args()
    config = PipelineConfig(geotiff_path=args.geotiff, output_dir=args.output_dir).resolve()
    run(config)
