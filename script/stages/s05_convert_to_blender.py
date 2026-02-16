"""Stage 5: Convert masks to Blender input coordinates + consolidation."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

logger = logging.getLogger("sam3_pipeline.s05")

_script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from pipeline_config import PipelineConfig


def run(config: PipelineConfig) -> None:
    """Execute Stage 5: Convert masks to Blender input.

    Reads ``*_masks.json`` from ``config.mask_on_clips_dir`` (stage 4 output),
    writes blender JSON + consolidated per-tag files to ``config.blender_clips_dir``.
    """
    logger.info("=== Stage 5: Convert masks to Blender input ===")

    if not config.tiles_dir:
        raise ValueError("tiles_dir is required for convert_to_blender stage")

    # Walk stage 4's output for mask JSONs (not the source geotiff directory)
    mask_search_dir = config.mask_on_clips_dir
    if not os.path.isdir(mask_search_dir):
        raise ValueError(
            f"mask_on_clips_dir not found: {mask_search_dir}. Run stage 4 first."
        )

    _convert_mask_to_blender_input(
        mask_json_file_path=mask_search_dir,
        tiles_json_path=config.tiles_dir,
        output_path=config.blender_clips_dir,
    )
    logger.info("Blender input files written to %s", config.blender_clips_dir)


def _convert_mask_to_blender_input(
    mask_json_file_path: str,
    tiles_json_path: str,
    output_path: str,
) -> None:
    """Convert all mask JSON files to Blender coordinate input, then consolidate."""
    from geo_sam3_blender_utils import map_mask_to_blender, consolidate_clips_by_tag

    os.makedirs(output_path, exist_ok=True)

    mask_json_files = []
    for root, dirs, files in os.walk(mask_json_file_path):
        for f in files:
            if f.endswith('_masks.json'):
                mask_json_files.append(os.path.join(root, f))

    if not mask_json_files:
        logger.warning("No *_masks.json files found in %s", mask_json_file_path)
        return

    logger.info("Found %d mask JSON files", len(mask_json_files))

    for mask_json_file in mask_json_files:
        try:
            result = map_mask_to_blender(mask_json_file, tiles_json_path, z_mode="zero", frame_mode="auto")

            rel_path = os.path.relpath(mask_json_file, mask_json_file_path)
            rel_dir = os.path.dirname(rel_path)
            base_name = os.path.basename(mask_json_file)
            output_filename = base_name.replace('_masks.json', '_blender.json')

            if rel_dir and rel_dir != '.':
                output_dir = os.path.join(output_path, rel_dir)
            else:
                output_dir = output_path
            os.makedirs(output_dir, exist_ok=True)

            output_file = os.path.join(output_dir, output_filename)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info("  Saved: %s", output_file)
        except Exception as e:
            logger.error("Error processing %s: %s", mask_json_file, e)
            continue

    consolidated = consolidate_clips_by_tag(output_path)
    if consolidated:
        logger.info("Consolidated %d tag files: %s", len(consolidated), list(consolidated.keys()))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Stage 5: Convert masks to Blender input")
    p.add_argument("--mask-dir", required=True, help="Directory with *_masks.json (stage 4 output)")
    p.add_argument("--tiles-dir", required=True, help="Directory with tileset.json")
    p.add_argument("--output-dir", default="output", help="Output base directory")
    args = p.parse_args()
    config = PipelineConfig(tiles_dir=args.tiles_dir, output_dir=args.output_dir).resolve()
    config.mask_on_clips_dir = os.path.abspath(args.mask_dir)
    run(config)
