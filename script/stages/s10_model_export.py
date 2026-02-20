"""Stage 10: Model export — clean, split, rename, batch, and export FBX.

Reads ``09_result/final_track.blend`` and exports split models as FBX files
suitable for Assetto Corsa.  Runs Blender in ``--background`` mode.

Tile levels are auto-detected from ``L{N}`` collections in the .blend file,
so no base_level/target_level parameters are needed.
"""
from __future__ import annotations

import argparse
import glob
import logging
import os
import subprocess
import sys

logger = logging.getLogger("sam3_pipeline.s10")

_script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from pipeline_config import PipelineConfig


def run(config: PipelineConfig) -> None:
    """Execute Stage 10: Model export.

    Reads:
    - ``config.blender_result_dir / final_track.blend`` from stage 9

    Writes to ``config.export_dir`` (``output/10_model_export/``).
    """
    logger.info("=== Stage 10: Model export ===")

    if not config.blender_exe:
        raise ValueError("blender_exe is required for model_export stage")

    # Locate input blend file from 09_result junction
    blend_input = os.path.join(config.blender_result_dir, "final_track.blend")
    if not os.path.isfile(blend_input):
        raise FileNotFoundError(
            f"Input blend file not found: {blend_input}\n"
            "Run Stage 9 (blender_automate) first."
        )

    os.makedirs(config.export_dir, exist_ok=True)

    blender_script = os.path.join(
        _script_dir, "..", "blender_scripts", "blender_export.py",
    )
    blender_script = os.path.abspath(blender_script)

    # Build Blender command (no base-level/target-level — auto-detected)
    cmd = [
        config.blender_exe, "--background",
        "--python", blender_script,
        "--",
        "--blend-input", blend_input,
        "--output-dir", config.export_dir,
        "--tiles-dir", config.tiles_dir,
        "--max-vertices", str(config.s10_max_vertices),
        "--max-batch-mb", str(config.s10_max_batch_mb),
        "--fbx-scale", str(config.s10_fbx_scale),
    ]

    # Find centerline.json from 07_result
    go_result = config.game_objects_result_dir
    if not os.path.isdir(go_result):
        go_result = os.path.dirname(config.game_objects_json)
    centerline_json = ""
    for candidate in glob.glob(os.path.join(go_result, "*", "centerline.json")):
        centerline_json = candidate
        break
    if not centerline_json:
        candidate = os.path.join(go_result, "centerline.json")
        if os.path.isfile(candidate):
            centerline_json = candidate
    if centerline_json:
        cmd.extend(["--centerline-json", centerline_json])
        logger.info("Using centerline: %s", centerline_json)
    else:
        logger.warning("centerline.json not found — road splitting will use XZ bisection")

    # Find geo_metadata.json (same search as s09)
    walls_result = config.walls_result_dir
    if not os.path.isdir(walls_result):
        walls_result = os.path.dirname(config.walls_json)
    geo_metadata = ""
    for candidate_dir in [walls_result, go_result,
                          os.path.dirname(config.walls_json),
                          os.path.dirname(config.game_objects_json)]:
        candidate = os.path.join(candidate_dir, "geo_metadata.json")
        if os.path.isfile(candidate):
            geo_metadata = candidate
            break
    if geo_metadata:
        cmd.extend(["--geo-metadata", geo_metadata])
        logger.info("Using geo metadata: %s", geo_metadata)
    else:
        logger.warning("geo_metadata.json not found — centerline coordinate conversion may fail")

    logger.info("Running Blender export: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    logger.info("=== Stage 10 complete: %s ===", config.export_dir)


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 10: Model export")
    parser.add_argument("--output-dir", default="output", help="Output base directory")
    parser.add_argument("--tiles-dir", default="", help="Directory with tileset.json")
    parser.add_argument("--blender-exe", default="", help="Path to Blender executable")
    parser.add_argument("--max-vertices", type=int, default=0, help="Max vertices per mesh")
    parser.add_argument("--max-batch-mb", type=int, default=0, help="Max FBX batch size MB")
    parser.add_argument("--fbx-scale", type=float, default=0.0, help="FBX export scale")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = PipelineConfig(
        tiles_dir=args.tiles_dir,
        output_dir=args.output_dir,
    ).resolve()
    if args.blender_exe:
        config.blender_exe = args.blender_exe
    if args.max_vertices > 0:
        config.s10_max_vertices = args.max_vertices
    if args.max_batch_mb > 0:
        config.s10_max_batch_mb = args.max_batch_mb
    if args.fbx_scale > 0:
        config.s10_fbx_scale = args.fbx_scale
    run(config)


if __name__ == "__main__":
    main()
