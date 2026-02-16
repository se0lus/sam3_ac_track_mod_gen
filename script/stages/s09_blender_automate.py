"""Stage 9: Run all Blender-side operations (headless automation)."""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys

logger = logging.getLogger("sam3_pipeline.s09")

_script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from pipeline_config import PipelineConfig


def run(config: PipelineConfig) -> None:
    """Execute Stage 9: Blender headless automation.

    Reads:
    - ``config.blend_file`` from stage 6
    - ``config.glb_dir`` from stage 1
    - ``config.blender_clips_dir`` from stage 5
    - ``config.walls_json`` from stage 7 (optional)
    - ``config.game_objects_json`` from stage 8 (optional)

    Writes ``final_track.blend`` to ``config.final_blend_file``.
    """
    logger.info("=== Stage 9: Blender headless automation ===")

    if not config.blender_exe:
        raise ValueError("blender_exe is required for blender_automate stage")
    if not os.path.isfile(config.blend_file):
        raise ValueError(f"blend_file not found: {config.blend_file}")

    out_dir = os.path.dirname(config.final_blend_file)
    os.makedirs(out_dir, exist_ok=True)

    blender_script = os.path.join(
        _script_dir, "..", "blender_scripts", "blender_automate.py",
    )
    blender_script = os.path.abspath(blender_script)

    cmd = [
        config.blender_exe,
        "--background",
        "--python", blender_script,
        "--",
        "--blend-input", config.blend_file,
        "--glb-dir", config.glb_dir,
        "--tiles-dir", config.tiles_dir,
        "--consolidated-clips-dir", config.blender_clips_dir,
        "--output", config.final_blend_file,
        "--base-level", str(config.base_level),
        "--target-level", str(config.target_fine_level),
    ]

    # Optional stages: walls and game objects
    if os.path.isfile(config.walls_json):
        cmd.extend(["--walls-json", config.walls_json])
    if os.path.isfile(config.game_objects_json):
        cmd.extend(["--game-objects-json", config.game_objects_json])

    logger.info("Running Blender automation: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    logger.info("Blender automation complete: %s", config.final_blend_file)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Stage 9: Blender headless automation")
    p.add_argument("--blender-exe", default="", help="Path to Blender executable")
    p.add_argument("--tiles-dir", required=True, help="Directory with tileset.json")
    p.add_argument("--output-dir", default="output", help="Output base directory")
    args = p.parse_args()
    config = PipelineConfig(tiles_dir=args.tiles_dir, output_dir=args.output_dir).resolve()
    if args.blender_exe:
        config.blender_exe = args.blender_exe
    run(config)
