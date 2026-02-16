"""Stage 6: Run Blender to create polygon meshes from blender clips."""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys

logger = logging.getLogger("sam3_pipeline.s06")

_script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from pipeline_config import PipelineConfig


def run(config: PipelineConfig) -> None:
    """Execute Stage 6: Blender polygon generation.

    Reads blender clips from ``config.blender_clips_dir`` (stage 5 output),
    writes ``polygons.blend`` to ``config.blend_file``.
    """
    logger.info("=== Stage 6: Blender polygon generation ===")

    if not config.blender_exe:
        raise ValueError("blender_exe is required for blender_polygons stage")
    if not os.path.isdir(config.blender_clips_dir):
        raise ValueError(f"blender_clips_dir not found: {config.blender_clips_dir}")

    out_dir = os.path.dirname(config.blend_file)
    os.makedirs(out_dir, exist_ok=True)

    blender_script = os.path.join(
        _script_dir, "..", "blender_scripts", "blender_create_polygons.py"
    )
    blender_script = os.path.abspath(blender_script)

    cmd = [
        config.blender_exe,
        "--background",
        "--python", blender_script,
        "--",
        "--input", config.blender_clips_dir,
        "--output", config.blend_file,
    ]
    logger.info("Running Blender: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    logger.info("Blender polygon generation complete: %s", config.blend_file)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Stage 6: Blender polygon generation")
    p.add_argument("--blender-exe", default="", help="Path to Blender executable")
    p.add_argument("--output-dir", default="output", help="Output base directory")
    args = p.parse_args()
    config = PipelineConfig(output_dir=args.output_dir).resolve()
    if args.blender_exe:
        config.blender_exe = args.blender_exe
    run(config)
