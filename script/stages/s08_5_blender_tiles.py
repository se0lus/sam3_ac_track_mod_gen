"""Stage 8.5: Load and refine terrain tiles in Blender."""
from __future__ import annotations
import logging
import os
import subprocess
import sys

logger = logging.getLogger("sam3_pipeline.s08_5")

_script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from pipeline_config import PipelineConfig


def run(config: PipelineConfig) -> None:
    """Execute Stage 8.5: Load terrain tiles."""
    logger.info("=== Stage 8.5: Blender tile loading ===")

    out_dir = config.stage_dir("blender_tiles")
    os.makedirs(out_dir, exist_ok=True)
    tiles_blend = os.path.join(out_dir, "tiles_loaded.blend")

    blender_script = os.path.join(
        _script_dir, "..", "blender_scripts", "blender_automate.py"
    )
    blender_script = os.path.abspath(blender_script)

    cmd = [config.blender_exe]
    if not config.s9_no_background:
        cmd.append("--background")
    cmd.extend([
        "--python", blender_script,
        "--",
        "--mode", "tiles",
        "--blend-input", config.blend_file,
        "--glb-dir", config.glb_dir,
        "--tiles-dir", config.tiles_dir,
        "--consolidated-clips-dir", config.merge_segments_result,
        "--output", tiles_blend,
        "--base-level", str(config.base_level),
        "--target-level", str(config.target_fine_level),
        "--tile-padding", str(config.s9_tile_padding),
    ])

    # Polygon directory for tile refinement plan
    polygon_dir = os.path.join(config.stage_dir("blender_polygons"), "gap_filled")
    if os.path.isdir(polygon_dir):
        cmd.extend(["--polygon-dir", polygon_dir])

    if config.s9_refine_tags:
        cmd.extend(["--refine-tags", ",".join(config.s9_refine_tags)])

    logger.info("Running Blender tile loading: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    logger.info("Tile loading complete: %s", tiles_blend)
