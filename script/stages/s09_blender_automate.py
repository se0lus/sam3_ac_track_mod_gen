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
    - ``config.blend_file`` from stage 8
    - ``config.glb_dir`` from stage 1
    - ``config.merge_segments_dir`` from stage 5
    - ``config.walls_json`` from stage 6 (optional)
    - ``config.game_objects_json`` from stage 7 (optional)

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

    # Read from result junctions (05_result, 06_result, 07_result)
    blender_clips = config.merge_segments_result
    if not os.path.isdir(blender_clips):
        blender_clips = config.merge_segments_dir  # fallback

    cmd = [config.blender_exe]
    if not config.s9_no_background:
        cmd.append("--background")
    cmd.extend([
        "--python", blender_script,
        "--",
        "--blend-input", config.blend_file,
        "--glb-dir", config.glb_dir,
        "--tiles-dir", config.tiles_dir,
        "--consolidated-clips-dir", blender_clips,
        "--output", config.final_blend_file,
        "--base-level", str(config.base_level),
        "--target-level", str(config.target_fine_level),
    ])

    # Polygon directory for tile refinement plan (Stage 8 gap_filled)
    polygon_dir = os.path.join(config.stage_dir("blender_polygons"), "gap_filled")
    if os.path.isdir(polygon_dir):
        cmd.extend(["--polygon-dir", polygon_dir])
        logger.info("Using polygon dir for tile plan: %s", polygon_dir)
    else:
        logger.warning("Polygon dir not found: %s (falling back to iterative refinement)",
                       polygon_dir)

    # Stage 9 skip flags
    if config.s9_no_surfaces:
        cmd.append("--skip-surfaces")
    if config.s9_no_textures:
        cmd.append("--skip-textures")

    # Refine tags
    if config.s9_refine_tags:
        cmd.extend(["--refine-tags", ",".join(config.s9_refine_tags)])

    # Tile plan padding
    cmd.extend(["--tile-padding", str(config.s9_tile_padding)])

    # Surface extraction parameters
    if config.surface_edge_simplify > 0:
        cmd.extend(["--edge-simplify", str(config.surface_edge_simplify)])
    cmd.extend(["--density-road", str(config.surface_density_road)])
    cmd.extend(["--density-kerb", str(config.surface_density_kerb)])
    cmd.extend(["--density-grass", str(config.surface_density_grass)])
    cmd.extend(["--density-sand", str(config.surface_density_sand)])
    cmd.extend(["--density-road2", str(config.surface_density_road2)])

    # Mesh simplification
    if config.s9_mesh_simplify:
        cmd.append("--mesh-simplify")
        cmd.extend(["--mesh-weld-distance", str(config.s9_mesh_weld_distance)])
        cmd.extend(["--mesh-decimate-ratio", str(config.s9_mesh_decimate_ratio)])

    # Walls from 06_result junction
    walls_result = config.walls_result_dir
    if not os.path.isdir(walls_result):
        walls_result = os.path.dirname(config.walls_json)  # fallback
    walls_json = os.path.join(walls_result, "walls.json")
    if config.s9_no_walls:
        cmd.append("--skip-walls")
        logger.info("Walls import disabled (s9_no_walls)")
    elif os.path.isfile(walls_json):
        cmd.extend(["--walls-json", walls_json])
        logger.info("Using walls: %s", walls_json)

    # Game objects from 07_result junction
    go_result = config.game_objects_result_dir
    if not os.path.isdir(go_result):
        go_result = os.path.dirname(config.game_objects_json)  # fallback
    go_json = os.path.join(go_result, "game_objects.json")
    if config.s9_no_game_objects:
        cmd.append("--skip-game-objects")
        logger.info("Game objects import disabled (s9_no_game_objects)")
    elif os.path.isfile(go_json):
        cmd.extend(["--game-objects-json", go_json])
        logger.info("Using game objects: %s", go_json)

    # Find geo_metadata.json for coordinate conversion
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
        logger.warning("geo_metadata.json not found, wall/object coordinates may be misaligned")

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
