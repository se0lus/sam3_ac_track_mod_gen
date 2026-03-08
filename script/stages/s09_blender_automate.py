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


def _run_parallel(config: PipelineConfig) -> None:
    """Execute Stage 9 with parallel surface extraction."""
    logger.info("Using parallel surface extraction (max_workers=%d)", config.max_workers)

    # Step 1: Compute tile plans
    from tile_planner import compute_tile_plans, save_tile_plan

    tiles_blend = os.path.join(config.stage_dir("blender_tiles"), "tiles_loaded.blend")
    if not os.path.isfile(tiles_blend):
        logger.error("tiles_loaded.blend not found, parallel mode requires Stage 8.5")
        raise FileNotFoundError(f"tiles_loaded.blend not found: {tiles_blend}")

    tags = ["road", "kerb", "grass", "sand", "road2"]
    densities = {
        "road": config.surface_density_road,
        "kerb": config.surface_density_kerb,
        "grass": config.surface_density_grass,
        "sand": config.surface_density_sand,
        "road2": config.surface_density_road2,
    }

    logger.info("Computing tile plans...")
    tile_plan = compute_tile_plans(
        tiles_blend,
        config.blender_exe,
        tags,
        densities
    )

    total_tiles = sum(len(tiles) for tiles in tile_plan.values())
    logger.info("Tile plan: %d total tiles", total_tiles)
    for tag, tiles in tile_plan.items():
        logger.info("  %s: %d tiles", tag, len(tiles))

    # Step 2: Parallel processing
    from parallel_surface_extractor import extract_surfaces_parallel

    logger.info("Extracting surfaces in parallel...")
    tile_outputs = extract_surfaces_parallel(config, tile_plan, config.max_workers)
    logger.info("Parallel extraction complete: %d tiles", len(tile_outputs))

    # Step 3: Merge results
    merge_script = os.path.join(
        _script_dir, "..", "blender_scripts", "merge_tiles.py"
    )
    merge_script = os.path.abspath(merge_script)

    logger.info("Merging tile results...")
    merged_blend = os.path.join(config.stage_dir("blender_automate"), "surfaces_merged.blend")
    cmd = [
        config.blender_exe,
        "--background",
        "--python", merge_script,
        "--",
        "--base", tiles_blend,
        "--tiles", ",".join(tile_outputs),
        "--output", merged_blend,
    ]
    subprocess.run(cmd, check=True)
    logger.info("Merge complete: %s", merged_blend)

    # Step 4: Continue with walls/objects/textures
    logger.info("Importing walls, objects, and textures...")
    blender_script = os.path.join(_script_dir, "..", "blender_scripts", "blender_automate.py")
    blender_script = os.path.abspath(blender_script)

    blender_clips = config.merge_segments_result
    if not os.path.isdir(blender_clips):
        blender_clips = config.merge_segments_dir

    cmd = [config.blender_exe]
    if config.s9_background:
        cmd.append("--background")
    cmd.extend([
        "--python", blender_script,
        "--",
        "--mode", "surfaces",
        "--blend-input", merged_blend,  # Use merged blend as input
        "--tiles-blend-input", merged_blend,  # Same file for tiles
        "--glb-dir", config.glb_dir,
        "--tiles-dir", config.tiles_dir,
        "--consolidated-clips-dir", blender_clips,
        "--output", config.final_blend_file,
        "--base-level", str(config.base_level),
        "--target-level", str(config.target_fine_level),
        "--skip-surfaces",  # Skip surface extraction (already done)
    ])

    # Add road-kerb-bool if configured
    if config.s9_road_kerb_method == "bool":
        cmd.append("--road-kerb-bool")

    if not config.s9_convert_textures:
        cmd.append("--skip-textures")

    # Walls
    walls_result = config.walls_result_dir
    if not os.path.isdir(walls_result):
        walls_result = os.path.dirname(config.walls_json)
    walls_json = os.path.join(walls_result, "walls.json")
    if not config.s9_import_walls:
        cmd.append("--skip-walls")
    elif os.path.isfile(walls_json):
        cmd.extend(["--walls-json", walls_json])

    # Game objects
    go_result = config.game_objects_result_dir
    if not os.path.isdir(go_result):
        go_result = os.path.dirname(config.game_objects_json)
    go_json = os.path.join(go_result, "game_objects.json")
    if not config.s9_import_game_objects:
        cmd.append("--skip-game-objects")
    elif os.path.isfile(go_json):
        cmd.extend(["--game-objects-json", go_json])

    # Geo metadata
    for candidate_dir in [walls_result, go_result]:
        candidate = os.path.join(candidate_dir, "geo_metadata.json")
        if os.path.isfile(candidate):
            cmd.extend(["--geo-metadata", candidate])
            break

    logger.info("Running Blender automation: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    logger.info("Stage 9 parallel complete: %s", config.final_blend_file)


def run(config: PipelineConfig) -> None:
    """Execute Stage 9: Blender surfaces & game objects.

    Reads:
    - ``tiles_loaded.blend`` from stage 8.5 (or ``config.blend_file`` fallback)
    - ``config.walls_json`` from stage 6 (optional)
    - ``config.game_objects_json`` from stage 7 (optional)

    Writes ``final_track.blend`` to ``config.final_blend_file``.
    """
    logger.info("=== Stage 9: Blender surfaces & game objects ===")

    # Check for parallel mode
    if (config.s9_parallel_surfaces
        and config.max_workers > 1
        and config.s9_extract_surfaces):
        _run_parallel(config)
        return

    if not config.blender_exe:
        raise ValueError("blender_exe is required for blender_automate stage")

    # Input: tiles_loaded.blend from Stage 8.5
    tiles_blend_dir = config.stage_dir("blender_tiles")
    tiles_blend = os.path.join(tiles_blend_dir, "tiles_loaded.blend")

    # Backward compatibility: fallback to full mode if Stage 8.5 not run
    if not os.path.isfile(tiles_blend):
        logger.warning("tiles_loaded.blend not found, using full mode")
        tiles_blend = config.blend_file
        mode = "full"
    else:
        mode = "surfaces"

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
    if config.s9_background:
        cmd.append("--background")
    cmd.extend([
        "--python", blender_script,
        "--",
        "--mode", mode,
        "--blend-input", config.blend_file,
        "--tiles-blend-input", tiles_blend,
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
    if not config.s9_extract_surfaces:
        cmd.append("--skip-surfaces")
    if not config.s9_convert_textures:
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

    # Road/Kerb extraction method
    if config.s9_road_kerb_method == "bool":
        cmd.append("--road-kerb-bool")

    # Boolean mesh debug saves
    if config.s9_debug_boolean:
        cmd.append("--debug-boolean")

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
    if not config.s9_import_walls:
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
    if not config.s9_import_game_objects:
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
