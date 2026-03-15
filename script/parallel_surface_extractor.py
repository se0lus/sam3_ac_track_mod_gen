"""
Parallel surface extractor for Stage 9.

Manages process pool to execute tile processing in parallel.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from pipeline_config import PipelineConfig

logger = logging.getLogger("sam3_pipeline.parallel_surface_extractor")


def _process_tile_worker(
    blender_exe: str,
    blender_script: str,
    tiles_blend: str,
    output_dir: str,
    tag: str,
    tile: Dict,
    args_dict: Dict
) -> str:
    """Process a single tile in a separate Blender process."""
    tx, tz = tile["tx"], tile["tz"]
    tile_spec = f"{tag}:{tx}:{tz}:{tile['min_x']}:{tile['max_x']}:{tile['min_z']}:{tile['max_z']}:{tile['y_value']}"

    tile_output = os.path.join(output_dir, f"tile_{tag}_{tx}_{tz}.blend")

    cmd = [
        blender_exe,
        "--background",
        "--python", blender_script,
        "--",
        "--mode", "surfaces",
        "--tile-mode",
        "--tile-spec", tile_spec,
        "--blend-input", args_dict["blend_input"],
        "--tiles-blend-input", tiles_blend,
        "--glb-dir", args_dict["glb_dir"],
        "--tiles-dir", args_dict["tiles_dir"],
        "--consolidated-clips-dir", args_dict["consolidated_clips_dir"],
        "--output", tile_output,
        "--base-level", str(args_dict["base_level"]),
        "--target-level", str(args_dict["target_level"]),
    ]

    # Add density arguments
    for key in ["density_road", "density_kerb", "density_grass", "density_sand", "density_road2"]:
        if key in args_dict:
            cmd.extend([f"--{key.replace('_', '-')}", str(args_dict[key])])

    # Add road-kerb-bool flag
    if args_dict.get("road_kerb_bool"):
        cmd.append("--road-kerb-bool")

    # Add debug-boolean flag
    if args_dict.get("debug_boolean"):
        cmd.append("--debug-boolean")

    logger.info(f"Processing tile {tag}_{tx}_{tz}...")
    logger.debug(f"Blender command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, capture_output=True)

    return tile_output


def extract_surfaces_parallel(
    config: PipelineConfig,
    tile_plan: Dict[str, List[Dict]],
    max_workers: int
) -> List[str]:
    """
    Execute tile processing in parallel.

    Returns:
        List of output .blend file paths
    """
    # Build task list
    tasks = []
    for tag, tiles in tile_plan.items():
        for tile in tiles:
            tasks.append((tag, tile))

    if not tasks:
        logger.warning("No tiles to process")
        return []

    logger.info(f"Processing {len(tasks)} tiles with {max_workers} workers...")

    # Prepare arguments
    blender_script = os.path.join(
        os.path.dirname(_script_dir), "blender_scripts", "blender_automate.py"
    )
    tiles_blend = os.path.join(config.stage_dir("blender_tiles"), "tiles_loaded.blend")
    output_dir = os.path.join(config.stage_dir("blender_automate"), "tiles")
    os.makedirs(output_dir, exist_ok=True)

    # Read from result junctions
    blender_clips = config.merge_segments_result
    if not os.path.isdir(blender_clips):
        blender_clips = config.merge_segments_dir

    args_dict = {
        "blend_input": config.blend_file,
        "glb_dir": config.glb_dir,
        "tiles_dir": config.tiles_dir,
        "consolidated_clips_dir": blender_clips,
        "base_level": config.base_level,
        "target_level": config.target_fine_level,
        "density_road": config.surface_density_road,
        "density_kerb": config.surface_density_kerb,
        "density_grass": config.surface_density_grass,
        "density_sand": config.surface_density_sand,
        "density_road2": config.surface_density_road2,
        "road_kerb_bool": config.s9_road_kerb_method == "bool",
        "debug_boolean": config.s9_debug_boolean,
    }

    # Execute in parallel with progress tracking
    import time
    results = []
    start_time = time.monotonic()

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for tag, tile in tasks:
            future = pool.submit(
                _process_tile_worker,
                config.blender_exe,
                blender_script,
                tiles_blend,
                output_dir,
                tag,
                tile,
                args_dict
            )
            futures[future] = (tag, tile["tx"], tile["tz"])

        # Collect results with progress and ETA
        for i, future in enumerate(as_completed(futures)):
            tag, tx, tz = futures[future]
            try:
                result = future.result()
                results.append(result)

                # Calculate progress and ETA
                completed = i + 1
                progress_pct = int(100 * completed / len(tasks))
                elapsed = time.monotonic() - start_time

                if completed > 0:
                    avg_time = elapsed / completed
                    remaining = (len(tasks) - completed) * avg_time
                    eta_str = _format_time(remaining)
                    msg = f"Tile {completed}/{len(tasks)}: {tag}_{tx}_{tz} (ETA: {eta_str})"
                    print(f"@@PROGRESS@@ {progress_pct} {msg}", flush=True)
                    logger.info(f"[{progress_pct}%] {msg}")
                else:
                    msg = f"Tile {completed}/{len(tasks)}: {tag}_{tx}_{tz}"
                    print(f"@@PROGRESS@@ {progress_pct} {msg}", flush=True)
                    logger.info(f"[{progress_pct}%] {msg}")
            except Exception as e:
                logger.error(f"Failed {tag}_{tx}_{tz}: {e}")

    elapsed_total = time.monotonic() - start_time
    logger.info(f"Parallel processing complete: {len(results)}/{len(tasks)} tiles in {_format_time(elapsed_total)}")
    return results


def _format_time(seconds: float) -> str:
    """Format seconds as human-readable time string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"
