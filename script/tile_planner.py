"""
Tile planner for Stage 9 parallelization.

Computes tile plans for all tags without starting Blender processes.
Outputs a JSON file with tile specifications for parallel execution.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from typing import Dict, List, TypedDict

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from pipeline_config import PipelineConfig


class TileSpec(TypedDict):
    """Specification for a single tile."""
    tx: int
    tz: int
    min_x: float
    max_x: float
    min_z: float
    max_z: float
    y_value: float


MAX_GRID_VERTS = 1_000_000


def _compute_tile_plan(
    width_x: float, width_z: float, density: float, max_verts: int = MAX_GRID_VERTS
) -> tuple[int, int]:
    """Compute tiling plan based on grid size."""
    subdivs_x = max(1, math.ceil(width_x / density))
    subdivs_z = max(1, math.ceil(width_z / density))
    total_verts = (subdivs_x + 1) * (subdivs_z + 1)

    if total_verts <= max_verts:
        return (1, 1)

    scale = math.sqrt(total_verts / max_verts)
    tiles_x = max(1, int(math.ceil(scale)))
    tiles_z = max(1, int(math.ceil(scale)))

    return (tiles_x, tiles_z)


def compute_tile_plans(
    blend_file: str,
    blender_exe: str,
    tags: List[str],
    densities: Dict[str, float],
    max_verts: int = MAX_GRID_VERTS
) -> Dict[str, List[TileSpec]]:
    """
    Compute tile plans for all tags by reading mask polygons from blend file.

    Uses a lightweight Blender script to extract mask bounds.
    """
    # Create temporary Python script to extract mask bounds
    script = f"""
import bpy
import json
import sys

bounds = {{}}
for tag in {tags}:
    mask_col_name = f"mask_polygon_{{tag}}"
    mask_col = bpy.data.collections.get(mask_col_name)
    if not mask_col or not mask_col.objects:
        continue

    mask_obj = mask_col.objects[0]
    mw = mask_obj.matrix_world

    min_x = min_z = float("inf")
    max_x = max_z = float("-inf")
    y_sum = 0.0
    count = 0

    for v in mask_obj.data.vertices:
        w = mw @ v.co
        min_x = min(min_x, w.x)
        max_x = max(max_x, w.x)
        min_z = min(min_z, w.z)
        max_z = max(max_z, w.z)
        y_sum += w.y
        count += 1

    if count > 0:
        bounds[tag] = {{
            "min_x": min_x,
            "max_x": max_x,
            "min_z": min_z,
            "max_z": max_z,
            "y_value": y_sum / count
        }}

print("@@BOUNDS@@" + json.dumps(bounds))
"""

    # Run Blender to extract bounds
    script_file = blend_file.replace(".blend", "_tile_plan.py")
    with open(script_file, "w") as f:
        f.write(script)

    try:
        result = subprocess.run(
            [blender_exe, "--background", blend_file, "--python", script_file],
            capture_output=True,
            text=True,
            check=True
        )

        # Parse bounds from output
        bounds_data = {}
        for line in result.stdout.split("\n"):
            if line.startswith("@@BOUNDS@@"):
                bounds_data = json.loads(line[10:])
                break

        # Compute tile plans
        tile_plans: Dict[str, List[TileSpec]] = {}

        for tag, bounds in bounds_data.items():
            density = densities.get(tag, 2.0)
            min_x, max_x = bounds["min_x"], bounds["max_x"]
            min_z, max_z = bounds["min_z"], bounds["max_z"]
            y_value = bounds["y_value"]

            # Add margin
            margin = 1.0
            min_x -= margin
            max_x += margin
            min_z -= margin
            max_z += margin

            width_x = max_x - min_x
            width_z = max_z - min_z

            tiles_x, tiles_z = _compute_tile_plan(width_x, width_z, density, max_verts)

            # Compute tile bounds
            cells_x = math.ceil(width_x / density)
            cells_z = math.ceil(width_z / density)
            cells_per_tile_x = math.ceil(cells_x / tiles_x)
            cells_per_tile_z = math.ceil(cells_z / tiles_z)

            tiles = []
            for tz in range(tiles_z):
                for tx in range(tiles_x):
                    start_cell_x = tx * cells_per_tile_x
                    end_cell_x = min((tx + 1) * cells_per_tile_x, cells_x)
                    start_cell_z = tz * cells_per_tile_z
                    end_cell_z = min((tz + 1) * cells_per_tile_z, cells_z)

                    tile_min_x = min_x + start_cell_x * density
                    tile_max_x = min_x + end_cell_x * density
                    tile_min_z = min_z + start_cell_z * density
                    tile_max_z = min_z + end_cell_z * density

                    tiles.append(TileSpec(
                        tx=tx, tz=tz,
                        min_x=tile_min_x, max_x=tile_max_x,
                        min_z=tile_min_z, max_z=tile_max_z,
                        y_value=y_value
                    ))

            tile_plans[tag] = tiles

        return tile_plans

    finally:
        if os.path.exists(script_file):
            os.remove(script_file)


def save_tile_plan(tile_plans: Dict[str, List[TileSpec]], output_path: str) -> None:
    """Save tile plans to JSON file."""
    with open(output_path, "w") as f:
        json.dump(tile_plans, f, indent=2)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Compute tile plans for Stage 9 parallelization")
    p.add_argument("--blend-file", required=True, help="Input blend file")
    p.add_argument("--blender-exe", required=True, help="Blender executable")
    p.add_argument("--output", required=True, help="Output JSON file")
    args = p.parse_args()

    tags = ["road", "kerb", "grass", "sand", "road2"]
    densities = {
        "road": 0.1,
        "kerb": 0.1,
        "grass": 2.0,
        "sand": 2.0,
        "road2": 2.0,
    }

    tile_plans = compute_tile_plans(
        args.blend_file,
        args.blender_exe,
        tags,
        densities
    )

    save_tile_plan(tile_plans, args.output)
    print(f"Tile plan saved to {args.output}")

    # Print summary
    for tag, tiles in tile_plans.items():
        print(f"  {tag}: {len(tiles)} tiles")
