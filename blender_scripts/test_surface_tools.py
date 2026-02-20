"""
Blender headless test script for the two collision surface tools.

Runs both Tool A (terrain_mesh_extractor — road/kerb) and Tool B
(boolean_mesh_generator — grass/sand/road2) with low-density test
parameters, then exports every collision mesh as OBJ + a summary
``report.json`` for offline validation.

Usage::

    "C:\\Program Files\\Blender Foundation\\Blender 5.0\\blender.exe" --background \
        --python blender_scripts/test_surface_tools.py -- \
        --blend-input output/08_blender_polygons/polygons.blend \
        --tiles-dir test_images_shajing/b3dm \
        --glb-dir output/01_b3dm_convert \
        --output-dir output/test_surface_tools

Steps:
  1. Open polygons.blend
  2. Override config with low densities for speed
  3. Register SAM3 operators
  4. Load L17 base tiles only (no refinement)
  5. Run sam3.extract_terrain_surfaces  (Tool A)
  6. Run sam3.generate_boolean_surfaces (Tool B)
  7. Export collision_* objects as OBJ + report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

# ---------------------------------------------------------------------------
# sys.path setup — BEFORE any project imports
# ---------------------------------------------------------------------------
_this_dir = os.path.dirname(os.path.realpath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

_script_dir = os.path.join(os.path.dirname(_this_dir), "script")
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

import bpy  # type: ignore[import-not-found]
from mathutils import Vector  # type: ignore[import-not-found]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("test_surface_tools")


# ---------------------------------------------------------------------------
# Argument parsing (after Blender's '--')
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Test surface tools in Blender headless mode")
    p.add_argument("--blend-input", required=True,
                    help="Path to polygons.blend (from stage 8)")
    p.add_argument("--glb-dir", required=True,
                    help="Directory with converted GLB tiles")
    p.add_argument("--tiles-dir", required=True,
                    help="Directory with tileset.json + b3dm")
    p.add_argument("--output-dir", required=True,
                    help="Directory for OBJ exports and report.json")
    p.add_argument("--base-level", type=int, default=17,
                    help="Tile level to load (default: 17)")

    # Consume everything after '--'
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# OBJ export helper
# ---------------------------------------------------------------------------

def _export_obj(obj: bpy.types.Object, filepath: str) -> None:
    """Export a single mesh object to Wavefront OBJ."""
    me = obj.data
    mw = obj.matrix_world

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# Exported from test_surface_tools: {obj.name}\n")
        for v in me.vertices:
            co = mw @ v.co
            f.write(f"v {co.x:.6f} {co.y:.6f} {co.z:.6f}\n")
        for poly in me.polygons:
            indices = " ".join(str(vi + 1) for vi in poly.vertices)
            f.write(f"f {indices}\n")


# ---------------------------------------------------------------------------
# Collision mesh statistics
# ---------------------------------------------------------------------------

def _mesh_stats(obj: bpy.types.Object) -> dict:
    """Gather statistics for a collision mesh object."""
    me = obj.data
    mw = obj.matrix_world

    n_verts = len(me.vertices)
    n_faces = len(me.polygons)

    # Compute world-space bounding box
    min_xyz = [float("inf")] * 3
    max_xyz = [float("-inf")] * 3
    for v in me.vertices:
        co = mw @ v.co
        for i in range(3):
            if co[i] < min_xyz[i]:
                min_xyz[i] = co[i]
            if co[i] > max_xyz[i]:
                max_xyz[i] = co[i]

    if n_verts == 0:
        min_xyz = [0, 0, 0]
        max_xyz = [0, 0, 0]

    y_range = max_xyz[1] - min_xyz[1]

    return {
        "verts": n_verts,
        "faces": n_faces,
        "bbox": {
            "min": [round(v, 4) for v in min_xyz],
            "max": [round(v, 4) for v in max_xyz],
        },
        "y_range": round(y_range, 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # Force line-buffered output
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(line_buffering=True)
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

    # Resolve paths
    blend_input = os.path.abspath(args.blend_input)
    glb_dir = os.path.abspath(args.glb_dir)
    tiles_dir = os.path.abspath(args.tiles_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    t_total = time.monotonic()

    # ------------------------------------------------------------------
    # Step 1: Open the input .blend
    # ------------------------------------------------------------------
    log.info("Step 1/7: Opening %s", blend_input)
    bpy.ops.wm.open_mainfile(filepath=blend_input)

    # ------------------------------------------------------------------
    # Step 2: Override config with low test densities
    # ------------------------------------------------------------------
    log.info("Step 2/7: Overriding config (low density for test speed)")
    import config
    config.BASE_TILES_DIR = tiles_dir
    config.GLB_DIR = glb_dir
    config.BASE_LEVEL = args.base_level
    config.TARGET_FINE_LEVEL = args.base_level  # no refinement

    # Low densities to keep test fast
    config.SURFACE_SAMPLING_DENSITY_ROAD = 1.0
    config.SURFACE_SAMPLING_DENSITY_KERB = 1.0
    config.SURFACE_SAMPLING_DENSITY_GRASS = 5.0
    config.SURFACE_SAMPLING_DENSITY_SAND = 5.0
    config.SURFACE_SAMPLING_DENSITY_ROAD2 = 5.0
    config.SURFACE_EDGE_SIMPLIFY = 0.0

    log.info("  densities: road/kerb=1.0m, grass/sand/road2=5.0m")

    # ------------------------------------------------------------------
    # Step 3: Register SAM3 operators
    # ------------------------------------------------------------------
    log.info("Step 3/7: Registering SAM3 operators")
    import blender_helpers
    blender_helpers.register()

    # ------------------------------------------------------------------
    # Step 4: Load base tiles (L17 only, no refinement)
    # ------------------------------------------------------------------
    log.info("Step 4/7: Loading base tiles (level=%d, no refinement)",
             args.base_level)
    from sam3_actions.c_tiles import CTile
    from sam3_actions.load_base_tiles import import_fullscene_with_ctile

    tileset_path = os.path.join(tiles_dir, "tileset.json")
    root_tile = CTile()
    root_tile.loadFromRootJson(tileset_path)
    import_fullscene_with_ctile(root_tile, glb_dir, min_level=args.base_level)
    log.info("  Base tiles loaded.")

    # ------------------------------------------------------------------
    # Step 5: Run Tool A — terrain mesh extraction (road + kerb)
    # ------------------------------------------------------------------
    log.info("Step 5/7: Running Tool A — sam3.extract_terrain_surfaces")
    t_a = time.monotonic()
    result_a = bpy.ops.sam3.extract_terrain_surfaces()
    elapsed_a = time.monotonic() - t_a
    log.info("  Tool A result: %s (%.1fs)", result_a, elapsed_a)

    # ------------------------------------------------------------------
    # Step 6: Run Tool B — boolean surfaces (grass/sand/road2)
    # ------------------------------------------------------------------
    log.info("Step 6/7: Running Tool B — sam3.generate_boolean_surfaces")
    t_b = time.monotonic()
    result_b = bpy.ops.sam3.generate_boolean_surfaces()
    elapsed_b = time.monotonic() - t_b
    log.info("  Tool B result: %s (%.1fs)", result_b, elapsed_b)

    # ------------------------------------------------------------------
    # Step 7: Export collision meshes + report
    # ------------------------------------------------------------------
    log.info("Step 7/7: Exporting collision meshes to %s", output_dir)

    results = []
    collision_collections = []
    for col in bpy.data.collections:
        if col.name == "collision" or col.name.startswith("collision_"):
            collision_collections.append(col)

    for col in sorted(collision_collections, key=lambda c: c.name):
        # Infer tag from collection name
        if col.name == "collision":
            tag = "wall"
        elif col.name.startswith("collision_"):
            tag = col.name[len("collision_"):]
        else:
            tag = col.name

        for obj in col.all_objects:
            if obj.type != "MESH":
                continue

            stats = _mesh_stats(obj)

            # Export OBJ
            col_subdir = os.path.join(output_dir, col.name)
            obj_path = os.path.join(col_subdir, f"{obj.name}.obj")
            _export_obj(obj, obj_path)

            entry = {
                "tag": tag,
                "collection": col.name,
                "object": obj.name,
                **stats,
            }
            results.append(entry)
            log.info("  %s/%s: %d verts, %d faces, y_range=%.2f",
                     col.name, obj.name, stats["verts"], stats["faces"],
                     stats["y_range"])

    # Write report
    report = {
        "tool_a_result": str(result_a),
        "tool_b_result": str(result_b),
        "tool_a_elapsed_s": round(elapsed_a, 2),
        "tool_b_elapsed_s": round(elapsed_b, 2),
        "total_objects": len(results),
        "results": results,
    }
    report_path = os.path.join(output_dir, "report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    total_elapsed = time.monotonic() - t_total
    log.info("=== Test complete: %d collision objects exported, %.1fs total ===",
             len(results), total_elapsed)
    log.info("Report: %s", report_path)


if __name__ == "__main__":
    main()
