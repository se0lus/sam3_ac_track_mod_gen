"""
Blender headless automation script.

Chains ALL Blender-side operations in sequence, producing a final .blend file
with no human intervention.  Designed to run in ``--background`` mode.

Usage::

    blender --background --python blender_automate.py -- \\
        --blend-input output/polygons.blend \\
        --glb-dir output/glb \\
        --tiles-dir test_images_shajing/b3dm \\
        --consolidated-clips-dir output/blender_clips \\
        --walls-json output/walls.json \\
        --game-objects-json output/game_objects.json \\
        --output output/final_track.blend

Steps executed:
  1. Open the input .blend (polygons from stage 6)
  2. Load base GLB tiles  (reuses import_fullscene_with_ctile)
  3. Refine tiles by mask  (reuses refine_by_mask_sync)
  4. Extract collision surfaces  (calls bpy.ops.sam3.extract_surfaces)
  5. Import virtual walls  (calls bpy.ops.sam3.import_walls)
  6. Import game objects  (calls bpy.ops.sam3.import_game_objects)
  7. Texture processing  (calls bpy.ops.sam3.unpack/convert/bsdf operators)
  8. Save final .blend
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# ---------------------------------------------------------------------------
# sys.path setup -- must happen BEFORE any project imports
# ---------------------------------------------------------------------------
_this_dir = os.path.dirname(os.path.realpath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

_script_dir = os.path.join(os.path.dirname(_this_dir), "script")
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

import bpy  # type: ignore[import-not-found]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("blender_automate")


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def _get_script_argv() -> list[str]:
    argv = sys.argv
    if "--" in argv:
        return argv[argv.index("--") + 1:]
    return []


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Blender headless automation -- full track pipeline",
    )
    p.add_argument("--blend-input", required=True,
                    help="Input .blend from polygon generation stage")
    p.add_argument("--glb-dir", required=True,
                    help="Directory containing converted GLB tiles")
    p.add_argument("--tiles-dir", required=True,
                    help="Directory containing tileset.json (b3dm source)")
    p.add_argument("--consolidated-clips-dir", required=True,
                    help="Directory with {tag}_clip.json files")
    p.add_argument("--walls-json", default="",
                    help="Wall JSON file (optional, skip if not provided)")
    p.add_argument("--game-objects-json", default="",
                    help="Game objects JSON file (optional, skip if not provided)")
    p.add_argument("--output", required=True,
                    help="Output .blend file path")
    p.add_argument("--base-level", type=int, default=17,
                    help="Base tile level to load (default: 17)")
    p.add_argument("--target-level", type=int, default=22,
                    help="Target refinement level (default: 22)")
    return p.parse_args(_get_script_argv())


# ---------------------------------------------------------------------------
# Main automation pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # Resolve all paths to absolute
    blend_input = os.path.abspath(args.blend_input)
    glb_dir = os.path.abspath(args.glb_dir)
    tiles_dir = os.path.abspath(args.tiles_dir)
    clips_dir = os.path.abspath(args.consolidated_clips_dir)
    walls_json = os.path.abspath(args.walls_json) if args.walls_json else ""
    go_json = os.path.abspath(args.game_objects_json) if args.game_objects_json else ""
    output = os.path.abspath(args.output)

    # ------------------------------------------------------------------
    # Override config values BEFORE any action modules are imported.
    # When blender_helpers.register() later imports the action modules,
    # they do ``from config import X`` and pick up these updated values.
    # ------------------------------------------------------------------
    import config
    config.BASE_TILES_DIR = tiles_dir
    config.GLB_DIR = glb_dir
    config.BASE_LEVEL = args.base_level
    config.TARGET_FINE_LEVEL = args.target_level
    config.CONSOLIDATED_CLIPS_DIR = clips_dir

    # ------------------------------------------------------------------
    # Step 1: Open the input .blend file
    # ------------------------------------------------------------------
    log.info("Step 1/8: Opening blend file: %s", blend_input)
    bpy.ops.wm.open_mainfile(filepath=blend_input)

    # ------------------------------------------------------------------
    # Step 2: Register all SAM3 operators
    # ------------------------------------------------------------------
    log.info("Step 2/8: Registering SAM3 operators...")
    import blender_helpers
    blender_helpers.register()

    # ------------------------------------------------------------------
    # Step 3: Load base tiles (synchronous -- modal operator unusable in bg)
    # ------------------------------------------------------------------
    log.info("Step 3/8: Loading base tiles (level=%d)...", args.base_level)
    from sam3_actions.c_tiles import CTile
    from sam3_actions.load_base_tiles import import_fullscene_with_ctile

    tileset_path = os.path.join(tiles_dir, "tileset.json")
    root_tile = CTile()
    root_tile.loadFromRootJson(tileset_path)
    import_fullscene_with_ctile(root_tile, glb_dir, min_level=args.base_level)
    log.info("Base tiles loaded.")

    # ------------------------------------------------------------------
    # Step 4: Refine tiles by mask to target level (synchronous)
    # ------------------------------------------------------------------
    log.info("Step 4/8: Refining tiles by mask to level %d...", args.target_level)
    from sam3_actions.load_base_tiles import refine_by_mask_sync

    # Collect mask objects from mask_polygon_collection
    mask_col = bpy.data.collections.get(config.ROOT_POLYGON_COLLECTION_NAME)
    if mask_col is not None:
        masks = list(mask_col.all_objects)
    else:
        masks = []

    if masks:
        # Reload tileset (may have been modified during load)
        root_tile2 = CTile()
        root_tile2.loadFromRootJson(tileset_path)
        refine_by_mask_sync(
            context=bpy.context,
            masks=masks,
            root_tile=root_tile2,
            glb_dir=glb_dir,
            target_level=args.target_level,
        )
        log.info("Tile refinement complete.")
    else:
        log.warning("No mask objects found in '%s', skipping refinement.",
                     config.ROOT_POLYGON_COLLECTION_NAME)

    # ------------------------------------------------------------------
    # Step 5: Extract collision surfaces
    # ------------------------------------------------------------------
    log.info("Step 5/8: Extracting collision surfaces...")
    result = bpy.ops.sam3.extract_surfaces()
    log.info("Extract surfaces result: %s", result)

    # ------------------------------------------------------------------
    # Step 6: Import virtual walls (optional)
    # ------------------------------------------------------------------
    if walls_json and os.path.isfile(walls_json):
        log.info("Step 6/8: Importing walls from %s...", walls_json)
        result = bpy.ops.sam3.import_walls(
            'EXEC_DEFAULT',
            filepath=walls_json,
        )
        log.info("Import walls result: %s", result)
    else:
        log.info("Step 6/8: Skipped (no walls JSON provided).")

    # ------------------------------------------------------------------
    # Step 7: Import game objects (optional)
    # ------------------------------------------------------------------
    if go_json and os.path.isfile(go_json):
        log.info("Step 7/8: Importing game objects from %s...", go_json)
        result = bpy.ops.sam3.import_game_objects(
            'EXEC_DEFAULT',
            filepath=go_json,
        )
        log.info("Import game objects result: %s", result)
    else:
        log.info("Step 7/8: Skipped (no game objects JSON provided).")

    # ------------------------------------------------------------------
    # Step 8: Texture processing + final save
    # ------------------------------------------------------------------
    log.info("Step 8/8: Processing textures and saving...")

    # Save first to establish the blend directory (needed by unpack_textures)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=output)

    bpy.ops.sam3.unpack_textures()
    bpy.ops.sam3.convert_textures_png()
    bpy.ops.sam3.convert_materials_bsdf()

    # Final save
    bpy.ops.wm.save_as_mainfile(filepath=output)
    log.info("Done! Output saved to: %s", output)


if __name__ == "__main__":
    main()
