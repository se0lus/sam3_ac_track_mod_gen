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
        --geo-metadata output/geo_metadata.json \\
        --output output/final_track.blend

Steps executed:
  1. Open the input .blend (polygons from stage 6)
  2. Register all SAM3 operators
  3. Load base GLB tiles  (reuses import_fullscene_with_ctile)
  4. Refine tiles by mask  (reuses refine_by_mask_sync)
  5. Extract collision surfaces  (calls bpy.ops.sam3.extract_surfaces)
  6. Import virtual walls  (geo-converted coordinates + terrain-height walls)
  7. Import game objects  (geo-converted coordinates)
  8. Assign hidden material + texture processing + save
"""

from __future__ import annotations

import argparse
import json
import logging
import math
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
import bmesh  # type: ignore[import-not-found]
from mathutils import Vector, Matrix  # type: ignore[import-not-found]

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
# Geo-conversion helpers
# ---------------------------------------------------------------------------

def _read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _pixel_to_geo(px: float, py: float, geo_meta: dict) -> list:
    """Convert pixel [x, y] in modelscale image to WGS84 [lon, lat]."""
    w = geo_meta["image_width"]
    h = geo_meta["image_height"]
    bounds = geo_meta["bounds"]
    lon = bounds["west"] + float(px) * (bounds["east"] - bounds["west"]) / w
    lat = bounds["north"] - float(py) * (bounds["north"] - bounds["south"]) / h
    return [lon, lat]


def _get_terrain_y_bounds() -> tuple:
    """Get Y (up) bounds of all terrain mesh objects in the scene.

    Excludes mask polygon and collision collections to only measure terrain.
    Returns (min_y, max_y). Falls back to (0.0, 20.0) if no terrain found.
    """
    import config as blender_config

    excluded_objs: set = set()
    for col_name in ("collision", "game_objects"):
        col = bpy.data.collections.get(col_name)
        if col:
            for obj in col.all_objects:
                excluded_objs.add(obj.name)
    mask_col = bpy.data.collections.get(blender_config.ROOT_POLYGON_COLLECTION_NAME)
    if mask_col:
        for obj in mask_col.all_objects:
            excluded_objs.add(obj.name)

    min_y = float("inf")
    max_y = float("-inf")
    for obj in bpy.data.objects:
        if obj.type != "MESH" or obj.name in excluded_objs:
            continue
        bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
        for v in bbox_corners:
            if v.y < min_y:
                min_y = v.y
            if v.y > max_y:
                max_y = v.y

    if min_y == float("inf"):
        log.warning("No terrain mesh objects found, using fallback Y bounds (0, 20)")
        return 0.0, 20.0
    return min_y, max_y


def _ensure_collection(name: str):
    """Get or create a Blender collection linked to the scene."""
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(col)
    return col


def _ensure_hidden_material():
    """Create or get the 'hidden' material with Principled BSDF."""
    mat = bpy.data.materials.get("hidden")
    if mat is None:
        mat = bpy.data.materials.new(name="hidden")
        mat.use_nodes = True
        # Default node tree already contains Principled BSDF
    return mat


def _assign_hidden_material_to_collision() -> int:
    """Assign 'hidden' material to all mesh objects in 'collision' collection."""
    mat = _ensure_hidden_material()
    col = bpy.data.collections.get("collision")
    if col is None:
        return 0
    count = 0
    for obj in col.all_objects:
        if obj.type == "MESH":
            obj.data.materials.clear()
            obj.data.materials.append(mat)
            count += 1
    return count


def _import_walls_geo(
    walls_json_path: str,
    geo_meta: dict,
    tf_info,
    wall_bottom: float,
    wall_top: float,
) -> int:
    """Import walls using geo-converted coordinates.

    Converts pixel coords from the wall JSON to Blender 3D coordinates via
    WGS84 -> ECEF -> tileset_local, then creates vertical wall meshes spanning
    from wall_bottom to wall_top (Blender Y axis).
    """
    from geo_sam3_blender_utils import geo_points_to_blender_xyz

    data = _read_json(walls_json_path)
    walls = data.get("walls", [])
    if not walls:
        log.warning("No walls found in %s", walls_json_path)
        return 0

    col = _ensure_collection("collision")
    created = 0

    for wall in walls:
        pts_pixel = wall.get("points", [])
        if len(pts_pixel) < 2:
            continue
        closed = wall.get("closed", True)

        # Pixel -> WGS84
        geo_xy = [_pixel_to_geo(p[0], p[1], geo_meta) for p in pts_pixel]
        # WGS84 -> Blender 3D (Y=0 on ground plane)
        blender_pts = geo_points_to_blender_xyz(geo_xy, tf_info, z_mode="zero")
        if len(blender_pts) < 2:
            continue

        name = f"1WALL_{created}"
        mesh = bpy.data.meshes.new(name)
        bm = bmesh.new()

        n = len(blender_pts)
        segments = n if closed else n - 1
        for i in range(segments):
            p0 = blender_pts[i]
            p1 = blender_pts[(i + 1) % n]
            v0 = bm.verts.new((p0[0], wall_bottom, p0[2]))
            v1 = bm.verts.new((p1[0], wall_bottom, p1[2]))
            v2 = bm.verts.new((p1[0], wall_top, p1[2]))
            v3 = bm.verts.new((p0[0], wall_top, p0[2]))
            bm.faces.new((v0, v1, v2, v3))

        bm.to_mesh(mesh)
        bm.free()
        mesh.update()

        obj = bpy.data.objects.new(name, mesh)
        col.objects.link(obj)
        created += 1

    return created


def _import_game_objects_geo(
    go_json_path: str,
    geo_meta: dict,
    tf_info,
    height_offset: float = 2.0,
) -> int:
    """Import game objects using geo-converted coordinates.

    Converts pixel positions and orientations from the game objects JSON to
    Blender 3D coordinates.  Objects are created as Empty markers.
    """
    from geo_sam3_blender_utils import geo_points_to_blender_xyz

    data = _read_json(go_json_path)
    objects_list = data.get("objects", [])
    if not objects_list:
        log.warning("No objects found in %s", go_json_path)
        return 0

    col = _ensure_collection("game_objects")
    created = 0

    for obj_data in objects_list:
        name = obj_data.get("name", f"OBJECT_{created}")
        pos = obj_data.get("position")
        if not pos or len(pos) < 2:
            continue

        px, py = float(pos[0]), float(pos[1])

        # Position: pixel -> WGS84 -> Blender
        pos_geo = [_pixel_to_geo(px, py, geo_meta)]
        pos_blender = geo_points_to_blender_xyz(pos_geo, tf_info, z_mode="zero")
        if not pos_blender:
            continue
        bpos = pos_blender[0]

        # Orientation: convert direction vector by transforming a nearby point
        forward_blender = None
        orient = obj_data.get("orientation_z")
        if orient and len(orient) >= 2:
            dx, dy = float(orient[0]), float(orient[1])
            ahead_geo = [_pixel_to_geo(px + dx * 10, py + dy * 10, geo_meta)]
            ahead_blender = geo_points_to_blender_xyz(ahead_geo, tf_info, z_mode="zero")
            if ahead_blender:
                apt = ahead_blender[0]
                fdx = apt[0] - bpos[0]
                fdz = apt[2] - bpos[2]
                length = math.sqrt(fdx * fdx + fdz * fdz)
                if length > 1e-9:
                    forward_blender = (fdx / length, fdz / length)

        empty = bpy.data.objects.new(name, None)
        empty.empty_display_type = "PLAIN_AXES"
        empty.empty_display_size = 1.0
        empty.location = (bpos[0], bpos[1] + height_offset, bpos[2])

        if forward_blender is not None:
            fdx, fdz = forward_blender
            forward = Vector((fdx, 0.0, fdz)).normalized()
            up = Vector((0.0, 1.0, 0.0))
            right = up.cross(forward).normalized()
            up = forward.cross(right).normalized()
            rot = Matrix((
                (right.x, up.x, forward.x),
                (right.y, up.y, forward.y),
                (right.z, up.z, forward.z),
            )).transposed()
            empty.rotation_euler = rot.to_euler()

        col.objects.link(empty)
        created += 1

    return created


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
    p.add_argument("--geo-metadata", default="",
                    help="Path to geo_metadata.json for coordinate conversion")
    p.add_argument("--output", required=True,
                    help="Output .blend file path")
    p.add_argument("--base-level", type=int, default=17,
                    help="Base tile level to load (default: 17)")
    p.add_argument("--target-level", type=int, default=22,
                    help="Target refinement level (default: 22)")
    p.add_argument("--skip-walls", action="store_true",
                    help="Skip importing walls")
    p.add_argument("--skip-game-objects", action="store_true",
                    help="Skip importing game objects")
    p.add_argument("--skip-surfaces", action="store_true",
                    help="Skip extracting collision surfaces")
    p.add_argument("--skip-textures", action="store_true",
                    help="Skip texture processing")
    p.add_argument("--refine-tags", default="road",
                    help="Comma-separated mask tags for refinement (default: road)")
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
    geo_meta_path = os.path.abspath(args.geo_metadata) if args.geo_metadata else ""
    output = os.path.abspath(args.output)

    # Load geo metadata for coordinate conversion (if available)
    geo_meta = None
    tf_info = None
    if geo_meta_path and os.path.isfile(geo_meta_path):
        geo_meta = _read_json(geo_meta_path)
        log.info("Loaded geo metadata from %s", geo_meta_path)

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

    # Collect mask objects filtered by refine-tags
    refine_tags = [t.strip() for t in args.refine_tags.split(",") if t.strip()]
    mask_col = bpy.data.collections.get(config.ROOT_POLYGON_COLLECTION_NAME)
    masks = []
    if mask_col is not None:
        for tag in refine_tags:
            sub_col = mask_col.children.get(f"mask_polygon_{tag}")
            if sub_col is not None:
                masks.extend(list(sub_col.all_objects))
    log.info("Refinement mask tags: %s (%d objects)", refine_tags, len(masks))

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
    # Prepare geo-transform and terrain bounds (for walls and game objects)
    # ------------------------------------------------------------------
    if geo_meta is not None:
        from geo_sam3_blender_utils import get_tileset_transform
        bounds = geo_meta.get("bounds", {})
        center_lon = (bounds.get("west", 0) + bounds.get("east", 0)) / 2
        center_lat = (bounds.get("north", 0) + bounds.get("south", 0)) / 2
        tf_info = get_tileset_transform(tiles_dir, sample_geo_xy=(center_lon, center_lat))
        log.info("Tileset transform: mode=%s, source=%s",
                 tf_info.effective_mode, tf_info.tf_source)

    terrain_min_y, terrain_max_y = _get_terrain_y_bounds()
    log.info("Terrain Y bounds: min=%.2f, max=%.2f", terrain_min_y, terrain_max_y)

    # ------------------------------------------------------------------
    # Step 5: Extract collision surfaces
    # ------------------------------------------------------------------
    if not args.skip_surfaces:
        log.info("Step 5/8: Extracting collision surfaces...")
        result = bpy.ops.sam3.extract_surfaces()
        log.info("Extract surfaces result: %s", result)
    else:
        log.info("Step 5/8: Skipped (--skip-surfaces)")

    # ------------------------------------------------------------------
    # Step 6: Import virtual walls (optional)
    # ------------------------------------------------------------------
    if not args.skip_walls and walls_json and os.path.isfile(walls_json):
        log.info("Step 6/8: Importing walls from %s...", walls_json)
        if geo_meta is not None and tf_info is not None:
            wall_bottom = terrain_min_y
            wall_top = terrain_max_y + 20.0
            created = _import_walls_geo(walls_json, geo_meta, tf_info,
                                        wall_bottom, wall_top)
            log.info("Created %d wall objects (geo-converted, Y: %.1f to %.1f)",
                     created, wall_bottom, wall_top)
        else:
            log.warning("No geo metadata -- falling back to operator-based wall import")
            result = bpy.ops.sam3.import_walls(
                'EXEC_DEFAULT',
                filepath=walls_json,
            )
            log.info("Import walls result: %s", result)
    else:
        log.info("Step 6/8: Skipped%s.", " (--skip-walls)" if args.skip_walls else " (no walls JSON)")

    # ------------------------------------------------------------------
    # Step 7: Import game objects (optional)
    # ------------------------------------------------------------------
    if not args.skip_game_objects and go_json and os.path.isfile(go_json):
        log.info("Step 7/8: Importing game objects from %s...", go_json)
        if geo_meta is not None and tf_info is not None:
            created = _import_game_objects_geo(go_json, geo_meta, tf_info)
            log.info("Created %d game objects (geo-converted)", created)
        else:
            log.warning("No geo metadata -- falling back to operator-based import")
            result = bpy.ops.sam3.import_game_objects(
                'EXEC_DEFAULT',
                filepath=go_json,
            )
            log.info("Import game objects result: %s", result)
    else:
        log.info("Step 7/8: Skipped%s.", " (--skip-game-objects)" if args.skip_game_objects else " (no game objects JSON)")

    # ------------------------------------------------------------------
    # Assign 'hidden' material to all collision mesh objects
    # ------------------------------------------------------------------
    hidden_count = _assign_hidden_material_to_collision()
    if hidden_count > 0:
        log.info("Assigned 'hidden' material to %d collision objects", hidden_count)

    # ------------------------------------------------------------------
    # Step 8: Texture processing + final save
    # ------------------------------------------------------------------
    log.info("Step 8/8: Processing textures and saving...")

    # Save first to establish the blend directory (needed by unpack_textures)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=output)

    if not args.skip_textures:
        bpy.ops.sam3.unpack_textures()
        bpy.ops.sam3.convert_textures_png()
        bpy.ops.sam3.convert_materials_bsdf()
    else:
        log.info("Texture processing skipped (--skip-textures)")

    # Final save
    bpy.ops.wm.save_as_mainfile(filepath=output)
    log.info("Done! Output saved to: %s", output)


if __name__ == "__main__":
    main()
