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
import time

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
# Progress reporting (structured lines for webtools dashboard)
# ---------------------------------------------------------------------------
def _emit_progress(pct, msg=""):
    """Emit structured progress line for webtools dashboard."""
    print(f"@@PROGRESS@@ {max(0,min(100,int(pct)))} {msg}".rstrip(), flush=True)


# Step time-weights (seconds, from real profiling).
# Step 4 (tile import) dominates; Step 5 (surface extraction) is second.
_STEP_WEIGHT = {1: 2, 2: 2, 3: 5, 4: 120, 5: 40, 6: 2, 7: 3, 8: 20}
_TOTAL_WEIGHT = sum(_STEP_WEIGHT.values())  # 194

# Cumulative weight at the *start* of each step (0-based fraction).
_STEP_START: dict[int, float] = {}
_cum = 0.0
for _sn in sorted(_STEP_WEIGHT):
    _STEP_START[_sn] = _cum / _TOTAL_WEIGHT
    _cum += _STEP_WEIGHT[_sn]
del _cum, _sn


def _step_pct(step: int, sub_frac: float = 0.0) -> int:
    """Map step number + intra-step fraction (0-1) to global 0-100 pct."""
    start = _STEP_START.get(step, 0)
    weight = _STEP_WEIGHT.get(step, 10) / _TOTAL_WEIGHT
    return int((start + weight * min(max(sub_frac, 0.0), 1.0)) * 100)


# ---------------------------------------------------------------------------
# Geo-conversion helpers
# ---------------------------------------------------------------------------

def _read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_tileset_tree(tiles_dir: str, CTile):
    """Load the CTile tree from *tiles_dir*.

    Tries ``tiles_dir/tileset.json`` first.  If that does not exist (common
    with block-based 3D Tiles datasets), walks subdirectories for individual
    ``tileset.json`` files and assembles them under a virtual root.
    """
    top_json = os.path.join(tiles_dir, "tileset.json")
    root_tile = CTile()
    if os.path.isfile(top_json):
        root_tile.loadFromRootJson(top_json)
        return root_tile

    # No root tileset.json — scan subdirectories for block-level tilesets
    found = 0
    for dirpath, _dirnames, filenames in os.walk(tiles_dir):
        for fname in filenames:
            if fname.lower() == "tileset.json":
                child = CTile()
                child.loadFromRootJson(os.path.join(dirpath, fname))
                if child.children or child.hasMesh:
                    root_tile.children.append(child)
                    child.parent = root_tile
                    found += 1
    if found:
        root_tile.canRefine = True
        log.info("Assembled virtual root from %d block tileset(s)", found)
    else:
        log.warning("No tileset.json found in %s or subdirectories", tiles_dir)
    return root_tile


def _pixel_to_geo(px: float, py: float, geo_meta: dict) -> list:
    """Convert pixel [x, y] in modelscale image to WGS84 [lon, lat].

    When *geo_meta* contains ``corners`` (top_left/top_right/bottom_left/
    bottom_right as [lat, lon]), uses bilinear interpolation to correctly
    handle UTM grid convergence.  Falls back to simplified rectangle otherwise.
    """
    w = geo_meta["image_width"]
    h = geo_meta["image_height"]
    corners = geo_meta.get("corners")
    if corners:
        u = float(px) / w
        v = float(py) / h
        # corners in geo_metadata.json are [lat, lon] — swap to [lon, lat]
        tl = corners["top_left"]
        tr = corners["top_right"]
        bl = corners["bottom_left"]
        br = corners["bottom_right"]
        lon = (1 - u) * (1 - v) * tl[1] + u * (1 - v) * tr[1] + \
              (1 - u) * v * bl[1] + u * v * br[1]
        lat = (1 - u) * (1 - v) * tl[0] + u * (1 - v) * tr[0] + \
              (1 - u) * v * bl[0] + u * v * br[0]
        return [lon, lat]
    bounds = geo_meta["bounds"]
    lon = bounds["west"] + float(px) * (bounds["east"] - bounds["west"]) / w
    lat = bounds["north"] - float(py) * (bounds["north"] - bounds["south"]) / h
    return [lon, lat]


def _setup_viewport_topdown() -> None:
    """Set up 3D viewport for visual monitoring (non-background mode only).

    - Orthographic top-down view (looking along -Y onto the XZ track plane)
    - Far clip plane at 10 000 m to avoid clipping
    - Frames all objects
    """
    screen = bpy.context.screen
    if screen is None:
        return

    from mathutils import Quaternion as Quat

    # Find the first 3D viewport
    area_3d = None
    for area in screen.areas:
        if area.type == "VIEW_3D":
            area_3d = area
            break
    if area_3d is None:
        log.warning("No 3D viewport found, skipping viewport setup")
        return

    space = area_3d.spaces.active
    r3d = space.region_3d

    # Far clip
    space.clip_end = 10000.0

    # Orthographic, looking down from +Y onto XZ plane
    # Rotation: +90° around X  →  view -Z maps to world +Y (camera above, looking down)
    r3d.view_perspective = "ORTHO"
    r3d.view_rotation = Quat((0.7071068, 0.7071068, 0.0, 0.0))

    # Frame all objects in viewport
    region = None
    for r in area_3d.regions:
        if r.type == "WINDOW":
            region = r
            break
    if region is not None:
        with bpy.context.temp_override(area=area_3d, region=region):
            bpy.ops.view3d.view_all()

    log.info("Viewport set to orthographic top-down (clip_end=10000)")


def _force_redraw() -> None:
    """Force a viewport redraw and pump Windows messages.

    The DRAW_WIN_SWAP redraws the viewport but does NOT process the
    Windows message queue, so the OS still marks Blender as 'Not Responding'.
    We explicitly pump pending messages via PeekMessageW to prevent that.
    """
    if bpy.app.background:
        return
    try:
        bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)
    except Exception:
        pass
    # Pump Windows message queue to prevent "Not Responding"
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes
            _user32 = ctypes.windll.user32
            _msg = wintypes.MSG()
            _PM_REMOVE = 0x0001
            for _ in range(20):
                if not _user32.PeekMessageW(ctypes.byref(_msg), None, 0, 0, _PM_REMOVE):
                    break
                _user32.TranslateMessage(ctypes.byref(_msg))
                _user32.DispatchMessageW(ctypes.byref(_msg))
        except Exception:
            pass


def _make_throttled_redraw(interval: float = 1.0):
    """Create a callback that redraws at most once per *interval* seconds.

    As the scene grows heavier, per-tile redraws get expensive.  Throttling
    keeps visual feedback smooth without accumulating redraw overhead.
    """
    state = {"last": 0.0}

    def _cb():
        now = time.monotonic()
        if now - state["last"] >= interval:
            _force_redraw()
            state["last"] = now

    return _cb


def _get_terrain_y_bounds() -> tuple:
    """Get Y (up) bounds of all terrain mesh objects in the scene.

    Excludes mask polygon and collision collections to only measure terrain.
    Returns (min_y, max_y). Falls back to (0.0, 20.0) if no terrain found.
    """
    import config as blender_config

    excluded_objs: set = set()
    for col in bpy.data.collections:
        if col.name in ("collision", "game_objects") or col.name.startswith("collision_"):
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
    """Assign 'hidden' material to all mesh objects in collision collections.

    Covers the legacy ``collision`` collection as well as per-tag collections
    like ``collision_road``, ``collision_grass``, etc.
    """
    mat = _ensure_hidden_material()
    count = 0
    for col in bpy.data.collections:
        if col.name == "collision" or col.name.startswith("collision_"):
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

    col = _ensure_collection("collision_walls")
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

        bmesh.ops.triangulate(bm, faces=bm.faces[:])
        bm.to_mesh(mesh)
        bm.free()
        mesh.update()

        obj = bpy.data.objects.new(name, mesh)
        col.objects.link(obj)
        created += 1

    return created


def _create_game_object_empty(
    obj_data: dict,
    blender_name: str,
    geo_meta: dict,
    tf_info,
    height_offset: float,
) -> "bpy.types.Object | None":
    """Create a single game-object Empty from *obj_data*.

    Returns the new Empty, or ``None`` if the position is invalid.
    Helper shared by both single-layout and multi-layout paths.
    """
    from geo_sam3_blender_utils import geo_points_to_blender_xyz

    pos = obj_data.get("position")
    if not pos or len(pos) < 2:
        return None

    px, py = float(pos[0]), float(pos[1])

    # Position: pixel -> WGS84 -> Blender
    pos_geo = [_pixel_to_geo(px, py, geo_meta)]
    pos_blender = geo_points_to_blender_xyz(pos_geo, tf_info, z_mode="zero")
    if not pos_blender:
        return None
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

    empty = bpy.data.objects.new(blender_name, None)
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

    return empty


def _import_game_objects_geo(
    go_json_path: str,
    geo_meta: dict,
    tf_info,
    height_offset: float = 2.0,
) -> int:
    """Import game objects using geo-converted coordinates.

    Converts pixel positions and orientations from the game objects JSON to
    Blender 3D coordinates.  Objects are created as Empty markers.

    **Multi-layout support**: when the JSON contains a ``layouts`` list with
    more than one entry, objects are grouped into sub-collections
    ``game_objects_{LayoutName}`` under a root ``game_objects`` collection.
    Each object's Blender name is prefixed ``{LayoutName}__`` to avoid
    duplicates (e.g. ``LayoutCW__AC_HOTLAP_START_0``).  A custom property
    ``_layout`` stores the layout name for downstream use.

    Single-layout (no ``layouts`` key or only one layout): behaviour is
    unchanged — all objects go directly into ``game_objects``.
    """
    data = _read_json(go_json_path)
    objects_list = data.get("objects", [])
    if not objects_list:
        log.warning("No objects found in %s", go_json_path)
        return 0

    layout_names = data.get("layouts", [])
    is_multi = len(layout_names) > 1

    root_col = _ensure_collection("game_objects")
    created = 0

    if is_multi:
        log.info("Multi-layout mode: %s", layout_names)

        # Group objects by _layout
        by_layout: dict[str, list] = {}
        for obj_data in objects_list:
            layout = obj_data.get("_layout", "Default")
            by_layout.setdefault(layout, []).append(obj_data)

        for layout_name, layout_objs in by_layout.items():
            sub_col_name = f"game_objects_{layout_name}"
            sub_col = bpy.data.collections.get(sub_col_name)
            if sub_col is None:
                sub_col = bpy.data.collections.new(sub_col_name)
            # Link sub_col under root_col (not scene root)
            if sub_col.name not in [c.name for c in root_col.children]:
                root_col.children.link(sub_col)

            for obj_data in layout_objs:
                original_name = obj_data.get("name", f"OBJECT_{created}")
                blender_name = f"{layout_name}__{original_name}"

                empty = _create_game_object_empty(
                    obj_data, blender_name, geo_meta, tf_info, height_offset,
                )
                if empty is None:
                    continue

                empty["_layout"] = layout_name
                sub_col.objects.link(empty)
                created += 1

            log.info("  Layout '%s': %d objects in '%s'",
                     layout_name, len(layout_objs), sub_col_name)
    else:
        # Single-layout: original behaviour
        for obj_data in objects_list:
            name = obj_data.get("name", f"OBJECT_{created}")

            empty = _create_game_object_empty(
                obj_data, name, geo_meta, tf_info, height_offset,
            )
            if empty is None:
                continue

            root_col.objects.link(empty)
            created += 1

    return created


def _project_game_objects_to_surface(height_above: float = 2.0) -> int:
    """Raycast each game-object Empty downward onto terrain tile surfaces.

    Sets ``empty.location.y = hit_y + height_above`` for every Empty in any
    ``game_objects*`` collection.  Returns the number of objects projected.

    Only accepts hits on terrain tile objects (those in ``L{digits}``
    collections), so mask polygons and collision meshes are ignored without
    needing to hide them (which doesn't work in ``--background`` mode).
    """
    # Collect empties from game_objects collections
    empties: list[bpy.types.Object] = []
    for col in bpy.data.collections:
        if col.name == "game_objects" or col.name.startswith("game_objects_"):
            for obj in col.all_objects:
                if obj.type == "EMPTY":
                    empties.append(obj)
    if not empties:
        return 0

    # Build set of terrain tile object names (in L{digits} collections)
    terrain_names: set[str] = set()
    for col in bpy.data.collections:
        if col.name.startswith("L") and col.name[1:].isdigit():
            for obj in col.all_objects:
                if obj.type == "MESH":
                    terrain_names.add(obj.name)

    if not terrain_names:
        log.warning("No terrain tile objects found for game object projection")
        return 0

    projected = 0
    ray_dir = Vector((0.0, 1.0, 0.0))   # gravity is +Y, cast toward ground
    ray_origin_y = -5000.0               # start high above (negative Y = up)

    # Per-object raycast against individual terrain meshes
    # (avoids scene.ray_cast which requires hide_viewport to work)
    for empty in empties:
        ex, ez = empty.location.x, empty.location.z
        best_y = None

        for obj_name in terrain_names:
            obj = bpy.data.objects.get(obj_name)
            if obj is None or obj.data is None:
                continue
            # Quick AABB check in world space
            bb = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
            xs = [v.x for v in bb]
            zs = [v.z for v in bb]
            if ex < min(xs) or ex > max(xs) or ez < min(zs) or ez > max(zs):
                continue

            # Ray in object local space
            inv = obj.matrix_world.inverted()
            local_origin = inv @ Vector((ex, ray_origin_y, ez))
            local_dir = (inv.to_3x3() @ ray_dir).normalized()
            hit, loc, _n, _idx = obj.ray_cast(local_origin, local_dir)
            if hit:
                world_loc = obj.matrix_world @ loc
                if best_y is None or world_loc.y < best_y:
                    best_y = world_loc.y

        if best_y is not None:
            empty.location.y = best_y - height_above
            projected += 1

    return projected


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
    p.add_argument("--polygon-dir", default="",
                    help="Stage 8 gap_filled polygon directory for tile refinement plan")
    p.add_argument("--refine-tags", default="road",
                    help="Comma-separated mask tags for refinement (default: road)")
    p.add_argument("--tile-padding", type=float, default=0.0,
                    help="Padding around polygon AABBs for tile plan (metres, default: 0)")
    p.add_argument("--edge-simplify", type=float, default=0.0,
                    help="Edge simplification epsilon in metres (0 = no simplification)")
    p.add_argument("--density-road", type=float, default=0.1,
                    help="Sampling density for road surfaces in metres")
    p.add_argument("--density-kerb", type=float, default=0.1,
                    help="Sampling density for kerb surfaces in metres")
    p.add_argument("--density-grass", type=float, default=2.0,
                    help="Sampling density for grass surfaces in metres")
    p.add_argument("--density-sand", type=float, default=2.0,
                    help="Sampling density for sand surfaces in metres")
    p.add_argument("--density-road2", type=float, default=2.0,
                    help="Sampling density for road2 surfaces in metres")
    p.add_argument("--mesh-simplify", action="store_true",
                    help="Enable mesh weld + decimate for terrain collision meshes")
    p.add_argument("--mesh-weld-distance", type=float, default=0.01,
                    help="Weld distance in metres (default: 0.01)")
    p.add_argument("--mesh-decimate-ratio", type=float, default=0.5,
                    help="Decimate ratio 0-1 (default: 0.5)")
    return p.parse_args(_get_script_argv())


# ---------------------------------------------------------------------------
# Mesh simplification (terrain extraction post-processing)
# ---------------------------------------------------------------------------

def _simplify_terrain_meshes(weld_distance: float, decimate_ratio: float) -> None:
    """Weld nearby vertices and decimate terrain collision meshes.

    Only processes MESH objects in ``collision_road`` and ``collision_kerb``.
    """
    target_collections = ["collision_road", "collision_kerb"]
    for col_name in target_collections:
        col = bpy.data.collections.get(col_name)
        if col is None:
            continue
        for obj in list(col.all_objects):
            if obj.type != "MESH":
                continue
            mesh = obj.data
            verts_before = len(mesh.vertices)
            faces_before = len(mesh.polygons)

            # Step 1: Weld (merge by distance)
            bm = bmesh.new()
            bm.from_mesh(mesh)
            bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=weld_distance)
            bm.to_mesh(mesh)
            bm.free()
            mesh.update()

            verts_after_weld = len(mesh.vertices)

            # Step 2: Decimate via modifier
            try:
                bpy.ops.object.select_all(action="DESELECT")
                obj.select_set(True)
                bpy.context.view_layer.objects.active = obj
                mod = obj.modifiers.new(name="Simplify", type="DECIMATE")
                mod.decimate_type = "COLLAPSE"
                mod.ratio = decimate_ratio
                bpy.ops.object.modifier_apply(modifier=mod.name)
            except Exception as e:
                log.warning("  %s: decimate failed: %s", obj.name, e)
                # Clean up modifier if apply failed
                if obj.modifiers.get("Simplify"):
                    obj.modifiers.remove(obj.modifiers["Simplify"])
                continue

            verts_final = len(mesh.vertices)
            faces_final = len(mesh.polygons)
            log.info("  %s: %d→%d verts (weld), %d→%d faces (decimate)",
                     obj.name, verts_before, verts_after_weld, faces_before, faces_final)

    log.info("Terrain mesh simplification complete (weld=%.4fm, ratio=%.2f)",
             weld_distance, decimate_ratio)


# ---------------------------------------------------------------------------
# Main automation pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # Force line-buffered stdout/stderr so that log output from Blender's
    # embedded Python appears immediately in the pipeline log file, instead
    # of being held in a full 8 KB C-stdio buffer until it fills.
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(line_buffering=True)
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass  # best-effort; older Blender builds may not support this

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
    config.SURFACE_EDGE_SIMPLIFY = args.edge_simplify
    config.SURFACE_SAMPLING_DENSITY_ROAD = args.density_road
    config.SURFACE_SAMPLING_DENSITY_KERB = args.density_kerb
    config.SURFACE_SAMPLING_DENSITY_GRASS = args.density_grass
    config.SURFACE_SAMPLING_DENSITY_SAND = args.density_sand
    config.SURFACE_SAMPLING_DENSITY_ROAD2 = args.density_road2
    config.MESH_SIMPLIFY = args.mesh_simplify
    config.MESH_WELD_DISTANCE = args.mesh_weld_distance
    config.MESH_DECIMATE_RATIO = args.mesh_decimate_ratio

    # ------------------------------------------------------------------
    # Step 1: Open the input .blend file
    # ------------------------------------------------------------------
    _emit_progress(_step_pct(1), "Opening blend file...")
    log.info("Step 1/8: Opening blend file: %s", blend_input)
    bpy.ops.wm.open_mainfile(filepath=blend_input)

    # ------------------------------------------------------------------
    # Step 2: Register all SAM3 operators
    # ------------------------------------------------------------------
    _emit_progress(_step_pct(2), "Registering operators...")
    log.info("Step 2/8: Registering SAM3 operators...")
    import blender_helpers
    blender_helpers.register()

    # ------------------------------------------------------------------
    # Steps 3-8: the heavy pipeline work.
    # In non-background mode, defer via timer so Blender's GUI is ready
    # and the user can watch objects appear in the viewport.
    # ------------------------------------------------------------------
    def _continue_pipeline() -> None:
        nonlocal tf_info

        # Viewport setup (only effective in GUI mode, after event loop starts)
        _setup_viewport_topdown()
        _force_redraw()

        # Throttled redraw: at most once per second to avoid expensive
        # redraws piling up as the scene gets heavier with more tiles.
        _tile_redraw = _make_throttled_redraw(interval=1.0)

        # --------------------------------------------------------------
        # Step 3+4: Pre-compute tile load plan, then load all at once
        # --------------------------------------------------------------
        import time as _time
        from sam3_actions.load_base_tiles import load_glb_tiles_by_dic_level_array

        polygon_dir = os.path.abspath(args.polygon_dir) if args.polygon_dir else ""
        refine_tags = [t.strip() for t in args.refine_tags.split(",") if t.strip()]

        if polygon_dir and os.path.isdir(polygon_dir):
            # New plan-based approach: pre-compute which tiles to load
            _emit_progress(_step_pct(3), "Computing tile load plan...")
            log.info("Step 3/8: Computing tile load plan (base=%d, target=%d)...",
                     args.base_level, args.target_level)
            log.info("  polygon_dir: %s", polygon_dir)
            log.info("  refine_tags: %s", refine_tags)

            from tile_plan import compute_plan_from_config
            _t3 = _time.time()
            plan = compute_plan_from_config(
                tiles_dir=tiles_dir,
                polygon_dir=polygon_dir,
                tags=refine_tags,
                base_level=args.base_level,
                target_level=args.target_level,
                padding_m=args.tile_padding,
            )
            plan_time = _time.time() - _t3
            total_tiles = sum(len(v) for v in plan.values())

            # --- Tile manifest: build text, log it, and save to file ---
            manifest_lines = []
            manifest_lines.append("=" * 60)
            manifest_lines.append("TILE LOAD PLAN  (base={}, target={})".format(
                args.base_level, args.target_level))
            manifest_lines.append("  polygon_dir: {}".format(polygon_dir))
            manifest_lines.append("  refine_tags: {}".format(refine_tags))
            manifest_lines.append("  padding_m: {}".format(args.tile_padding))
            manifest_lines.append("=" * 60)
            for lv in sorted(plan.keys()):
                tiles_at_lv = plan[lv]
                manifest_lines.append("  Level {}: {} tiles".format(lv, len(tiles_at_lv)))
                for t in tiles_at_lv:
                    content = t.content or "(no content)"
                    manifest_lines.append("    - {}".format(content))
            manifest_lines.append("-" * 60)
            manifest_lines.append("  TOTAL: {} tiles across {} levels".format(
                total_tiles, len(plan)))
            manifest_lines.append("  Plan computed in {:.1f}s".format(plan_time))
            manifest_lines.append("=" * 60)

            for line in manifest_lines:
                log.info(line)

            # Save manifest to Stage 9 output directory
            manifest_path = os.path.join(os.path.dirname(output), "tile_load_plan.txt")
            with open(manifest_path, "w", encoding="utf-8") as _mf:
                _mf.write("\n".join(manifest_lines) + "\n")
            log.info("Tile load plan saved to %s", manifest_path)

            # Load all tiles in one pass
            _emit_progress(_step_pct(4), "Loading tiles...")
            log.info("Step 4/8: Loading %d planned tiles...", total_tiles)
            # Emit "need to load" lines for Dashboard progress parser
            # (load_glb_tiles_by_dic_level_array doesn't print these;
            #  they're normally emitted by import_fullscene_with_ctile)
            for lv in sorted(plan.keys()):
                print("need to load level {}:{} tiles".format(lv, len(plan[lv])))
            _tiles_loaded_count = [0]

            def _tile_progress_cb():
                _tile_redraw()
                _tiles_loaded_count[0] += 1
                if total_tiles > 0 and _tiles_loaded_count[0] % 5 == 0:
                    # Quadratic mapping: per-tile import time grows linearly
                    # with scene size, so elapsed time T(i) ∝ i².
                    # Use sub = (i/N)² so progress advances at constant speed.
                    sub = (_tiles_loaded_count[0] / total_tiles) ** 2
                    _emit_progress(_step_pct(4, sub),
                                   f"Tiles {_tiles_loaded_count[0]}/{total_tiles}")

            _t4 = _time.time()
            load_glb_tiles_by_dic_level_array(glb_dir, plan, on_tile_loaded=_tile_progress_cb)
            log.info("All tiles loaded in %.1fs.", _time.time() - _t4)
        else:
            # Fallback: no polygon dir → load all base tiles (old Step 3 only)
            _emit_progress(_step_pct(3), "Loading base tiles...")
            log.info("Step 3/8: Loading base tiles (level=%d, no polygon plan)...",
                     args.base_level)
            from sam3_actions.c_tiles import CTile
            from sam3_actions.load_base_tiles import import_fullscene_with_ctile

            _t3 = _time.time()
            root_tile = _load_tileset_tree(tiles_dir, CTile)
            import_fullscene_with_ctile(root_tile, glb_dir, min_level=args.base_level,
                                        on_tile_loaded=_tile_redraw)
            log.info("Base tiles loaded in %.1fs.", _time.time() - _t3)

            # Old Step 4: iterative refinement (fallback when no polygon dir)
            _emit_progress(_step_pct(4), "Refining tiles by mask...")
            log.info("Step 4/8: Refining tiles by mask to level %d...", args.target_level)
            from sam3_actions.load_base_tiles import refine_by_mask_sync

            mask_col = bpy.data.collections.get(config.ROOT_POLYGON_COLLECTION_NAME)
            masks = []
            if mask_col is not None:
                for tag in refine_tags:
                    sub_col = mask_col.children.get(f"mask_polygon_{tag}")
                    if sub_col is not None:
                        masks.extend(list(sub_col.all_objects))
            log.info("Refinement mask tags: %s (%d objects)", refine_tags, len(masks))

            if masks:
                _t4 = _time.time()
                root_tile2 = _load_tileset_tree(tiles_dir, CTile)
                refine_by_mask_sync(
                    context=bpy.context,
                    masks=masks,
                    root_tile=root_tile2,
                    glb_dir=glb_dir,
                    target_level=args.target_level,
                    on_tile_loaded=_tile_redraw,
                )
                log.info("Tile refinement complete in %.1fs.", _time.time() - _t4)
            else:
                log.warning("No mask objects found in '%s', skipping refinement.",
                             config.ROOT_POLYGON_COLLECTION_NAME)

        _force_redraw()

        # Hide mask polygons so tiles are clearly visible
        mask_root = bpy.data.collections.get(config.ROOT_POLYGON_COLLECTION_NAME)
        if mask_root is not None:
            mask_root.hide_viewport = True
            log.info("Mask polygons hidden for better visibility.")
            _force_redraw()

        # --------------------------------------------------------------
        # Prepare geo-transform and terrain bounds
        # --------------------------------------------------------------
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

        # --------------------------------------------------------------
        # Step 5: Extract collision surfaces
        # --------------------------------------------------------------
        if not args.skip_surfaces:
            _emit_progress(_step_pct(5), "Extracting collision surfaces...")
            log.info("Step 5/8: Extracting collision surfaces...")

            # Set sub-progress ranges for sub-modules
            import sam3_actions.terrain_mesh_extractor as _tme
            _tme.PROGRESS_RANGE = (_step_pct(5, 0.0), _step_pct(5, 0.45))
            import sam3_actions.boolean_mesh_generator as _bmg
            _bmg.PROGRESS_RANGE = (_step_pct(5, 0.5), _step_pct(5, 1.0))

            log.info("  Step 5a: Terrain extraction (road + kerb)...")
            result_a = bpy.ops.sam3.extract_terrain_surfaces()
            log.info("  Terrain extraction result: %s", result_a)

            if args.mesh_simplify:
                log.info("  Step 5a+: Simplifying terrain meshes...")
                _simplify_terrain_meshes(args.mesh_weld_distance, args.mesh_decimate_ratio)

            log.info("  Step 5b: Boolean surfaces (grass/sand/road2)...")
            result_b = bpy.ops.sam3.generate_boolean_surfaces()
            log.info("  Boolean surfaces result: %s", result_b)
        else:
            _emit_progress(_step_pct(5), "Skipped surfaces")
            log.info("Step 5/8: Skipped (--skip-surfaces)")
        _force_redraw()

        # --------------------------------------------------------------
        # Step 6: Import virtual walls (optional)
        # --------------------------------------------------------------
        if not args.skip_walls and walls_json and os.path.isfile(walls_json):
            _emit_progress(_step_pct(6), "Importing walls...")
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
        _force_redraw()

        # --------------------------------------------------------------
        # Step 7: Import game objects (optional)
        # --------------------------------------------------------------
        if not args.skip_game_objects and go_json and os.path.isfile(go_json):
            _emit_progress(_step_pct(7), "Importing game objects...")
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
            # Project game objects down onto terrain surface + 2m
            projected = _project_game_objects_to_surface(height_above=2.0)
            log.info("Projected %d game objects onto terrain (2m above surface)", projected)
        else:
            log.info("Step 7/8: Skipped%s.", " (--skip-game-objects)" if args.skip_game_objects else " (no game objects JSON)")

        # --------------------------------------------------------------
        # Assign 'hidden' material to all collision mesh objects
        # --------------------------------------------------------------
        hidden_count = _assign_hidden_material_to_collision()
        if hidden_count > 0:
            log.info("Assigned 'hidden' material to %d collision objects", hidden_count)

        # --------------------------------------------------------------
        # Step 8: Texture processing + final save
        # --------------------------------------------------------------
        _emit_progress(_step_pct(8), "Processing textures and saving...")
        log.info("Step 8/8: Processing textures and saving...")

        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        _emit_progress(_step_pct(8, 0.3), "Saving blend file...")
        bpy.ops.wm.save_as_mainfile(filepath=output)

        if not args.skip_textures:
            _emit_progress(_step_pct(8, 0.5), "Unpacking textures...")
            bpy.ops.sam3.unpack_textures()
            _emit_progress(_step_pct(8, 0.7), "Converting textures...")
            bpy.ops.sam3.convert_textures_png()
            _emit_progress(_step_pct(8, 0.85), "Converting materials...")
            bpy.ops.sam3.convert_materials_bsdf()
        else:
            log.info("Texture processing skipped (--skip-textures)")

        # Final save
        bpy.ops.wm.save_as_mainfile(filepath=output)
        _emit_progress(100, "Done")
        log.info("Done! Output saved to: %s", output)

    # ------------------------------------------------------------------
    # Dispatch: background → synchronous, GUI → deferred via timer
    # ------------------------------------------------------------------
    if bpy.app.background:
        _continue_pipeline()
    else:
        def _timer_wrapper():
            try:
                _continue_pipeline()
            except Exception:
                log.exception("Pipeline failed in GUI mode")
            return None  # one-shot, do not repeat
        bpy.app.timers.register(_timer_wrapper, first_interval=5.0)
        log.info("GUI mode: pipeline deferred 5s for viewport init...")


if __name__ == "__main__":
    main()
