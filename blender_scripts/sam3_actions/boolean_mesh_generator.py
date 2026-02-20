"""
Blender action: Generate Boolean Surfaces (Grass / Sand / Road2).

Creates a regular Grid plane at the configured density, boolean-intersects it
with a solidified mask polygon to cut the shape, then raycasts every vertex
downward onto the terrain to project the flat grid into 3D.

Algorithm
---------
1. Read mask mesh XZ bounding box, expand slightly.
2. Create a Grid mesh with subdivisions = ceil(extent / density).
3. Copy + Solidify the mask (extrude ±Y) to form a closed volume.
4. Boolean Intersect: grid ∩ solidified mask → cut grid to mask shape.
5. Raycast each remaining vertex downward (-Y) onto terrain.
6. Remove vertices that miss terrain.
7. Place result in the appropriate ``collision_{tag}`` collection.

Two operators:
- ``sam3.generate_boolean_surfaces``          — batch: grass + sand + road2
- ``sam3.generate_boolean_surface_selected``  — manual: selected mask(s)
"""

from __future__ import annotations

import math
import os
import sys
import time
from typing import Optional

import bpy  # type: ignore[import-not-found]
import bmesh  # type: ignore[import-not-found]
from mathutils import Vector  # type: ignore[import-not-found]

from . import ActionSpec

# ---------------------------------------------------------------------------
# Ensure script/ is importable
# ---------------------------------------------------------------------------
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_script_dir = os.path.join(_project_root, "script")
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from surface_extraction import (  # noqa: E402
    COLLISION_COLLECTION_MAP,
    generate_collision_name,
)
from config import (  # noqa: E402
    ROOT_POLYGON_COLLECTION_NAME,
    SURFACE_SAMPLING_DENSITY_DEFAULT,
    SURFACE_SAMPLING_DENSITY_GRASS,
    SURFACE_SAMPLING_DENSITY_ROAD2,
    SURFACE_SAMPLING_DENSITY_SAND,
)

BOOLEAN_TAGS = ["grass", "sand", "road2"]

_DENSITY_MAP: dict[str, float] = {
    "grass": SURFACE_SAMPLING_DENSITY_GRASS,
    "sand": SURFACE_SAMPLING_DENSITY_SAND,
    "road2": SURFACE_SAMPLING_DENSITY_ROAD2,
}


def _get_density(tag: str) -> float:
    return _DENSITY_MAP.get(tag.strip().lower(), SURFACE_SAMPLING_DENSITY_DEFAULT)


# ---------------------------------------------------------------------------
# Logging / formatting helpers
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    sys.stderr.write(f"[boolean_mesh_generator] {msg}\n")
    sys.stderr.flush()


def _fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"


# ---------------------------------------------------------------------------
# Collection helpers
# ---------------------------------------------------------------------------

def _get_or_create_collection(name: str) -> bpy.types.Collection:
    scene_root = bpy.context.scene.collection
    for c in scene_root.children:
        if c.name == name:
            return c
    col = bpy.data.collections.new(name)
    scene_root.children.link(col)
    return col


def _link_to_collection(obj: bpy.types.Object, col: bpy.types.Collection) -> None:
    if obj.name not in col.objects:
        col.objects.link(obj)
    try:
        root = bpy.context.scene.collection
        if obj.name in root.objects:
            root.objects.unlink(obj)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Scene helpers
# ---------------------------------------------------------------------------

def _build_excluded_set() -> set[str]:
    """Return names of objects in mask_* / collision_* collections."""
    excluded: set[str] = set()
    for col in bpy.data.collections:
        name = col.name
        if (name.startswith("mask_")
                or name == "collision"
                or name.startswith("collision_")):
            for obj in col.all_objects:
                excluded.add(obj.name)
    return excluded


def _get_terrain_max_y() -> float:
    """Find the maximum Y (world-space) among non-excluded mesh objects."""
    excluded = _build_excluded_set()
    max_y = 0.0
    for obj in bpy.data.objects:
        if obj.type != "MESH" or obj.name in excluded:
            continue
        try:
            for corner in obj.bound_box:
                w = obj.matrix_world @ Vector(corner)
                if w.y > max_y:
                    max_y = w.y
        except Exception:
            continue
    return max_y


# ---------------------------------------------------------------------------
# Mask XZ bounding box
# ---------------------------------------------------------------------------

def _mask_xz_bounds(
    mask_obj: bpy.types.Object, depsgraph, margin: float = 1.0,
) -> tuple[float, float, float, float, float] | None:
    """Return (min_x, max_x, min_z, max_z, y_value) for a mask mesh.

    *margin* expands the box slightly.  *y_value* is the average Y of the
    mask vertices (they should all be near the same Y).
    """
    try:
        obj_eval = mask_obj.evaluated_get(depsgraph)
        bm = bmesh.new()
        try:
            bm.from_object(obj_eval, depsgraph)
        except Exception:
            bm.from_mesh(mask_obj.data)
    except Exception:
        return None

    if len(bm.verts) < 3:
        bm.free()
        return None

    mw = mask_obj.matrix_world
    min_x = min_z = float("inf")
    max_x = max_z = float("-inf")
    y_sum = 0.0

    for v in bm.verts:
        w = mw @ v.co
        min_x = min(min_x, w.x)
        max_x = max(max_x, w.x)
        min_z = min(min_z, w.z)
        max_z = max(max_z, w.z)
        y_sum += w.y

    y_value = y_sum / len(bm.verts) if len(bm.verts) > 0 else 0.0
    bm.free()

    return (min_x - margin, max_x + margin,
            min_z - margin, max_z + margin,
            y_value)


# ---------------------------------------------------------------------------
# Grid creation
# ---------------------------------------------------------------------------

def _create_grid_plane(
    min_x: float, max_x: float,
    min_z: float, max_z: float,
    density: float,
    y_value: float,
) -> bpy.types.Object:
    """Create a subdivided Grid mesh covering the given XZ bounds."""
    width_x = max_x - min_x
    width_z = max_z - min_z
    subdivs_x = max(1, math.ceil(width_x / density))
    subdivs_z = max(1, math.ceil(width_z / density))

    _log(f"  Grid: {subdivs_x}x{subdivs_z} = "
         f"{(subdivs_x + 1) * (subdivs_z + 1):,} verts "
         f"(density={density}m, {width_x:.1f}x{width_z:.1f}m)")

    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int, int]] = []
    cols = subdivs_x + 1

    for iz in range(subdivs_z + 1):
        for ix in range(subdivs_x + 1):
            x = min_x + ix * density
            z = min_z + iz * density
            verts.append((x, y_value, z))

    for iz in range(subdivs_z):
        for ix in range(subdivs_x):
            v0 = iz * cols + ix
            v1 = v0 + 1
            v2 = v0 + cols + 1
            v3 = v0 + cols
            faces.append((v0, v1, v2, v3))

    mesh = bpy.data.meshes.new("_bool_grid_tmp")
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    obj = bpy.data.objects.new("_bool_grid_tmp", mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


# ---------------------------------------------------------------------------
# Mask solidify (extrude to volume)
# ---------------------------------------------------------------------------

def _solidify_mask(
    mask_obj: bpy.types.Object,
    depsgraph,
    thickness: float = 200.0,
) -> bpy.types.Object | None:
    """Copy a mask mesh to world space and solidify it along Y.

    Returns a temporary solid object suitable for Boolean operations.
    The caller is responsible for cleanup.
    """
    bm = bmesh.new()
    try:
        obj_eval = mask_obj.evaluated_get(depsgraph)
        try:
            bm.from_object(obj_eval, depsgraph)
        except Exception:
            bm.from_mesh(mask_obj.data)
    except Exception:
        bm.free()
        return None

    if len(bm.verts) < 3 or len(bm.faces) == 0:
        bm.free()
        return None

    # Transform to world space
    mw = mask_obj.matrix_world
    bmesh.ops.transform(bm, matrix=mw, verts=bm.verts)

    # Pre-clean degenerate geometry that can break boolean
    bmesh.ops.dissolve_degenerate(bm, dist=0.001, edges=bm.edges)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.001)

    temp_mesh = bpy.data.meshes.new("_bool_solid_tmp")
    bm.to_mesh(temp_mesh)
    bm.free()

    temp_obj = bpy.data.objects.new("_bool_solid_tmp", temp_mesh)
    bpy.context.scene.collection.objects.link(temp_obj)

    # Solidify modifier: extrude along normals (Y axis for flat XZ mesh)
    mod = temp_obj.modifiers.new("solidify", "SOLIDIFY")
    mod.thickness = thickness
    mod.offset = 0.0  # Symmetric: half up, half down

    return temp_obj


# ---------------------------------------------------------------------------
# Boolean intersect
# ---------------------------------------------------------------------------

def _boolean_intersect(
    grid_obj: bpy.types.Object,
    solid_mask_obj: bpy.types.Object,
    depsgraph,
) -> bpy.types.Object | None:
    """Boolean INTERSECT *grid_obj* with *solid_mask_obj*.

    Returns a new object with the result, or None if the boolean produces
    empty geometry.  Caller is responsible for cleanup of inputs.
    """
    mod = grid_obj.modifiers.new("bool_intersect", "BOOLEAN")
    mod.operation = "INTERSECT"
    mod.object = solid_mask_obj
    mod.solver = "EXACT"

    depsgraph.update()

    # Read the evaluated (post-boolean) geometry
    bm = bmesh.new()
    try:
        grid_eval = grid_obj.evaluated_get(depsgraph)
        bm.from_object(grid_eval, depsgraph)
    except Exception as exc:
        bm.free()
        _log(f"  Boolean evaluation failed: {exc}")
        return None

    n_verts = len(bm.verts)
    n_faces = len(bm.faces)
    if n_verts < 3 or n_faces == 0:
        bm.free()
        _log("  Boolean produced empty result")
        return None

    result_mesh = bpy.data.meshes.new("_bool_result_tmp")
    bm.to_mesh(result_mesh)
    bm.free()
    result_mesh.update()

    result_obj = bpy.data.objects.new("_bool_result_tmp", result_mesh)
    bpy.context.scene.collection.objects.link(result_obj)

    _log(f"  Boolean result: {n_verts:,} verts, {n_faces:,} faces")
    return result_obj


# ---------------------------------------------------------------------------
# Terrain projection
# ---------------------------------------------------------------------------

def _project_to_terrain(
    obj: bpy.types.Object,
    scene: bpy.types.Scene,
    depsgraph,
    excluded: set[str],
    ray_origin_y: float,
) -> int:
    """Raycast each vertex of *obj* downward onto the terrain.

    Vertices that miss terrain are deleted.  Returns the number of hits.
    """
    me = obj.data
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()

    direction = Vector((0.0, -1.0, 0.0))
    hit_count = 0
    miss_verts: list = []
    total_v = len(bm.verts)
    t0 = time.monotonic()
    last_pct = -1

    for vi, v in enumerate(bm.verts):
        pct = (vi * 100) // total_v if total_v else 100
        if pct >= last_pct + 10:
            last_pct = pct
            elapsed = time.monotonic() - t0
            _log(f"  Projection: {pct}% ({vi:,}/{total_v:,}) "
                 f"elapsed {_fmt_time(elapsed)}")

        origin = Vector((v.co.x, ray_origin_y, v.co.z))
        hit = False
        for _ in range(5):
            result, location, _nrm, _idx, hit_obj, _mtx = scene.ray_cast(
                depsgraph, origin, direction,
            )
            if not result:
                break
            if hit_obj is None or hit_obj.name not in excluded:
                v.co = location
                hit_count += 1
                hit = True
                break
            origin = location + direction * 0.001
        if not hit:
            miss_verts.append(v)

    if miss_verts:
        _log(f"  Removing {len(miss_verts):,} verts with no terrain hit")
        bmesh.ops.delete(bm, geom=miss_verts, context="VERTS")

    bm.to_mesh(me)
    bm.free()
    me.update()

    elapsed = time.monotonic() - t0
    _log(f"  Projection done: {hit_count:,}/{total_v:,} hits "
         f"({_fmt_time(elapsed)})")
    return hit_count


# ---------------------------------------------------------------------------
# Cleanup helpers
# ---------------------------------------------------------------------------

def _safe_remove_object(obj: bpy.types.Object | None) -> None:
    """Remove a temporary Blender object + its mesh data."""
    if obj is None:
        return
    mesh = obj.data if obj.type == "MESH" else None
    try:
        bpy.data.objects.remove(obj, do_unlink=True)
    except Exception:
        pass
    if mesh is not None:
        try:
            if mesh.users == 0:
                bpy.data.meshes.remove(mesh)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Core: single mask → boolean surface
# ---------------------------------------------------------------------------

def generate_boolean_surface(
    mask_obj: bpy.types.Object,
    tag: str,
    density: float,
    scene: bpy.types.Scene,
    depsgraph,
    ray_origin_y: float,
    excluded: set[str],
) -> bpy.types.Object | None:
    """Generate a collision surface for *tag* by boolean-cutting a grid.

    Returns the final collision mesh object, or None on failure.
    """
    t0 = time.monotonic()
    label = f"{tag}/{mask_obj.name}"
    _log(f"--- {label}: boolean grid (density={density}m) ---")

    # 1. Mask XZ bounds
    bounds = _mask_xz_bounds(mask_obj, depsgraph, margin=density)
    if bounds is None:
        _log(f"  {label}: cannot read mask bounds, skipping")
        return None
    min_x, max_x, min_z, max_z, y_value = bounds

    # 2. Create grid
    grid_obj = _create_grid_plane(min_x, max_x, min_z, max_z, density, y_value)

    # 3. Solidify mask
    solid_obj = _solidify_mask(mask_obj, depsgraph, thickness=200.0)
    if solid_obj is None:
        _log(f"  {label}: mask solidify failed, skipping")
        _safe_remove_object(grid_obj)
        return None

    # Update depsgraph so modifiers evaluate
    depsgraph.update()

    # 4. Boolean intersect
    result_obj = _boolean_intersect(grid_obj, solid_obj, depsgraph)

    # Clean up intermediates (grid + solid)
    _safe_remove_object(grid_obj)
    _safe_remove_object(solid_obj)
    depsgraph.update()

    if result_obj is None:
        _log(f"  {label}: boolean failed — mask may have self-intersections")
        return None

    # 5. Project to terrain
    # Add result to excluded so raycasts don't hit it
    excluded_local = excluded | {result_obj.name}
    hit_count = _project_to_terrain(
        result_obj, scene, depsgraph, excluded_local, ray_origin_y,
    )

    # 5b. Clean up degenerate faces created by projection
    #     (vertices collapsed to same location → zero-area triangles)
    me = result_obj.data
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.dissolve_degenerate(bm, dist=0.001, edges=bm.edges)
    n_before = len(bm.faces)
    # Also remove loose vertices left behind
    loose = [v for v in bm.verts if not v.link_faces]
    if loose:
        bmesh.ops.delete(bm, geom=loose, context="VERTS")
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bm.to_mesh(me)
    bm.free()
    me.update()
    n_after = len(me.polygons)
    if n_before != n_after:
        _log(f"  {label}: cleaned {n_before - n_after} degenerate faces")

    # Check if enough vertices survived
    n_verts = len(result_obj.data.vertices)
    n_faces = len(result_obj.data.polygons)
    if n_verts < 3 or n_faces == 0:
        _log(f"  {label}: too few geometry after projection, skipping")
        _safe_remove_object(result_obj)
        return None

    # 6. Rename and place in correct collection
    col_name = COLLISION_COLLECTION_MAP.get(tag, f"collision_{tag}")
    col = _get_or_create_collection(col_name)

    try:
        prefix = generate_collision_name(tag, 0).rsplit("_", 1)[0]
    except ValueError:
        prefix = f"1{tag.upper()}"
    idx = sum(1 for o in col.all_objects
              if o.type == "MESH" and o.name.startswith(prefix))
    try:
        obj_name = generate_collision_name(tag, idx)
    except ValueError:
        obj_name = f"1{tag.upper()}_{idx}"

    result_obj.name = obj_name
    result_obj.data.name = f"{obj_name}_mesh"
    _link_to_collection(result_obj, col)

    elapsed = time.monotonic() - t0
    _log(f"  => {obj_name}: {n_verts:,} verts, {n_faces:,} faces "
         f"-> {col_name} ({_fmt_time(elapsed)} total)")
    return result_obj


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

class SAM3_OT_generate_boolean_surfaces(bpy.types.Operator):
    """Generate boolean collision surfaces for grass, sand, road2 (batch)."""

    bl_idname = "sam3.generate_boolean_surfaces"
    bl_label = "Generate Boolean Surfaces (Grass/Sand/Road2)"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context: bpy.types.Context) -> set[str]:
        mask_root = bpy.data.collections.get(ROOT_POLYGON_COLLECTION_NAME)
        if mask_root is None:
            self.report({"ERROR"},
                        f"Collection '{ROOT_POLYGON_COLLECTION_NAME}' not found")
            return {"CANCELLED"}

        depsgraph = context.evaluated_depsgraph_get()
        scene = context.scene
        excluded = _build_excluded_set()
        ray_origin_y = _get_terrain_max_y() + 100.0

        _log(f"Excluded {len(excluded)} objects, ray_origin_y = {ray_origin_y:.1f}")

        total_created = 0
        t_all = time.monotonic()

        for ti, tag in enumerate(BOOLEAN_TAGS):
            sub_col_name = f"mask_polygon_{tag}"
            sub_col = mask_root.children.get(sub_col_name)
            if sub_col is None:
                _log(f"[{ti + 1}/{len(BOOLEAN_TAGS)}] "
                     f"No collection '{sub_col_name}', skipping {tag}")
                continue

            mesh_objs = [o for o in sub_col.all_objects if o.type == "MESH"]
            if not mesh_objs:
                _log(f"[{ti + 1}/{len(BOOLEAN_TAGS)}] {tag}: no mesh objects")
                continue

            density = _get_density(tag)
            _log(f"[{ti + 1}/{len(BOOLEAN_TAGS)}] ===== {tag} ===== "
                 f"({len(mesh_objs)} mesh(es), density={density}m)")

            for mi, mask_obj in enumerate(mesh_objs):
                _log(f"[{ti + 1}/{len(BOOLEAN_TAGS)}] "
                     f"{tag} mesh {mi + 1}/{len(mesh_objs)}: {mask_obj.name}")
                result = generate_boolean_surface(
                    mask_obj, tag, density, scene, depsgraph,
                    ray_origin_y, excluded,
                )
                if result is not None:
                    total_created += 1
                    excluded.add(result.name)
                # Re-update depsgraph after each mask
                depsgraph.update()

        elapsed = time.monotonic() - t_all
        msg = (f"Created {total_created} boolean collision mesh(es) "
               f"in {_fmt_time(elapsed)}")
        _log(msg)
        self.report({"INFO"}, msg)
        return {"FINISHED"}


class SAM3_OT_generate_boolean_surface_selected(bpy.types.Operator):
    """Generate boolean collision surface from selected mask polygon(s)."""

    bl_idname = "sam3.generate_boolean_surface_selected"
    bl_label = "Generate Boolean Surface (Selected)"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context: bpy.types.Context) -> set[str]:
        from .mask_select_utils import get_mask_objects

        selected = get_mask_objects(context)
        if not selected:
            self.report({"WARNING"}, "No mask polygon objects selected")
            return {"CANCELLED"}

        depsgraph = context.evaluated_depsgraph_get()
        scene = context.scene
        excluded = _build_excluded_set()
        ray_origin_y = _get_terrain_max_y() + 100.0

        total_created = 0
        t_all = time.monotonic()

        for mask_obj in selected:
            if mask_obj.type != "MESH":
                continue

            tag = None
            for col in getattr(mask_obj, "users_collection", []) or []:
                name = getattr(col, "name", "")
                if name.startswith("mask_polygon_"):
                    tag = name[len("mask_polygon_"):]
                    break

            if tag is None:
                self.report({"WARNING"},
                            f"Cannot infer tag for {mask_obj.name}, skipping")
                continue

            if tag not in BOOLEAN_TAGS:
                _log(f"Tag '{tag}' not in {BOOLEAN_TAGS}, skipping {mask_obj.name}")
                continue

            density = _get_density(tag)
            result = generate_boolean_surface(
                mask_obj, tag, density, scene, depsgraph,
                ray_origin_y, excluded,
            )
            if result is not None:
                total_created += 1
                excluded.add(result.name)
            depsgraph.update()

        elapsed = time.monotonic() - t_all
        msg = (f"Created {total_created} boolean collision mesh(es) "
               f"from selection in {_fmt_time(elapsed)}")
        _log(msg)
        self.report({"INFO"}, msg)
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Action specs (auto-registered by blender_helpers)
# ---------------------------------------------------------------------------

ACTION_SPECS = [
    ActionSpec(
        operator_cls=SAM3_OT_generate_boolean_surfaces,
        menu_label="Generate Boolean Surfaces (Grass/Sand/Road2)",
        icon="MOD_BOOLEAN",
    ),
    ActionSpec(
        operator_cls=SAM3_OT_generate_boolean_surface_selected,
        menu_label="Generate Boolean Surface (Selected)",
        icon="MOD_BOOLEAN",
    ),
]
