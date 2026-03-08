"""
Blender action: Generate Boolean Surfaces.

Creates a regular Grid plane at the configured density, boolean-intersects it
with a solidified mask polygon to cut the shape, then raycasts every vertex
onto the terrain to project the flat grid into 3D.

Algorithm
---------
1. Read mask mesh XZ bounding box, expand slightly.
2. Create a Grid mesh with subdivisions = ceil(extent / density).
3. Extrude the mask mesh along ±Y to form a closed volume.
4. Boolean Intersect: grid ∩ extruded mask → cut grid to mask shape.
5. Raycast each remaining vertex onto terrain.
6. Remove vertices that miss terrain.
7. Place result in the appropriate ``collision_{tag}`` collection.

Two operators:
- ``sam3.generate_boolean_surfaces``          — batch: all configured tags
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
    SURFACE_SAMPLING_DENSITY_KERB,
    SURFACE_SAMPLING_DENSITY_ROAD,
    SURFACE_SAMPLING_DENSITY_ROAD2,
    SURFACE_SAMPLING_DENSITY_SAND,
)

BOOLEAN_TAGS = ["grass", "sand", "road2"]

_DENSITY_MAP: dict[str, float] = {
    "grass": SURFACE_SAMPLING_DENSITY_GRASS,
    "sand": SURFACE_SAMPLING_DENSITY_SAND,
    "road2": SURFACE_SAMPLING_DENSITY_ROAD2,
    "road": SURFACE_SAMPLING_DENSITY_ROAD,
    "kerb": SURFACE_SAMPLING_DENSITY_KERB,
}


def _get_density(tag: str) -> float:
    return _DENSITY_MAP.get(tag.strip().lower(), SURFACE_SAMPLING_DENSITY_DEFAULT)


# ---------------------------------------------------------------------------
# Progress reporting / debug (set by blender_automate before invoking)
# ---------------------------------------------------------------------------
PROGRESS_RANGE = None  # (start_pct, end_pct) set by blender_automate
DEBUG_SAVE_DIR: str | None = None  # set by blender_automate to enable debug saves

# Progress tracking context
_progress_ctx = {
    "current_tile": 0,
    "total_tiles": 0,
    "start_time": 0.0,
}


def _report_sub_progress(sub_frac, msg=""):
    """Report sub-progress within PROGRESS_RANGE."""
    if PROGRESS_RANGE is None:
        return
    start, end = PROGRESS_RANGE
    pct = int(start + sub_frac * (end - start))
    print(f"@@PROGRESS@@ {max(0,min(100,pct))} {msg}".rstrip(), flush=True)


def _report_tile_progress(msg=""):
    """Report progress based on completed tiles with ETA."""
    ctx = _progress_ctx
    if ctx["total_tiles"] == 0:
        return

    frac = ctx["current_tile"] / ctx["total_tiles"]
    elapsed = time.monotonic() - ctx["start_time"]

    if ctx["current_tile"] > 0 and elapsed > 0:
        avg_time = elapsed / ctx["current_tile"]
        remaining = (ctx["total_tiles"] - ctx["current_tile"]) * avg_time
        eta = _fmt_time(remaining)
        msg = f"{msg} (ETA: {eta})" if msg else f"ETA: {eta}"

    _report_sub_progress(frac, msg)


def _debug_save(stage_name: str) -> None:
    """Save a copy of the current .blend file for debugging."""
    if DEBUG_SAVE_DIR is None:
        return
    os.makedirs(DEBUG_SAVE_DIR, exist_ok=True)
    path = os.path.join(DEBUG_SAVE_DIR, f"{stage_name}.blend")
    bpy.ops.wm.save_as_mainfile(filepath=path, copy=True)
    _log(f"  DEBUG saved: {path}")


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

# 单个 grid 的最大顶点数阈值（避免内存爆炸）
MAX_GRID_VERTS = 1_000_000  # 100万顶点


def _compute_tile_plan(
    width_x: float, width_z: float, density: float, max_verts: int = MAX_GRID_VERTS
) -> tuple[int, int]:
    """计算智能分片方案。

    Args:
        width_x, width_z: XZ 方向的尺寸（米）
        density: 采样密度（米）
        max_verts: 单个 tile 的最大顶点数

    Returns:
        (tiles_x, tiles_z): X 和 Z 方向的分片数量
    """
    subdivs_x = max(1, math.ceil(width_x / density))
    subdivs_z = max(1, math.ceil(width_z / density))
    total_verts = (subdivs_x + 1) * (subdivs_z + 1)

    if total_verts <= max_verts:
        return (1, 1)  # 不需要分片

    # 计算需要的分片数量（保持长宽比）
    scale = math.sqrt(total_verts / max_verts)
    tiles_x = max(1, int(math.ceil(scale)))
    tiles_z = max(1, int(math.ceil(scale)))

    return (tiles_x, tiles_z)


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
    """Copy a mask mesh to world space and extrude it into a closed volume.

    Uses ``bmesh.ops.extrude_face_region`` instead of the Solidify modifier
    to guarantee manifold side walls at **both** outer and inner (hole)
    boundaries.  ``recalc_face_normals`` then ensures consistent outward
    normals so that Boolean INTERSECT can correctly determine inside/outside.

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

    # Log boundary info for debugging
    n_boundary = sum(1 for e in bm.edges if e.is_boundary)
    _log(f"  Mask mesh: {len(bm.verts):,} verts, {len(bm.faces):,} faces, "
         f"{n_boundary} boundary edges")

    # --- Manual extrusion to create a closed solid ---
    half = thickness / 2.0

    # Move the original faces down (+Y = gravity) — they become the bottom cap
    for v in bm.verts:
        v.co.y += half

    # Extrude face region: duplicates faces and auto-creates side-wall quads
    # at every boundary edge (outer boundary AND inner hole boundaries).
    ret = bmesh.ops.extrude_face_region(bm, geom=bm.faces[:])
    new_verts = [e for e in ret["geom"] if isinstance(e, bmesh.types.BMVert)]

    # Move extruded verts up (-Y = sky) by full thickness — they become the top cap
    for v in new_verts:
        v.co.y -= thickness

    # Ensure consistent outward normals for the closed volume.
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])

    n_verts = len(bm.verts)
    n_faces = len(bm.faces)

    temp_mesh = bpy.data.meshes.new("_bool_solid_tmp")
    bm.to_mesh(temp_mesh)
    bm.free()
    temp_mesh.update()

    temp_obj = bpy.data.objects.new("_bool_solid_tmp", temp_mesh)
    bpy.context.scene.collection.objects.link(temp_obj)

    _log(f"  Solidified: {n_verts:,} verts, {n_faces:,} faces "
         f"(thickness={thickness}m)")
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

    Tries FLOAT solver first, validates result, falls back to EXACT if needed.
    Returns a new object with the result, or None if both solvers fail.
    """
    grid_verts = len(grid_obj.data.vertices)
    float_result = None
    exact_result = None

    # Try FLOAT solver
    for m in list(grid_obj.modifiers):
        if m.type == "BOOLEAN":
            grid_obj.modifiers.remove(m)

    mod = grid_obj.modifiers.new("bool_intersect", "BOOLEAN")
    mod.operation = "INTERSECT"
    mod.object = solid_mask_obj
    mod.solver = "FLOAT"

    depsgraph.update()

    bm = bmesh.new()
    try:
        grid_eval = grid_obj.evaluated_get(depsgraph)
        bm.from_object(grid_eval, depsgraph)
    except Exception as exc:
        bm.free()
        _log(f"  Boolean FLOAT failed: {exc}, trying EXACT")
    else:
        n_verts = len(bm.verts)
        n_faces = len(bm.faces)
        if n_verts >= 3 and n_faces > 0:
            # FLOAT succeeded, save result
            result_mesh = bpy.data.meshes.new("_bool_float_tmp")
            bm.to_mesh(result_mesh)
            result_mesh.update()
            float_result = (n_verts, n_faces, result_mesh)
            _log(f"  Boolean FLOAT: {n_verts:,} verts, {n_faces:,} faces")
        bm.free()

    # Try EXACT solver
    for m in list(grid_obj.modifiers):
        if m.type == "BOOLEAN":
            grid_obj.modifiers.remove(m)

    mod = grid_obj.modifiers.new("bool_intersect", "BOOLEAN")
    mod.operation = "INTERSECT"
    mod.object = solid_mask_obj
    mod.solver = "EXACT"

    depsgraph.update()

    bm = bmesh.new()
    try:
        grid_eval = grid_obj.evaluated_get(depsgraph)
        bm.from_object(grid_eval, depsgraph)
    except Exception as exc:
        bm.free()
        _log(f"  Boolean EXACT failed: {exc}")
    else:
        n_verts = len(bm.verts)
        n_faces = len(bm.faces)
        if n_verts >= 3 and n_faces > 0:
            result_mesh = bpy.data.meshes.new("_bool_exact_tmp")
            bm.to_mesh(result_mesh)
            result_mesh.update()
            exact_result = (n_verts, n_faces, result_mesh)
            _log(f"  Boolean EXACT: {n_verts:,} verts, {n_faces:,} faces")
        bm.free()

    # Choose best result
    if float_result is None and exact_result is None:
        return None

    if float_result is None:
        _log(f"  Using EXACT (FLOAT failed)")
        result_obj = bpy.data.objects.new("_bool_result_tmp", exact_result[2])
        bpy.context.scene.collection.objects.link(result_obj)
        return result_obj

    if exact_result is None:
        _log(f"  Using FLOAT (EXACT failed)")
        result_obj = bpy.data.objects.new("_bool_result_tmp", float_result[2])
        bpy.context.scene.collection.objects.link(result_obj)
        return result_obj

    # Both succeeded, compare results
    float_verts, float_faces, float_mesh = float_result
    exact_verts, exact_faces, exact_mesh = exact_result

    # Compute quality metrics
    vert_ratio = float_verts / exact_verts if exact_verts > 0 else 1.0
    max_faces = max(float_faces, exact_faces)
    face_diff_pct = abs(float_faces - exact_faces) / max_faces if max_faces > 0 else 0.0

    # Heuristic: detect both under-tessellation and over-tessellation
    if vert_ratio < 0.7:
        _log(f"  Using EXACT (FLOAT verts {float_verts:,} < 70% of EXACT {exact_verts:,})")
        bpy.data.meshes.remove(float_mesh)
        result_obj = bpy.data.objects.new("_bool_result_tmp", exact_mesh)
        bpy.context.scene.collection.objects.link(result_obj)
        return result_obj

    if vert_ratio > 1.5:
        _log(f"  Using EXACT (FLOAT verts {float_verts:,} > 150% of EXACT {exact_verts:,})")
        bpy.data.meshes.remove(float_mesh)
        result_obj = bpy.data.objects.new("_bool_result_tmp", exact_mesh)
        bpy.context.scene.collection.objects.link(result_obj)
        return result_obj

    if face_diff_pct > 0.3:
        _log(f"  Using EXACT (face diff {face_diff_pct:.1%}: FLOAT={float_faces:,}, EXACT={exact_faces:,})")
        bpy.data.meshes.remove(float_mesh)
        result_obj = bpy.data.objects.new("_bool_result_tmp", exact_mesh)
        bpy.context.scene.collection.objects.link(result_obj)
        return result_obj

    # Otherwise prefer FLOAT (faster)
    _log(f"  Using FLOAT (validated: verts={float_verts:,}, faces={float_faces:,})")
    bpy.data.meshes.remove(exact_mesh)
    result_obj = bpy.data.objects.new("_bool_result_tmp", float_mesh)
    bpy.context.scene.collection.objects.link(result_obj)
    return result_obj


# ---------------------------------------------------------------------------
# Terrain projection
# ---------------------------------------------------------------------------

def _raycast_terrain(
    scene, depsgraph, excluded: set[str],
    x: float, z: float, ray_origin_y: float,
    direction: Vector,
) -> Vector | None:
    """Single-point raycast, skipping excluded objects.  Returns hit location."""
    origin = Vector((x, ray_origin_y, z))
    for _ in range(5):
        result, location, _nrm, _idx, hit_obj, _mtx = scene.ray_cast(
            depsgraph, origin, direction,
        )
        if not result:
            return None
        if hit_obj is None or hit_obj.name not in excluded:
            return location
        origin = location + direction * 0.001
    return None


# Small XZ offsets for fallback jitter probes — catches tile seam gaps
_JITTER_OFFSETS = [
    (0.05, 0.0), (-0.05, 0.0), (0.0, 0.05), (0.0, -0.05),
    (0.15, 0.0), (-0.15, 0.0), (0.0, 0.15), (0.0, -0.15),
]


def _project_to_terrain(
    obj: bpy.types.Object,
    scene: bpy.types.Scene,
    depsgraph,
    excluded: set[str],
    ray_origin_y: float,
) -> int:
    """Raycast each vertex of *obj* downward onto the terrain.

    Miss vertices are recovered via jittered raycasts and neighbour
    interpolation instead of being deleted outright.  Only truly
    unreachable vertices (no hit neighbours at all) are removed.
    Returns the number of direct + recovered hits.
    """
    me = obj.data
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()

    direction = Vector((0.0, -1.0, 0.0))
    hit_count = 0
    miss_indices: list[int] = []
    hit_set: set[int] = set()          # indices of verts that got a hit
    total_v = len(bm.verts)
    t0 = time.monotonic()
    last_pct = -1

    # --- Phase 1: standard raycast ---
    for vi, v in enumerate(bm.verts):
        pct = (vi * 100) // total_v if total_v else 100
        if pct >= last_pct + 10:
            last_pct = pct
            elapsed = time.monotonic() - t0
            _log(f"  Projection: {pct}% ({vi:,}/{total_v:,}) "
                 f"elapsed {_fmt_time(elapsed)}")

        loc = _raycast_terrain(
            scene, depsgraph, excluded, v.co.x, v.co.z,
            ray_origin_y, direction,
        )
        if loc is not None:
            v.co = loc
            hit_count += 1
            hit_set.add(vi)
        else:
            miss_indices.append(vi)

    # --- Phase 2: jittered fallback for misses (tile seam recovery) ---
    still_miss: list[int] = []
    jitter_recovered = 0
    if miss_indices:
        _log(f"  Phase 2: jitter probe for {len(miss_indices):,} miss verts")
        for vi in miss_indices:
            v = bm.verts[vi]
            recovered = False
            for dx, dz in _JITTER_OFFSETS:
                loc = _raycast_terrain(
                    scene, depsgraph, excluded,
                    v.co.x + dx, v.co.z + dz,
                    ray_origin_y, direction,
                )
                if loc is not None:
                    # Use the hit Y but keep original XZ to avoid shifting
                    v.co.y = loc.y
                    hit_count += 1
                    hit_set.add(vi)
                    jitter_recovered += 1
                    recovered = True
                    break
            if not recovered:
                still_miss.append(vi)
        if jitter_recovered:
            _log(f"  Jitter recovered {jitter_recovered:,} verts")

    # --- Phase 3: interpolate from connected hit neighbours ---
    interp_recovered = 0
    truly_orphan: list = []
    if still_miss:
        _log(f"  Phase 3: neighbour interpolation for "
             f"{len(still_miss):,} remaining misses")
        remaining = set(still_miss)
        for _pass in range(3):
            newly_resolved: list[int] = []
            for vi in remaining:
                v = bm.verts[vi]
                y_sum = 0.0
                y_cnt = 0
                for edge in v.link_edges:
                    other = edge.other_vert(v)
                    if other.index in hit_set:
                        y_sum += other.co.y
                        y_cnt += 1
                if y_cnt > 0:
                    v.co.y = y_sum / y_cnt
                    hit_set.add(vi)
                    hit_count += 1
                    interp_recovered += 1
                    newly_resolved.append(vi)
            for vi in newly_resolved:
                remaining.discard(vi)
            if not newly_resolved:
                break
        if interp_recovered:
            _log(f"  Interpolation recovered {interp_recovered:,} verts")
        truly_orphan = [bm.verts[vi] for vi in remaining]

    # --- Phase 4: delete only truly orphaned verts ---
    if truly_orphan:
        _log(f"  Removing {len(truly_orphan):,} truly orphaned verts "
             f"(no reachable neighbours)")
        bmesh.ops.delete(bm, geom=truly_orphan, context="VERTS")

    bm.to_mesh(me)
    bm.free()
    me.update()

    elapsed = time.monotonic() - t0
    _log(f"  Projection done: {hit_count:,}/{total_v:,} hits "
         f"(jitter={jitter_recovered}, interp={interp_recovered}, "
         f"orphan={len(truly_orphan)}) ({_fmt_time(elapsed)})")
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
) -> list[bpy.types.Object]:
    """Generate collision surface(s) for *tag* by boolean-cutting a grid.

    Returns a list of collision mesh objects (may be multiple if tiled).
    """
    t0 = time.monotonic()
    label = f"{tag}/{mask_obj.name}"
    _log(f"--- {label}: boolean grid (density={density}m) ---")

    # 1. Mask XZ bounds
    bounds = _mask_xz_bounds(mask_obj, depsgraph, margin=max(density * 3, 5.0))
    if bounds is None:
        _log(f"  {label}: cannot read mask bounds, skipping")
        return []
    min_x, max_x, min_z, max_z, y_value = bounds
    width_x = max_x - min_x
    width_z = max_z - min_z

    # 2. 计算分片方案
    tiles_x, tiles_z = _compute_tile_plan(width_x, width_z, density)
    if tiles_x > 1 or tiles_z > 1:
        _log(f"  Tiling: {tiles_x}x{tiles_z} tiles ({width_x:.1f}x{width_z:.1f}m)")

    # 3. 分片处理（确保 tile 边界对齐到 density grid）
    results = []

    # 计算每个 tile 的 grid 单元数（确保整数）
    cells_x = math.ceil(width_x / density)
    cells_z = math.ceil(width_z / density)
    cells_per_tile_x = math.ceil(cells_x / tiles_x)
    cells_per_tile_z = math.ceil(cells_z / tiles_z)

    for tz in range(tiles_z):
        for tx in range(tiles_x):
            # 基于 grid 单元计算 tile 边界（确保对齐）
            start_cell_x = tx * cells_per_tile_x
            end_cell_x = min((tx + 1) * cells_per_tile_x, cells_x)
            start_cell_z = tz * cells_per_tile_z
            end_cell_z = min((tz + 1) * cells_per_tile_z, cells_z)

            tile_min_x = min_x + start_cell_x * density
            tile_max_x = min_x + end_cell_x * density
            tile_min_z = min_z + start_cell_z * density
            tile_max_z = min_z + end_cell_z * density

            result = _process_tile(
                mask_obj, tag, density, scene, depsgraph,
                tile_min_x, tile_max_x, tile_min_z, tile_max_z,
                y_value, ray_origin_y, excluded, tx, tz
            )
            if result is not None:
                results.append(result)

    elapsed = time.monotonic() - t0
    _log(f"  => {label}: {len(results)} tile(s) ({_fmt_time(elapsed)} total)")
    return results


def _process_tile(
    mask_obj: bpy.types.Object,
    tag: str,
    density: float,
    scene: bpy.types.Scene,
    depsgraph,
    min_x: float, max_x: float,
    min_z: float, max_z: float,
    y_value: float,
    ray_origin_y: float,
    excluded: set[str],
    tile_x: int, tile_z: int,
) -> bpy.types.Object | None:
    """Process a single tile of the mask."""
    # Create grid
    grid_obj = _create_grid_plane(min_x, max_x, min_z, max_z, density, y_value)
    _debug_save(f"{tag}_tile{tile_x}_{tile_z}_00_grid")

    # Solidify mask
    solid_obj = _solidify_mask(mask_obj, depsgraph, thickness=200.0)
    if solid_obj is None:
        _safe_remove_object(grid_obj)
        return None

    depsgraph.update()
    _debug_save(f"{tag}_tile{tile_x}_{tile_z}_01_solidified")

    # Boolean intersect
    result_obj = _boolean_intersect(grid_obj, solid_obj, depsgraph)
    _debug_save(f"{tag}_tile{tile_x}_{tile_z}_02_bool")

    _safe_remove_object(grid_obj)
    _safe_remove_object(solid_obj)
    depsgraph.update()

    if result_obj is None:
        return None

    # Project to terrain
    excluded_local = excluded | {result_obj.name}
    hit_count = _project_to_terrain(
        result_obj, scene, depsgraph, excluded_local, ray_origin_y,
    )

    # Clean up
    me = result_obj.data
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.dissolve_degenerate(bm, dist=0.001, edges=bm.edges)
    loose = [v for v in bm.verts if not v.link_faces]
    if loose:
        bmesh.ops.delete(bm, geom=loose, context="VERTS")
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bm.to_mesh(me)
    bm.free()
    me.update()

    n_verts = len(result_obj.data.vertices)
    n_faces = len(result_obj.data.polygons)
    if n_verts < 3 or n_faces == 0:
        _safe_remove_object(result_obj)
        return None

    result_obj.name = f"_tile_{tag}_{tile_x}_{tile_z}"
    result_obj.data.name = f"{result_obj.name}_mesh"

    # Report tile completion
    _progress_ctx["current_tile"] += 1
    _report_tile_progress(f"Tile {tile_x},{tile_z}")

    return result_obj


# ---------------------------------------------------------------------------
# Tile merging
# ---------------------------------------------------------------------------

def _merge_tiles(
    tile_objs: list[bpy.types.Object],
    tag: str,
    base_idx: int,
) -> bpy.types.Object | None:
    """合并多个 tile 碰撞 mesh 成单一对象，并焊接顶点。"""
    if not tile_objs:
        return None

    if len(tile_objs) == 1:
        # 单个 tile，直接重命名
        obj = tile_objs[0]
        col_name = COLLISION_COLLECTION_MAP.get(tag, f"collision_{tag}")
        col = _get_or_create_collection(col_name)
        try:
            obj_name = generate_collision_name(tag, base_idx)
        except ValueError:
            obj_name = f"1{tag.upper()}_{base_idx}"
        obj.name = obj_name
        obj.data.name = f"{obj_name}_mesh"
        _link_to_collection(obj, col)
        n_verts = len(obj.data.vertices)
        n_faces = len(obj.data.polygons)
        _log(f"  => {obj_name}: {n_verts:,} verts, {n_faces:,} faces (single tile)")
        return obj

    # 合并多个 tiles
    _log(f"  Merging {len(tile_objs)} tiles...")
    bm_merged = bmesh.new()

    for tile_obj in tile_objs:
        temp_bm = bmesh.new()
        temp_bm.from_mesh(tile_obj.data)
        temp_bm.transform(tile_obj.matrix_world)

        vert_map = {}
        for v in temp_bm.verts:
            vert_map[v.index] = bm_merged.verts.new(v.co)
        bm_merged.verts.ensure_lookup_table()

        for f in temp_bm.faces:
            try:
                bm_merged.faces.new([vert_map[v.index] for v in f.verts])
            except Exception:
                pass
        temp_bm.free()

    # 焊接顶点（tile 边界处的重复顶点）
    verts_before = len(bm_merged.verts)
    bmesh.ops.remove_doubles(bm_merged, verts=bm_merged.verts[:], dist=0.01)
    verts_after = len(bm_merged.verts)
    _log(f"  Welded {verts_before - verts_after} duplicate verts")

    # 创建合并后的 mesh
    merged_mesh = bpy.data.meshes.new(f"_merged_{tag}_mesh")
    bm_merged.to_mesh(merged_mesh)
    bm_merged.free()
    merged_mesh.update()

    merged_obj = bpy.data.objects.new(f"_merged_{tag}", merged_mesh)
    bpy.context.scene.collection.objects.link(merged_obj)

    # 删除原始 tiles
    for tile_obj in tile_objs:
        _safe_remove_object(tile_obj)

    # 重命名并放入正确的 collection
    col_name = COLLISION_COLLECTION_MAP.get(tag, f"collision_{tag}")
    col = _get_or_create_collection(col_name)
    try:
        obj_name = generate_collision_name(tag, base_idx)
    except ValueError:
        obj_name = f"1{tag.upper()}_{base_idx}"

    merged_obj.name = obj_name
    merged_obj.data.name = f"{obj_name}_mesh"
    _link_to_collection(merged_obj, col)

    n_verts = len(merged_obj.data.vertices)
    n_faces = len(merged_obj.data.polygons)
    _log(f"  => {obj_name}: {n_verts:,} verts, {n_faces:,} faces (merged from {len(tile_objs)} tiles)")
    return merged_obj


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

class SAM3_OT_generate_boolean_surfaces(bpy.types.Operator):
    """Generate boolean collision surfaces for configured tags (batch)."""

    bl_idname = "sam3.generate_boolean_surfaces"
    bl_label = "Generate Boolean Surfaces"
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
        _log(f"BOOLEAN_TAGS = {BOOLEAN_TAGS}")

        # 预先计算总 tile 数
        total_tiles = 0
        for tag in BOOLEAN_TAGS:
            sub_col_name = f"mask_polygon_{tag}"
            sub_col = mask_root.children.get(sub_col_name)
            if sub_col is None:
                continue
            mesh_objs = [o for o in sub_col.all_objects if o.type == "MESH"]
            density = _get_density(tag)
            for mask_obj in mesh_objs:
                bounds = _mask_xz_bounds(mask_obj, depsgraph, margin=density)
                if bounds is not None:
                    min_x, max_x, min_z, max_z, _ = bounds
                    tiles_x, tiles_z = _compute_tile_plan(max_x - min_x, max_z - min_z, density)
                    total_tiles += tiles_x * tiles_z

        _log(f"Total tiles to process: {total_tiles}")
        _progress_ctx["current_tile"] = 0
        _progress_ctx["total_tiles"] = total_tiles
        _progress_ctx["start_time"] = time.monotonic()

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
                tile_results = generate_boolean_surface(
                    mask_obj, tag, density, scene, depsgraph,
                    ray_origin_y, excluded,
                )
                if tile_results:
                    for result in tile_results:
                        excluded.add(result.name)
                    # 合并 tiles 成单一碰撞 mesh
                    merged = _merge_tiles(tile_results, tag, total_created)
                    if merged is not None:
                        total_created += 1
                        excluded.add(merged.name)
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
            tile_results = generate_boolean_surface(
                mask_obj, tag, density, scene, depsgraph,
                ray_origin_y, excluded,
            )
            if tile_results:
                for result in tile_results:
                    excluded.add(result.name)
                merged = _merge_tiles(tile_results, tag, total_created)
                if merged is not None:
                    total_created += 1
                    excluded.add(merged.name)
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
        menu_label="Generate Boolean Surfaces",
        icon="MOD_BOOLEAN",
    ),
    ActionSpec(
        operator_cls=SAM3_OT_generate_boolean_surface_selected,
        menu_label="Generate Boolean Surface (Selected)",
        icon="MOD_BOOLEAN",
    ),
]
