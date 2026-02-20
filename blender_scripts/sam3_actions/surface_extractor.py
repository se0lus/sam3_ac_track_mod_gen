"""
Blender action: Extract Collision Surfaces from mask polygon meshes.

**Approach**: Voxel Remesh + terrain projection.

1. Copy the mask polygon mesh to world space.
2. Apply Blender's built-in Voxel Remesh (voxel_size = density) to generate
   uniform interior vertices while preserving topology (holes, rings, etc.).
3. Raycast every vertex downward onto the terrain.
4. Remove vertices that miss terrain; create the collision mesh object.

Two operators:
- ``sam3.extract_surfaces``          -- batch: all surface tags
- ``sam3.extract_surface_selected``  -- manual: selected mask polygon(s)
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

SURFACE_TAGS = ["road", "kerb", "road2", "grass", "sand"]

_DENSITY_MAP = {
    "road": SURFACE_SAMPLING_DENSITY_ROAD,
    "grass": SURFACE_SAMPLING_DENSITY_GRASS,
    "kerb": SURFACE_SAMPLING_DENSITY_KERB,
    "sand": SURFACE_SAMPLING_DENSITY_SAND,
    "road2": SURFACE_SAMPLING_DENSITY_ROAD2,
}


def _get_density(tag: str) -> float:
    return _DENSITY_MAP.get(tag.strip().lower(), SURFACE_SAMPLING_DENSITY_DEFAULT)


def _log(msg: str) -> None:
    sys.stderr.write(f"[surface_extractor] {msg}\n")
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
# Core: Voxel Remesh + terrain projection
# ---------------------------------------------------------------------------

def extract_surface_for_mask(
    mask_obj: bpy.types.Object,
    tag: str,
    density: float,
    scene: bpy.types.Scene,
    depsgraph,
    ray_origin_y: float,
    excluded: set[str],
    edge_simplify: float = 0.0,
) -> Optional[bpy.types.Object]:
    """Generate a collision surface by remeshing a mask mesh and projecting
    onto terrain.

    1. Copy mask mesh to world space.
    2. Voxel Remesh at *density* to add uniform interior vertices.
    3. Raycast every vertex downward (-Y) onto the terrain.
    4. Remove misses, create collision mesh.
    """
    t0 = time.monotonic()
    label = f"{tag}/{mask_obj.name}"
    _log(f"--- {label}: voxel remesh + projection (density={density}m) ---")

    # ---- 1. Copy mask mesh to world space ----
    bm_src = bmesh.new()
    try:
        obj_eval = mask_obj.evaluated_get(depsgraph)
        bm_src.from_object(obj_eval, depsgraph)
    except Exception:
        try:
            bm_src.from_mesh(mask_obj.data)
        except Exception:
            bm_src.free()
            _log(f"  {label}: cannot read mesh, skipping")
            return None

    if len(bm_src.verts) < 3 or len(bm_src.faces) == 0:
        bm_src.free()
        _log(f"  {label}: mesh too simple, skipping")
        return None

    # Transform to world space so remesh operates in world units
    mw = mask_obj.matrix_world
    bmesh.ops.transform(bm_src, matrix=mw, verts=bm_src.verts)

    _log(f"  {label}: source: {len(bm_src.verts):,} verts, "
         f"{len(bm_src.faces):,} faces")

    temp_mesh = bpy.data.meshes.new(f"_temp_{tag}")
    bm_src.to_mesh(temp_mesh)
    bm_src.free()

    # ---- 2. Voxel Remesh via modifier ----
    temp_obj = bpy.data.objects.new(f"_temp_remesh_{tag}", temp_mesh)
    bpy.context.scene.collection.objects.link(temp_obj)

    mod = temp_obj.modifiers.new("remesh", "REMESH")
    mod.mode = "VOXEL"
    mod.voxel_size = density

    # Evaluate modifier to get remeshed geometry
    depsgraph.update()
    temp_eval = temp_obj.evaluated_get(depsgraph)

    bm = bmesh.new()
    bm.from_object(temp_eval, depsgraph)

    # Clean up temp object immediately
    bpy.data.objects.remove(temp_obj, do_unlink=True)
    bpy.data.meshes.remove(temp_mesh)

    # Re-update depsgraph after removing temp object
    depsgraph.update()

    if len(bm.verts) < 3 or len(bm.faces) == 0:
        bm.free()
        _log(f"  {label}: remesh produced empty result, skipping")
        return None

    _log(f"  {label}: remeshed: {len(bm.verts):,} verts, "
         f"{len(bm.faces):,} faces")

    # ---- 3. Raycast each vertex to terrain ----
    bm.verts.ensure_lookup_table()
    direction = Vector((0.0, -1.0, 0.0))
    hit_count = 0
    miss_verts: list = []
    total_v = len(bm.verts)
    t_ray = time.monotonic()
    last_pct = -1

    for vi, v in enumerate(bm.verts):
        pct = (vi * 100) // total_v if total_v else 100
        if pct >= last_pct + 5:
            last_pct = pct
            elapsed = time.monotonic() - t_ray
            if pct > 0:
                eta = elapsed / pct * (100 - pct)
                _log(f"  {label} raycast: {pct}% ({vi:,}/{total_v:,}) "
                     f"elapsed {_fmt_time(elapsed)}, ETA {_fmt_time(eta)}")
            else:
                _log(f"  {label} raycast: 0% ({total_v:,} verts)")

        origin = Vector((v.co.x, ray_origin_y, v.co.z))
        hit = False
        for _ in range(5):
            result, location, _nrm, _idx, obj, _mtx = scene.ray_cast(
                depsgraph, origin, direction,
            )
            if not result:
                break
            if obj is None or obj.name not in excluded:
                v.co = location
                hit_count += 1
                hit = True
                break
            origin = location + direction * 0.001
        if not hit:
            miss_verts.append(v)

    _log(f"  {label} raycast done: {hit_count:,}/{total_v:,} hits "
         f"in {_fmt_time(time.monotonic() - t_ray)}")

    # ---- 4. Remove missed vertices ----
    if miss_verts:
        _log(f"  {label}: removing {len(miss_verts):,} verts with no terrain")
        bmesh.ops.delete(bm, geom=miss_verts, context="VERTS")

    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    if len(bm.verts) < 3 or len(bm.faces) == 0:
        bm.free()
        _log(f"  {label}: too few geometry after raycast")
        return None

    # ---- 5. Create collision mesh object ----
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

    n_verts = len(bm.verts)
    n_faces = len(bm.faces)
    _log(f"  {label}: creating {obj_name} ({n_verts:,} verts, {n_faces:,} faces)")

    mesh_data = bpy.data.meshes.new(f"{obj_name}_mesh")
    bm.to_mesh(mesh_data)
    bm.free()
    mesh_data.update()

    obj = bpy.data.objects.new(obj_name, mesh_data)
    _link_to_collection(obj, col)

    elapsed = time.monotonic() - t0
    _log(f"  => {obj_name}: {n_verts:,} verts, {n_faces:,} faces "
         f"-> {col_name} ({_fmt_time(elapsed)} total)")
    return obj


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

class SAM3_OT_extract_surfaces(bpy.types.Operator):
    """Extract collision surfaces from all mask polygon meshes (batch)."""

    bl_idname = "sam3.extract_surfaces"
    bl_label = "Extract All Collision Surfaces"
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
        _log(f"Excluded {len(excluded)} objects from raycast")

        ray_origin_y = _get_terrain_max_y() + 100.0
        _log(f"Ray origin Y = {ray_origin_y:.1f}")

        total_created = 0
        t_all = time.monotonic()

        for ti, tag in enumerate(SURFACE_TAGS):
            sub_col_name = f"mask_polygon_{tag}"
            sub_col = mask_root.children.get(sub_col_name)
            if sub_col is None:
                _log(f"[{ti+1}/{len(SURFACE_TAGS)}] No collection "
                     f"'{sub_col_name}', skipping {tag}")
                continue

            mesh_objs = [o for o in sub_col.all_objects if o.type == "MESH"]
            if not mesh_objs:
                _log(f"[{ti+1}/{len(SURFACE_TAGS)}] {tag}: no mesh objects")
                continue

            density = _get_density(tag)
            _log(f"[{ti+1}/{len(SURFACE_TAGS)}] ===== {tag} ===== "
                 f"({len(mesh_objs)} mesh(es), density={density}m)")

            for mi, mask_obj in enumerate(mesh_objs):
                _log(f"[{ti+1}/{len(SURFACE_TAGS)}] "
                     f"{tag} mesh {mi+1}/{len(mesh_objs)}: {mask_obj.name}")
                result = extract_surface_for_mask(
                    mask_obj, tag, density, scene, depsgraph,
                    ray_origin_y, excluded,
                )
                if result is not None:
                    total_created += 1
                    excluded.add(result.name)

        elapsed = time.monotonic() - t_all
        msg = (f"Created {total_created} collision mesh(es) "
               f"in {_fmt_time(elapsed)}")
        _log(msg)
        self.report({"INFO"}, msg)
        return {"FINISHED"}


class SAM3_OT_extract_surface_selected(bpy.types.Operator):
    """Extract collision surface from selected mask polygon mesh(es)."""

    bl_idname = "sam3.extract_surface_selected"
    bl_label = "Extract Collision Surface (Selected)"
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

            tag = "unknown"
            for col in getattr(mask_obj, "users_collection", []) or []:
                name = getattr(col, "name", "")
                if name.startswith("mask_polygon_"):
                    tag = name[len("mask_polygon_"):]
                    break

            if tag == "unknown":
                self.report({"WARNING"},
                            f"Cannot infer tag for {mask_obj.name}, skipping")
                continue

            density = _get_density(tag)
            result = extract_surface_for_mask(
                mask_obj, tag, density, scene, depsgraph,
                ray_origin_y, excluded,
            )
            if result is not None:
                total_created += 1
                excluded.add(result.name)

        elapsed = time.monotonic() - t_all
        msg = (f"Created {total_created} collision mesh(es) from selection "
               f"in {_fmt_time(elapsed)}")
        _log(msg)
        self.report({"INFO"}, msg)
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Action specs
# ---------------------------------------------------------------------------

ACTION_SPECS = [
    ActionSpec(
        operator_cls=SAM3_OT_extract_surfaces,
        menu_label="Extract All Collision Surfaces",
        icon="MESH_DATA",
    ),
    ActionSpec(
        operator_cls=SAM3_OT_extract_surface_selected,
        menu_label="Extract Collision Surface (Selected)",
        icon="MESH_DATA",
    ),
]
