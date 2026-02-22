"""
Blender action: Extract Terrain Surfaces (Road + Kerb).

Directly copies triangular faces from 3D terrain tiles, classified by mask
polygon containment.  No remeshing — preserves original terrain geometry.

Algorithm
---------
1. Build BVHTree from kerb / road mask polygon meshes (XZ containment via
   downward ray-cast on the flat mask mesh).
2. Iterate all L{level} terrain tile objects and their triangulated faces.
3. Each face's XZ centroid → BVH ray-cast:
   - Inside kerb mask  → kerb face
   - Inside road mask (but not kerb) → road face
   - Neither → skip
4. Build per-tag collision meshes; ``remove_doubles`` welds shared vertices.

Two operators:
- ``sam3.extract_terrain_surfaces``          — batch: auto-detect road + kerb
- ``sam3.extract_terrain_surface_selected``  — manual: selected mask(s)
"""

from __future__ import annotations

import math
import os
import re
import sys
import time
from typing import Optional

import bpy  # type: ignore[import-not-found]
import bmesh  # type: ignore[import-not-found]
from mathutils import Vector  # type: ignore[import-not-found]
from mathutils.bvhtree import BVHTree  # type: ignore[import-not-found]

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
from config import ROOT_POLYGON_COLLECTION_NAME  # noqa: E402

# ---------------------------------------------------------------------------
# Logging / formatting helpers
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    sys.stderr.write(f"[terrain_mesh_extractor] {msg}\n")
    sys.stderr.flush()


def _fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"


# ---------------------------------------------------------------------------
# Collection helpers (same pattern as surface_extractor)
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

_TERRAIN_COL_RE = re.compile(r"^L\d+$")


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


def _get_terrain_objects(excluded: set[str]) -> list[bpy.types.Object]:
    """Collect MESH objects from L{digits} collections (terrain tiles)."""
    objs: list[bpy.types.Object] = []
    seen: set[str] = set()
    for col in bpy.data.collections:
        if not _TERRAIN_COL_RE.match(col.name):
            continue
        for obj in col.all_objects:
            if obj.type == "MESH" and obj.name not in excluded and obj.name not in seen:
                objs.append(obj)
                seen.add(obj.name)
    return objs


def _get_terrain_max_y(terrain_objs: list[bpy.types.Object]) -> float:
    """Find the maximum Y (world space) across terrain objects."""
    max_y = 0.0
    for obj in terrain_objs:
        try:
            for corner in obj.bound_box:
                w = obj.matrix_world @ Vector(corner)
                if w.y > max_y:
                    max_y = w.y
        except Exception:
            continue
    return max_y


# ---------------------------------------------------------------------------
# BVH construction from mask meshes
# ---------------------------------------------------------------------------

def _build_bvh_from_masks(
    mask_objs: list[bpy.types.Object],
    depsgraph,
) -> BVHTree | None:
    """Merge multiple mask meshes (world space) into a single BVHTree.

    Mask meshes are flat in the XZ plane.  A downward ray-cast hitting this
    BVH indicates the query point's XZ projection lies inside the mask.
    """
    if not mask_objs:
        return None

    merged = bmesh.new()
    total_faces = 0

    for mask_obj in mask_objs:
        try:
            obj_eval = mask_obj.evaluated_get(depsgraph)
            temp_bm = bmesh.new()
            try:
                temp_bm.from_object(obj_eval, depsgraph)
            except Exception:
                temp_bm.from_mesh(mask_obj.data)

            mw = mask_obj.matrix_world
            bmesh.ops.transform(temp_bm, matrix=mw, verts=temp_bm.verts)

            # Append into merged bmesh via temporary mesh
            temp_mesh = bpy.data.meshes.new("_bvh_merge_tmp")
            temp_bm.to_mesh(temp_mesh)
            temp_bm.free()

            merged.from_mesh(temp_mesh)
            bpy.data.meshes.remove(temp_mesh)
        except Exception as exc:
            _log(f"  Warning: failed to read mask '{mask_obj.name}': {exc}")
            continue

    total_faces = len(merged.faces)
    if total_faces == 0:
        merged.free()
        return None

    # Ensure faces are triangulated for BVH
    bmesh.ops.triangulate(merged, faces=merged.faces)

    bvh = BVHTree.FromBMesh(merged)
    merged.free()
    _log(f"  BVH built: {total_faces} faces from {len(mask_objs)} mask(s)")
    return bvh


def _point_inside_bvh(bvh: BVHTree, x: float, z: float, ray_y: float) -> bool:
    """Test if (x, z) falls inside the mask by casting a ray from above."""
    origin = Vector((x, ray_y, z))
    direction = Vector((0.0, -1.0, 0.0))
    location, _normal, _index, _dist = bvh.ray_cast(origin, direction)
    return location is not None


# ---------------------------------------------------------------------------
# Collision mesh builder
# ---------------------------------------------------------------------------

def _build_collision_mesh(
    faces_data: list[tuple],
    tag: str,
    col_name: str,
    weld_dist: float = 0.001,
) -> bpy.types.Object | None:
    """Create a collision mesh from a list of world-space triangles.

    Parameters
    ----------
    faces_data : list of ((x,y,z), (x,y,z), (x,y,z))
        Each entry is a triangle's three vertices in world space.
    tag : str
        Surface tag for AC naming (e.g. "road", "kerb").
    col_name : str
        Target collection name.
    weld_dist : float
        Vertex merge distance for remove_doubles.
    """
    if not faces_data:
        return None

    t0 = time.monotonic()

    # Build flat vertex / face arrays
    verts: list[tuple[float, float, float]] = []
    face_indices: list[tuple[int, int, int]] = []
    for v0, v1, v2 in faces_data:
        base = len(verts)
        verts.append(v0)
        verts.append(v1)
        verts.append(v2)
        face_indices.append((base, base + 1, base + 2))

    # Determine collision object name
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

    _log(f"  Building {obj_name}: {len(faces_data):,} tris, "
         f"{len(verts):,} verts (pre-weld) ...")

    mesh_data = bpy.data.meshes.new(f"{obj_name}_mesh")
    mesh_data.from_pydata(verts, [], face_indices)
    mesh_data.update()

    # Weld shared vertices
    bm = bmesh.new()
    bm.from_mesh(mesh_data)
    n_before = len(bm.verts)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=weld_dist)
    n_after = len(bm.verts)
    bm.to_mesh(mesh_data)
    bm.free()
    mesh_data.update()

    obj = bpy.data.objects.new(obj_name, mesh_data)
    _link_to_collection(obj, col)

    elapsed = time.monotonic() - t0
    _log(f"  => {obj_name}: {n_before:,} -> {n_after:,} verts "
         f"(welded), {len(face_indices):,} faces -> {col_name} "
         f"({_fmt_time(elapsed)})")
    return obj


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_terrain_for_road_kerb(
    road_masks: list[bpy.types.Object],
    kerb_masks: list[bpy.types.Object],
    scene: bpy.types.Scene,
    depsgraph,
) -> tuple[bpy.types.Object | None, bpy.types.Object | None]:
    """Extract terrain faces classified as road or kerb.

    Returns (road_obj, kerb_obj).  Kerb takes priority over road where masks
    overlap, ensuring no duplication.
    """
    t_start = time.monotonic()

    excluded = _build_excluded_set()
    terrain_objs = _get_terrain_objects(excluded)
    if not terrain_objs:
        _log("ERROR: No terrain objects found in L{n} collections")
        return None, None
    _log(f"Found {len(terrain_objs)} terrain objects")

    # Build BVH for each tag
    _log("Building BVH for kerb masks ...")
    kerb_bvh = _build_bvh_from_masks(kerb_masks, depsgraph) if kerb_masks else None
    _log("Building BVH for road masks ...")
    road_bvh = _build_bvh_from_masks(road_masks, depsgraph) if road_masks else None

    if kerb_bvh is None and road_bvh is None:
        _log("No valid mask BVH could be built — nothing to extract")
        return None, None

    # Ray origin Y: above all terrain
    ray_y = _get_terrain_max_y(terrain_objs) + 100.0
    _log(f"Ray origin Y = {ray_y:.1f}")

    road_faces: list[tuple] = []
    kerb_faces: list[tuple] = []

    total_terrain_tris = 0
    t_iter = time.monotonic()

    for oi, obj in enumerate(terrain_objs):
        try:
            obj_eval = obj.evaluated_get(depsgraph)
            me = obj_eval.to_mesh()
        except Exception as exc:
            _log(f"  Warning: cannot read mesh '{obj.name}': {exc}")
            continue

        try:
            me.calc_loop_triangles()
            mw = obj.matrix_world
            verts = me.vertices
            n_obj_tris = len(me.loop_triangles)
            total_terrain_tris += n_obj_tris

            for lt in me.loop_triangles:
                vi = lt.vertices
                v0 = mw @ verts[vi[0]].co
                v1 = mw @ verts[vi[1]].co
                v2 = mw @ verts[vi[2]].co

                tri = (
                    (float(v0.x), float(v0.y), float(v0.z)),
                    (float(v1.x), float(v1.y), float(v1.z)),
                    (float(v2.x), float(v2.y), float(v2.z)),
                )

                # Test 3 vertices + centroid — select if ANY hits the mask.
                # Vertices catch edge-straddling tris; centroid catches the
                # rare case where all 3 vertices are outside but the center
                # pokes through a narrow mask region.
                cx = (v0.x + v1.x + v2.x) / 3.0
                cz = (v0.z + v1.z + v2.z) / 3.0
                pts = ((v0.x, v0.z), (v1.x, v1.z), (v2.x, v2.z), (cx, cz))

                # Kerb takes priority
                if kerb_bvh is not None and any(
                    _point_inside_bvh(kerb_bvh, px, pz, ray_y) for px, pz in pts
                ):
                    kerb_faces.append(tri)
                elif road_bvh is not None and any(
                    _point_inside_bvh(road_bvh, px, pz, ray_y) for px, pz in pts
                ):
                    road_faces.append(tri)
        finally:
            obj_eval.to_mesh_clear()

        # Progress every 10 objects or at the end
        if (oi + 1) % 10 == 0 or oi == len(terrain_objs) - 1:
            elapsed = time.monotonic() - t_iter
            _log(f"  Terrain scan: {oi + 1}/{len(terrain_objs)} objects, "
                 f"{total_terrain_tris:,} tris scanned, "
                 f"road={len(road_faces):,} kerb={len(kerb_faces):,} "
                 f"({_fmt_time(elapsed)})")

    _log(f"Face classification done: "
         f"road={len(road_faces):,}, kerb={len(kerb_faces):,} "
         f"from {total_terrain_tris:,} terrain tris")

    # Build collision meshes
    road_obj = None
    kerb_obj = None

    if road_faces:
        road_col = COLLISION_COLLECTION_MAP.get("road", "collision_road")
        road_obj = _build_collision_mesh(road_faces, "road", road_col)

    if kerb_faces:
        kerb_col = COLLISION_COLLECTION_MAP.get("kerb", "collision_kerb")
        kerb_obj = _build_collision_mesh(kerb_faces, "kerb", kerb_col)

    total_elapsed = time.monotonic() - t_start
    _log(f"=== Terrain extraction complete ({_fmt_time(total_elapsed)}) ===")
    return road_obj, kerb_obj


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

class SAM3_OT_extract_terrain_surfaces(bpy.types.Operator):
    """Extract terrain surfaces for road + kerb from mask polygons (batch)."""

    bl_idname = "sam3.extract_terrain_surfaces"
    bl_label = "Extract Terrain Surfaces (Road+Kerb)"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context: bpy.types.Context) -> set[str]:
        mask_root = bpy.data.collections.get(ROOT_POLYGON_COLLECTION_NAME)
        if mask_root is None:
            self.report({"ERROR"},
                        f"Collection '{ROOT_POLYGON_COLLECTION_NAME}' not found")
            return {"CANCELLED"}

        # Collect road and kerb mask objects
        road_masks: list[bpy.types.Object] = []
        kerb_masks: list[bpy.types.Object] = []

        road_col = mask_root.children.get("mask_polygon_road")
        if road_col:
            road_masks = [o for o in road_col.all_objects if o.type == "MESH"]

        kerb_col = mask_root.children.get("mask_polygon_kerb")
        if kerb_col:
            kerb_masks = [o for o in kerb_col.all_objects if o.type == "MESH"]

        if not road_masks and not kerb_masks:
            self.report({"WARNING"}, "No road or kerb mask polygons found")
            return {"CANCELLED"}

        _log(f"Road masks: {len(road_masks)}, Kerb masks: {len(kerb_masks)}")

        depsgraph = context.evaluated_depsgraph_get()
        scene = context.scene

        road_obj, kerb_obj = extract_terrain_for_road_kerb(
            road_masks, kerb_masks, scene, depsgraph,
        )

        created = sum(1 for o in (road_obj, kerb_obj) if o is not None)
        parts = []
        if road_obj:
            parts.append(f"road: {road_obj.name}")
        if kerb_obj:
            parts.append(f"kerb: {kerb_obj.name}")

        msg = f"Created {created} collision mesh(es): {', '.join(parts) or 'none'}"
        _log(msg)
        self.report({"INFO"}, msg)
        return {"FINISHED"}


class SAM3_OT_extract_terrain_surface_selected(bpy.types.Operator):
    """Extract terrain surface from selected mask polygon(s) (road/kerb only)."""

    bl_idname = "sam3.extract_terrain_surface_selected"
    bl_label = "Extract Terrain Surface (Selected)"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context: bpy.types.Context) -> set[str]:
        from .mask_select_utils import get_mask_objects

        selected = get_mask_objects(context)
        if not selected:
            self.report({"WARNING"}, "No mask polygon objects selected")
            return {"CANCELLED"}

        # Classify selected objects by tag
        road_masks: list[bpy.types.Object] = []
        kerb_masks: list[bpy.types.Object] = []
        skipped: list[str] = []

        for obj in selected:
            if obj.type != "MESH":
                continue
            tag = None
            for col in getattr(obj, "users_collection", []) or []:
                name = getattr(col, "name", "")
                if name.startswith("mask_polygon_"):
                    tag = name[len("mask_polygon_"):]
                    break

            if tag == "road":
                road_masks.append(obj)
            elif tag == "kerb":
                kerb_masks.append(obj)
            else:
                skipped.append(f"{obj.name} (tag={tag})")

        if skipped:
            _log(f"Skipped non-road/kerb: {', '.join(skipped)}")

        if not road_masks and not kerb_masks:
            self.report({"WARNING"},
                        "No road or kerb mask polygon objects in selection")
            return {"CANCELLED"}

        depsgraph = context.evaluated_depsgraph_get()
        scene = context.scene

        road_obj, kerb_obj = extract_terrain_for_road_kerb(
            road_masks, kerb_masks, scene, depsgraph,
        )

        created = sum(1 for o in (road_obj, kerb_obj) if o is not None)
        msg = f"Created {created} terrain collision mesh(es) from selection"
        _log(msg)
        self.report({"INFO"}, msg)
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Action specs (auto-registered by blender_helpers)
# ---------------------------------------------------------------------------

ACTION_SPECS = [
    ActionSpec(
        operator_cls=SAM3_OT_extract_terrain_surfaces,
        menu_label="Extract Terrain Surfaces (Road+Kerb)",
        icon="MESH_DATA",
    ),
    ActionSpec(
        operator_cls=SAM3_OT_extract_terrain_surface_selected,
        menu_label="Extract Terrain Surface (Selected)",
        icon="MESH_DATA",
    ),
]
