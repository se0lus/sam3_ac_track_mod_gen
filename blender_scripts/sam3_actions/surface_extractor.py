"""
Blender action: Extract Collision Surfaces from mask clip JSONs.

For each consolidated ``{tag}_clip.json`` file in the configured clips
directory, this operator:

1. Reads the polygon boundary from the JSON.
2. Generates an interior sampling grid (density varies by material type).
3. Raycasts each sample point downward (-Y in Blender) onto scene geometry.
4. Builds a mesh from the projected 3D points via Delaunay triangulation.
5. Names the object using Assetto Corsa collision conventions (TODO-6).
6. Places the object in the ``collision`` collection.
"""

from __future__ import annotations

import glob
import os
import sys

import bpy  # type: ignore[import-not-found]
import bmesh  # type: ignore[import-not-found]
from mathutils import Vector  # type: ignore[import-not-found]

from . import ActionSpec

# Ensure the project ``script/`` directory is importable so we can use the
# pure-Python helpers from ``surface_extraction.py``.
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_script_dir = os.path.join(_project_root, "script")
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from surface_extraction import (  # noqa: E402
    extract_polygon_xz,
    generate_collision_name,
    generate_sampling_grid,
    load_clip_polygons,
    triangulate_points,
)

# Import config -- blender_scripts/ is already on sys.path by the time
# sam3_actions is loaded (blender_helpers ensures this).
from config import (  # noqa: E402
    COLLISION_COLLECTION_NAME,
    CONSOLIDATED_CLIPS_DIR,
    SURFACE_SAMPLING_DENSITY_DEFAULT,
    SURFACE_SAMPLING_DENSITY_GRASS,
    SURFACE_SAMPLING_DENSITY_KERB,
    SURFACE_SAMPLING_DENSITY_ROAD,
    SURFACE_SAMPLING_DENSITY_SAND,
)

# Map material tag -> sampling density
_DENSITY_MAP = {
    "road": SURFACE_SAMPLING_DENSITY_ROAD,
    "grass": SURFACE_SAMPLING_DENSITY_GRASS,
    "kerb": SURFACE_SAMPLING_DENSITY_KERB,
    "sand": SURFACE_SAMPLING_DENSITY_SAND,
}


def _get_density(tag: str) -> float:
    return _DENSITY_MAP.get(tag.strip().lower(), SURFACE_SAMPLING_DENSITY_DEFAULT)


def _get_or_create_collection(name: str) -> bpy.types.Collection:
    """Return existing collection *name* or create and link it to the scene."""
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


def _raycast_down(
    scene_objects: list[bpy.types.Object],
    depsgraph,
    x: float,
    z: float,
    ray_origin_y: float = 1000.0,
) -> tuple[float, float, float] | None:
    """Cast a ray from (x, ray_origin_y, z) in -Y direction.

    Returns the first hit point ``(x, y, z)`` in world space, or ``None``.
    """
    origin = Vector((x, ray_origin_y, z))
    direction = Vector((0.0, -1.0, 0.0))

    best_hit: tuple[float, float, float] | None = None
    best_dist: float = float("inf")

    for obj in scene_objects:
        if obj.type != "MESH":
            continue
        try:
            # Transform ray into object local space
            inv = obj.matrix_world.inverted_safe()
            local_origin = inv @ origin
            local_dir = (inv.to_3x3() @ direction).normalized()

            success, location, _normal, _index = obj.ray_cast(local_origin, local_dir, depsgraph=depsgraph)
            if success:
                world_hit = obj.matrix_world @ location
                dist = (world_hit - origin).length
                if dist < best_dist:
                    best_dist = dist
                    best_hit = (float(world_hit.x), float(world_hit.y), float(world_hit.z))
        except Exception:
            continue

    return best_hit


def _is_in_collision_collection(obj: bpy.types.Object) -> bool:
    for col in getattr(obj, "users_collection", []) or []:
        if getattr(col, "name", "") == COLLISION_COLLECTION_NAME:
            return True
    return False


def _is_in_mask_collection(obj: bpy.types.Object) -> bool:
    for col in getattr(obj, "users_collection", []) or []:
        if getattr(col, "name", "").startswith("mask_"):
            return True
    return False


def _get_scene_raycast_targets() -> list[bpy.types.Object]:
    """Collect visible MESH objects excluding mask and collision collections."""
    targets = []
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        if _is_in_collision_collection(obj):
            continue
        if _is_in_mask_collection(obj):
            continue
        targets.append(obj)
    return targets


class SAM3_OT_extract_surfaces(bpy.types.Operator):
    """Extract 3D collision surfaces from consolidated mask clip JSONs."""

    bl_idname = "sam3.extract_surfaces"
    bl_label = "Extract Collision Surfaces"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context: bpy.types.Context) -> set[str]:
        clips_dir = CONSOLIDATED_CLIPS_DIR
        if not os.path.isdir(clips_dir):
            self.report({"ERROR"}, f"Clips directory not found: {clips_dir}")
            return {"CANCELLED"}

        clip_files = sorted(glob.glob(os.path.join(clips_dir, "*_clip.json")))
        if not clip_files:
            self.report({"WARNING"}, f"No *_clip.json files in {clips_dir}")
            return {"CANCELLED"}

        depsgraph = context.evaluated_depsgraph_get()
        scene_targets = _get_scene_raycast_targets()
        if not scene_targets:
            self.report({"WARNING"}, "No scene mesh objects available for raycasting")
            return {"CANCELLED"}

        collision_col = _get_or_create_collection(COLLISION_COLLECTION_NAME)

        total_created = 0
        # Track per-tag index for collision naming
        tag_counters: dict[str, int] = {}

        for clip_path in clip_files:
            try:
                clip_data = load_clip_polygons(clip_path)
            except Exception as e:
                print(f"[surface_extractor] Failed to read {clip_path}: {e}")
                continue

            tag = clip_data["tag"]
            density = _get_density(tag)
            include_polys = clip_data.get("include") or []

            for poly_dict in include_polys:
                polygon_xz = extract_polygon_xz(poly_dict)
                if len(polygon_xz) < 3:
                    continue

                # Generate sampling grid
                try:
                    grid_pts, boundary_idx = generate_sampling_grid(polygon_xz, density)
                except Exception as e:
                    print(f"[surface_extractor] Grid generation failed for {tag}: {e}")
                    continue

                if not grid_pts:
                    continue

                # Raycast each grid point down onto scene geometry
                points_3d: list[tuple[float, float, float]] = []
                for gx, gz in grid_pts:
                    hit = _raycast_down(scene_targets, depsgraph, gx, gz)
                    if hit is not None:
                        points_3d.append(hit)

                if len(points_3d) < 3:
                    continue

                # Triangulate
                faces = triangulate_points(points_3d)
                if not faces:
                    continue

                # Generate name
                idx = tag_counters.get(tag, 0)
                tag_counters[tag] = idx + 1
                try:
                    obj_name = generate_collision_name(tag, idx)
                except ValueError:
                    obj_name = f"1{tag.upper()}_{idx}"

                # Build Blender mesh
                mesh_data = bpy.data.meshes.new(f"{obj_name}_mesh")
                bm = bmesh.new()
                try:
                    verts = [bm.verts.new(p) for p in points_3d]
                    bm.verts.ensure_lookup_table()
                    for f in faces:
                        try:
                            bm.faces.new([verts[f[0]], verts[f[1]], verts[f[2]]])
                        except Exception:
                            continue
                    bm.to_mesh(mesh_data)
                finally:
                    bm.free()

                obj = bpy.data.objects.new(obj_name, mesh_data)
                _link_to_collection(obj, collision_col)
                total_created += 1

        self.report({"INFO"}, f"Created {total_created} collision mesh(es) in '{COLLISION_COLLECTION_NAME}' collection")
        return {"FINISHED"}


ACTION_SPECS = [
    ActionSpec(
        operator_cls=SAM3_OT_extract_surfaces,
        menu_label="Extract Collision Surfaces",
        icon="MESH_DATA",
    ),
]
