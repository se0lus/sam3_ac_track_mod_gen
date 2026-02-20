"""
Blender action: Import virtual wall JSON and create wall meshes.

Reads a wall JSON file (produced by ai_wall_generator), converts 2D pixel
coordinates to Blender 3D coordinates, and creates tall thin face-only meshes
for each wall segment in the 'collision' collection with 1WALL_N naming.
"""

from __future__ import annotations

import json
import os
import sys

import bpy  # type: ignore[import-not-found]
import bmesh  # type: ignore[import-not-found]
from bpy.props import StringProperty, FloatProperty  # type: ignore[import-not-found]

from . import ActionSpec


def _ensure_collection(name: str) -> bpy.types.Collection:
    """Get or create a collection by name, linked to the scene."""
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(col)
    return col


def _read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _create_wall_mesh(
    name: str,
    points_2d: list,
    closed: bool,
    wall_height: float,
    collection: bpy.types.Collection,
    pixels_per_unit: float,
    image_height: int,
) -> bpy.types.Object:
    """Create a thin face-only mesh for a wall segment.

    The wall lies in the Blender X-Z plane (Y=0 at the base, Y=wall_height at
    the top).  2D pixel coordinates are mapped so that:
      Blender X = pixel_x / pixels_per_unit
      Blender Z = (image_height - pixel_y) / pixels_per_unit   (flip Y axis)
    """
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()

    n = len(points_2d)
    segments = n if closed else n - 1

    for i in range(segments):
        p0 = points_2d[i]
        p1 = points_2d[(i + 1) % n]

        x0 = p0[0] / pixels_per_unit
        z0 = (image_height - p0[1]) / pixels_per_unit
        x1 = p1[0] / pixels_per_unit
        z1 = (image_height - p1[1]) / pixels_per_unit

        v0 = bm.verts.new((x0, 0.0, z0))
        v1 = bm.verts.new((x1, 0.0, z1))
        v2 = bm.verts.new((x1, wall_height, z1))
        v3 = bm.verts.new((x0, wall_height, z0))
        bm.faces.new((v0, v1, v2, v3))

    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    return obj


class SAM3_OT_import_walls(bpy.types.Operator):
    """Import wall JSON and create collision wall meshes"""

    bl_idname = "sam3.import_walls"
    bl_label = "Import Virtual Walls"
    bl_options = {"REGISTER", "UNDO"}

    filepath: StringProperty(  # type: ignore[valid-type]
        name="Wall JSON File",
        subtype="FILE_PATH",
        default="",
    )
    wall_height: FloatProperty(  # type: ignore[valid-type]
        name="Wall Height",
        default=5.0,
        min=0.1,
        description="Height of generated wall meshes (Blender units)",
    )
    pixels_per_unit: FloatProperty(  # type: ignore[valid-type]
        name="Pixels Per Unit",
        default=1.0,
        min=0.001,
        description="How many image pixels correspond to 1 Blender unit",
    )
    image_height: FloatProperty(  # type: ignore[valid-type]
        name="Image Height",
        default=1024.0,
        min=1.0,
        description="Height of the source image in pixels (for Y-axis flip)",
    )

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        if not self.filepath:
            self.report({"ERROR"}, "No file selected")
            return {"CANCELLED"}

        try:
            data = _read_json(self.filepath)
        except Exception as e:
            self.report({"ERROR"}, f"Failed to read JSON: {e}")
            return {"CANCELLED"}

        walls = data.get("walls")
        if not isinstance(walls, list) or len(walls) == 0:
            self.report({"ERROR"}, "No walls found in JSON")
            return {"CANCELLED"}

        col = _ensure_collection("collision_walls")
        img_h = int(self.image_height)
        created = 0

        for i, wall in enumerate(walls):
            pts = wall.get("points") or []
            if len(pts) < 2:
                continue
            closed = wall.get("closed", True)
            name = f"1WALL_{created}"
            _create_wall_mesh(
                name=name,
                points_2d=pts,
                closed=closed,
                wall_height=self.wall_height,
                collection=col,
                pixels_per_unit=self.pixels_per_unit,
                image_height=img_h,
            )
            created += 1

        self.report({"INFO"}, f"Created {created} wall objects in 'collision_walls' collection")
        return {"FINISHED"}


ACTION_SPECS = [
    ActionSpec(
        operator_cls=SAM3_OT_import_walls,
        menu_label="Import Virtual Walls",
        icon="MESH_PLANE",
    ),
]
