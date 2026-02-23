"""
Blender action: Import game objects JSON and create Empty objects.

Reads a game objects JSON file (produced by ai_game_objects), converts 2D
pixel positions to Blender 3D coordinates, and creates Empty objects at those
positions with correct orientation.

Objects are invisible (no mesh) -- they serve as game-logic markers only.
"""

from __future__ import annotations

import json
import math

import bpy  # type: ignore[import-not-found]
from bpy.props import StringProperty, FloatProperty  # type: ignore[import-not-found]
from mathutils import Vector, Matrix  # type: ignore[import-not-found]

from . import ActionSpec


def _ensure_collection(name: str) -> bpy.types.Collection:
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(col)
    return col


def _read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _orient_empty(
    obj: bpy.types.Object,
    orient_2d: list,
) -> None:
    """Set the rotation of an Empty so that its local Z axis points in the
    driving direction (mapped to Blender X-Z plane) and local Y points up.

    In the 2D image, orientation_z = [dx, dy] is the forward direction.
    In Blender:
      forward = (dx, 0, -dy)   (Y-flip for image coords)
      up      = (0, 1, 0)
    """
    dx, dy = float(orient_2d[0]), float(orient_2d[1])
    length = math.sqrt(dx * dx + dy * dy)
    if length < 1e-9:
        return
    dx /= length
    dy /= length

    # Blender forward direction in X-Z plane
    forward = Vector((dx, 0.0, -dy)).normalized()
    up = Vector((0.0, 1.0, 0.0))
    right = up.cross(forward).normalized()
    # Re-orthogonalize up
    up = forward.cross(right).normalized()

    # Rotation matrix: columns = local axes in world space.
    # Col 0 = local X (right), Col 1 = local Y (up), Col 2 = local Z (forward).
    rot = Matrix((
        (right.x, up.x, forward.x),
        (right.y, up.y, forward.y),
        (right.z, up.z, forward.z),
    ))

    obj.rotation_euler = rot.to_euler()


class SAM3_OT_import_game_objects(bpy.types.Operator):
    """Import game objects JSON and create Empty markers"""

    bl_idname = "sam3.import_game_objects"
    bl_label = "Import Game Objects"
    bl_options = {"REGISTER", "UNDO"}

    filepath: StringProperty(  # type: ignore[valid-type]
        name="Game Objects JSON File",
        subtype="FILE_PATH",
        default="",
    )
    height_offset: FloatProperty(  # type: ignore[valid-type]
        name="Height Offset",
        default=2.0,
        description="Height above track surface (Y in Blender)",
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

        objects_list = data.get("objects")
        if not isinstance(objects_list, list) or len(objects_list) == 0:
            self.report({"ERROR"}, "No objects found in JSON")
            return {"CANCELLED"}

        col = _ensure_collection("game_objects")
        img_h = int(self.image_height)
        created = 0

        for obj_data in objects_list:
            name = obj_data.get("name", f"OBJECT_{created}")
            pos = obj_data.get("position")
            if not pos or len(pos) < 2:
                continue

            px, py = float(pos[0]), float(pos[1])
            bx = px / self.pixels_per_unit
            by = self.height_offset
            bz = (img_h - py) / self.pixels_per_unit

            empty = bpy.data.objects.new(name, None)
            empty.empty_display_type = "PLAIN_AXES"
            empty.empty_display_size = 1.0
            empty.location = (bx, by, bz)

            orient = obj_data.get("orientation_z")
            if orient and len(orient) >= 2:
                _orient_empty(empty, orient)

            col.objects.link(empty)
            created += 1

        self.report({"INFO"}, f"Created {created} game objects in 'game_objects' collection")
        return {"FINISHED"}


ACTION_SPECS = [
    ActionSpec(
        operator_cls=SAM3_OT_import_game_objects,
        menu_label="Import Game Objects",
        icon="EMPTY_AXIS",
    ),
]
