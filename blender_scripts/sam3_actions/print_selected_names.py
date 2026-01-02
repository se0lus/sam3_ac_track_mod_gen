from __future__ import annotations

from typing import List

import bpy  # type: ignore[import-not-found]

from . import ActionSpec


def _selected_objects(context: bpy.types.Context) -> List[bpy.types.Object]:
    return list(getattr(context, "selected_objects", None) or [])


class SAM3_OT_selected_print_names(bpy.types.Operator):
    """Example: print selected object names (to verify the framework)."""

    bl_idname = "sam3.selected_print_names"
    bl_label = "Print Selected Object Names"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context: bpy.types.Context):
        sel = _selected_objects(context)
        if not sel:
            self.report({"WARNING"}, "No objects selected")
            return {"CANCELLED"}
        names = [o.name for o in sel]
        print(f"[SAM3] selected_objects({len(names)}): {names}")
        self.report({"INFO"}, f"Printed {len(names)} selected object names (see Console)")
        return {"FINISHED"}


ACTION_SPECS = [
    ActionSpec(
        operator_cls=SAM3_OT_selected_print_names,
        menu_label="Print Selected Object Names",
        icon="INFO",
    )
]

