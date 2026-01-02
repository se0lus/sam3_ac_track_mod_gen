from __future__ import annotations

import bpy  # type: ignore[import-not-found]

from . import ActionSpec
from .mask_select_utils import build_mask_cache, get_mask_objects, objects_hit_by_mask_cache_xz


class SAM3_OT_selected_obj_by_mask_polygons(bpy.types.Operator):
    """
    首先，将选中的所有对象作为mask对象，遍历这些mask
    遍历场景中所有不在”mask_*“集合中的可见对象，作为候选对象
    如果候选对象的boundingbox 落在mask对象y方向的投影范围内（x，z平面），则选中这个候选对象
    """

    bl_idname = "sam3.selected_action_1"
    bl_label = "Quick Action 1 (Entry)"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context: bpy.types.Context):
        # 沿 Y 方向投影（忽略 Y），在 XZ 平面上用“mask网格三角形投影”对候选对象 bbox 做相交判断。
        masks = get_mask_objects(context)
        if not masks:
            self.report({"WARNING"}, "未选中 mask 对象（请先选择 mask 对象）")
            return {"CANCELLED"}

        mask_cache, mask_stats = build_mask_cache(context, masks)
        if not mask_cache:
            self.report({"WARNING"}, "mask 对象没有有效网格面（需要 MESH 且有面）")
            return {"CANCELLED"}

        hit_objs, stats = objects_hit_by_mask_cache_xz(
            context=context,
            masks=masks,
            mask_cache=mask_cache,
            select_hit_objects=True,
            deselect_first=True,
            set_active=True,
        )

        self.report(
            {"INFO"},
            f"Selected {len(hit_objs)} objects (candidates={stats.get('considered')}, masks={mask_stats.get('masks')}, mask_meshes={mask_stats.get('mask_meshes')}, mask_tris={mask_stats.get('mask_tris')})",
        )
        return {"FINISHED"}


ACTION_SPECS = [
    ActionSpec(
        operator_cls=SAM3_OT_selected_obj_by_mask_polygons,
        menu_label="Select Objects By Mask XZ Polygons",
        icon="TOOL_SETTINGS",
    )
]

