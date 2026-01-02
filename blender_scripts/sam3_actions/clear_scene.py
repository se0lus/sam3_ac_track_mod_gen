from __future__ import annotations

import bpy  # type: ignore[import-not-found]

from . import ActionSpec

# 清除场景中的所有对象，包括材质和纹理，
# 除了 mask_* collection 中的对象以及 script 对象之外全部删除。
#
# 说明（由于项目里没有明确“script对象”的唯一约定，这里做了最稳妥的兼容）：
# - 任何属于 name 以 "mask_" 开头 collection 的对象：保留
# - 任何属于 name 以 "script" 开头 collection 的对象：保留
# - 任何对象名以 "script" 开头（不区分大小写）：保留
#
# 其余对象删除后，再做一次“孤儿数据块”清理（users==0）。
class SAM3_OT_clear_scene_keep_masks(bpy.types.Operator):
    """Clear scene, keep mask_ collections and script objects."""

    bl_idname = "sam3.clear_scene_keep_masks"
    bl_label = "Clear Scene (Keep mask_ & script)"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context: bpy.types.Context):
        # Ensure we're in OBJECT mode for predictable data removal.
        try:
            if getattr(context, "mode", "") != "OBJECT":
                bpy.ops.object.mode_set(mode="OBJECT")
        except Exception:
            pass

        scene = getattr(context, "scene", None)
        if scene is None:
            self.report({"ERROR"}, "No active scene")
            return {"CANCELLED"}

        def _pointer_key(obj: bpy.types.Object) -> int:
            try:
                return int(obj.as_pointer())
            except Exception:
                return id(obj)

        def _collect_objects_from_collections(prefixes: tuple[str, ...]) -> set[int]:
            """
            Collect all objects that are contained in any bpy.data.collections whose name startswith
            one of `prefixes`. Uses `all_objects` to include nested child collections.
            """
            out: set[int] = set()
            try:
                cols = list(getattr(bpy.data, "collections", []) or [])
            except Exception:
                cols = []

            for col in cols:
                try:
                    n = (getattr(col, "name", "") or "").lower()
                    if not any(n.startswith(p) for p in prefixes):
                        continue
                    # all_objects includes nested collection objects; objects may be linked multiple times.
                    for o in list(getattr(col, "all_objects", None) or []):
                        if o is None:
                            continue
                        out.add(_pointer_key(o))
                except Exception:
                    continue
            return out

        def _is_script_named(obj: bpy.types.Object) -> bool:
            try:
                n = (getattr(obj, "name", "") or "").lower()
                if n.startswith("script"):
                    return True
            except Exception:
                pass
            return False

        # Build a robust keep-set up front:
        # - any object under mask_* collections (including nested child collections)
        # - any object under script* collections (including nested)
        # - any object named script*
        # - any collection instance (EMPTY) that instances a mask_*/script* collection
        keep: set[int] = set()
        keep |= _collect_objects_from_collections(("mask_",))
        keep |= _collect_objects_from_collections(("script",))

        for obj in list(getattr(bpy.data, "objects", []) or []):
            if obj is None:
                continue
            if _is_script_named(obj):
                keep.add(_pointer_key(obj))
                continue
            # Keep collection instances that instance mask_*/script* collections
            try:
                if getattr(obj, "type", "") == "EMPTY":
                    inst_col = getattr(obj, "instance_collection", None)
                    if inst_col is not None:
                        cn = (getattr(inst_col, "name", "") or "").lower()
                        if cn.startswith("mask_") or cn.startswith("script"):
                            keep.add(_pointer_key(obj))
            except Exception:
                pass

        to_delete: list[bpy.types.Object] = []
        for obj in list(getattr(bpy.data, "objects", []) or []):
            if obj is None:
                continue
            key = _pointer_key(obj)
            if key in keep:
                continue
            to_delete.append(obj)

        deleted_objs = 0
        for obj in to_delete:
            try:
                bpy.data.objects.remove(obj, do_unlink=True)
                deleted_objs += 1
            except Exception:
                # As a fallback, try operator deletion (needs selection context).
                try:
                    bpy.ops.object.select_all(action="DESELECT")
                except Exception:
                    pass
                try:
                    obj.select_set(True)
                    context.view_layer.objects.active = obj
                except Exception:
                    pass
                try:
                    bpy.ops.object.delete(use_global=False, confirm=False)
                    deleted_objs += 1
                except Exception:
                    pass

        # Helper to remove orphan datablocks (users==0).
        def _try_remove_id(collection, block) -> bool:
            """
            collection: bpy.data.<type> (ID collection)
            block: ID datablock
            """
            try:
                # Some remove() accept do_unlink; some don't.
                try:
                    collection.remove(block, do_unlink=True)  # type: ignore[call-arg]
                except TypeError:
                    collection.remove(block)
                return True
            except Exception:
                return False

        def _purge_orphans_in(collection) -> int:
            removed = 0
            for block in list(collection) if collection is not None else []:
                try:
                    if getattr(block, "users", 0) == 0:
                        if _try_remove_id(collection, block):
                            removed += 1
                except Exception:
                    continue
            return removed

        removed_data = 0
        # Order roughly: object-linked geometry first, then materials/textures/images, etc.
        for col in (
            getattr(bpy.data, "meshes", None),
            getattr(bpy.data, "curves", None),
            getattr(bpy.data, "metaballs", None),
            getattr(bpy.data, "armatures", None),
            getattr(bpy.data, "cameras", None),
            getattr(bpy.data, "lights", None),
            getattr(bpy.data, "materials", None),
            getattr(bpy.data, "textures", None),
            getattr(bpy.data, "images", None),
            getattr(bpy.data, "node_groups", None),
            getattr(bpy.data, "actions", None),
            getattr(bpy.data, "worlds", None),
            getattr(bpy.data, "grease_pencils", None),
        ):
            if col is None:
                continue
            removed_data += _purge_orphans_in(col)

        # Best-effort recursive orphan purge (may fail without outliner context).
        try:
            bpy.ops.outliner.orphans_purge(do_recursive=True)
        except Exception:
            pass

        # Keep selection clean.
        try:
            bpy.ops.object.select_all(action="DESELECT")
        except Exception:
            pass
        try:
            context.view_layer.objects.active = None
        except Exception:
            pass

        # Report summary (ASCII only for menu UI consistency).
        kept_count = 0
        try:
            kept_count = sum(1 for o in (list(getattr(bpy.data, "objects", []) or [])) if _pointer_key(o) in keep)
        except Exception:
            kept_count = 0
        self.report({"INFO"}, f"Cleared: deleted_objects={deleted_objs}, kept_objects={kept_count}, purged_datablocks={removed_data}")
        return {"FINISHED"}


ACTION_SPECS = [
    ActionSpec(
        operator_cls=SAM3_OT_clear_scene_keep_masks,
        menu_label="Clear Scene (Keep mask_ & script)",
        icon="TOOL_SETTINGS",
    )
]

