"""
Blender object context menu quick-actions framework.

Usage (pick one):
1) Install as an add-on: install/enable this file as an Add-on.
2) Run as a script: run this file in Blender's Text Editor, then in Python Console:
   import blender_helpers; blender_helpers.install_object_context_menu()

Notes:
- This file provides the menu framework and loads actions from `sam3_actions/*`.
- Menu location: View3D > Object Context Menu (Right Click).
"""

from __future__ import annotations

import importlib
import os
import pkgutil
import sys
from typing import TYPE_CHECKING, List, Optional, Sequence, Type

import bpy  # type: ignore[import-not-found]

if TYPE_CHECKING:
    from sam3_actions import ActionSpec

# Debug info for UI when actions fail to load.
_LAST_ACTION_LOAD_ERROR: str = ""
_LAST_ACTION_LOAD_MODULE_ERRORS: List[str] = []

bl_info = {
    "name": "SAM3 Track Seg - Object Context Menu Helpers",
    "author": "sam3_track_seg",
    "version": (0, 1, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Object Context Menu (Right Click)",
    "description": "Add custom quick actions to the right-click menu for selected objects.",
    "category": "Object",
}

# Keys used to make repeated script runs idempotent (Text Editor "Run Script").
_DRIVER_NAMESPACE_DRAW_KEY = "sam3_object_context_menu_draw_func"
_DRIVER_NAMESPACE_OUTLINER_DRAW_KEY = "sam3_outliner_context_menu_draw_func"
_DRIVER_NAMESPACE_VIEW3D_DRAW_KEY = "sam3_view3d_context_menu_draw_func"
_DRIVER_NAMESPACE_CLASSNAMES_KEY = "sam3_object_context_menu_registered_classnames"

# Optional import root override (absolute path).
# If Blender cannot resolve imports when running this script, set this to the directory
# that CONTAINS the `sam3_actions/` folder, e.g.:
#   _SAM3_IMPORT_ROOT_OVERRIDE = r"E:\sam3_track_seg\blender_scripts"
_SAM3_IMPORT_ROOT_OVERRIDE: Optional[str] = r"E:\sam3_track_seg\blender_scripts"

# Optional environment variable override (same meaning as above).
_SAM3_IMPORT_ROOT_ENVVAR = "SAM3_BLENDER_SCRIPTS_DIR"

# Action modules to load. Add/remove modules here.
# Each module should provide ACTION_SPECS or get_action_specs().
# If empty, modules will be auto-discovered from the sam3_actions package.
_ACTION_MODULES: Sequence[str] = ()
# (
#     "sam3_actions.print_selected_names",
#     "sam3_actions.quick_action_1",
#     "sam3_actions.quick_action_2",
# )


def _ensure_local_import_path() -> None:
    """
    Make sure the directory containing this file is on sys.path so that
    `import sam3_actions.*` works when running from Blender's Text Editor.
    """
    candidates: List[str] = []

    # 0) Hard overrides (most reliable).
    if _SAM3_IMPORT_ROOT_OVERRIDE:
        candidates.append(_SAM3_IMPORT_ROOT_OVERRIDE)
    env_root = os.environ.get(_SAM3_IMPORT_ROOT_ENVVAR, "") or ""
    if env_root:
        candidates.append(env_root)

    # 0.5) Blender Preferences > File Paths > Scripts
    try:
        prefs = getattr(getattr(bpy.context, "preferences", None), "filepaths", None)
        script_dir = getattr(prefs, "script_directory", "") if prefs is not None else ""
        if script_dir:
            candidates.append(script_dir)
    except Exception:
        pass

    # 1) Standard add-on execution: __file__ should exist.
    try:
        this_dir = os.path.dirname(os.path.realpath(__file__))
        candidates.append(this_dir)
        # Also consider the parent (some users keep sam3_actions next to the add-on file).
        candidates.append(os.path.dirname(this_dir))
    except Exception:
        pass

    # 2) Blender Text Editor execution: try to get the active text filepath.
    try:
        space = getattr(bpy.context, "space_data", None)
        txt = getattr(space, "text", None) if space is not None else None
        fp = getattr(txt, "filepath", "") if txt is not None else ""
        if fp:
            candidates.append(os.path.dirname(os.path.realpath(fp)))
    except Exception:
        pass

    # 2.5) Also consider all open Text datablocks that have a filepath.
    # This helps when the add-on is installed as a single file, but the project scripts
    # are opened in Blender's Text Editor.
    try:
        for t in getattr(bpy.data, "texts", []) or []:
            fp = getattr(t, "filepath", "") or ""
            if fp:
                candidates.append(os.path.dirname(os.path.realpath(fp)))
    except Exception:
        pass

    # 3) Fallback: current .blend directory.
    try:
        blend_dir = bpy.path.abspath("//") or ""
        if blend_dir:
            candidates.append(blend_dir)
            # Common project layout fallback: <blend_dir>/blender_scripts
            candidates.append(os.path.join(blend_dir, "blender_scripts"))
            # Also try parent/blender_scripts (common repo layout).
            candidates.append(os.path.join(os.path.dirname(os.path.normpath(blend_dir)), "blender_scripts"))
    except Exception:
        pass

    # 4) Blender script paths (user/system).
    try:
        for p in bpy.utils.script_paths():
            if p:
                candidates.append(p)
                candidates.append(os.path.join(p, "addons"))
    except Exception:
        pass

    # Deduplicate while preserving order.
    seen = set()
    uniq: List[str] = []
    for c in candidates:
        c2 = os.path.normpath(os.path.expanduser(str(c)))
        if not c2 or c2 in seen:
            continue
        seen.add(c2)
        uniq.append(c2)

    # Insert the first candidate that looks like it contains sam3_actions.
    for root in uniq:
        try:
            if os.path.isdir(os.path.join(root, "sam3_actions")):
                if root not in sys.path:
                    sys.path.insert(0, root)
                return
        except Exception:
            continue

    # As a last resort, just add all candidates that exist.
    for root in uniq:
        try:
            if os.path.isdir(root) and root not in sys.path:
                sys.path.insert(0, root)
        except Exception:
            pass


def _is_ascii(s: str) -> bool:
    try:
        s.encode("ascii")
        return True
    except Exception:
        return False


def _safe_ascii_label(s: str, *, fallback: str) -> str:
    """
    Ensure UI/report strings are ASCII-only as requested.
    """
    if isinstance(s, str) and _is_ascii(s) and s.strip():
        return s
    return fallback


# =========================
# Idempotent UI registration helpers
# =========================

def _remove_drawfuncs_by_name(menu_type, func_name: str) -> None:
    """
    Blender's Menu.append/remove API removes by *function object identity*.
    When re-running scripts, Blender may accumulate multiple function objects
    that share the same __name__. This helper removes *all* draw callbacks with
    a matching name, making repeated runs idempotent.
    """
    try:
        draw = getattr(menu_type, "draw", None)
        funcs = getattr(draw, "_draw_funcs", None)
        if not funcs:
            return
        for f in list(funcs):
            try:
                if getattr(f, "__name__", "") != func_name:
                    continue
                menu_type.remove(f)
            except Exception:
                continue
    except Exception:
        return


def _remove_all_sam3_menu_draw_callbacks() -> None:
    """
    Best-effort purge for all SAM3 menu draw callbacks, even if they were added
    multiple times across module reloads/script re-runs.
    """
    # Object context menu
    try:
        _remove_drawfuncs_by_name(bpy.types.VIEW3D_MT_object_context_menu, "_draw_in_object_context_menu")
    except Exception:
        pass
    # Generic View3D context menu
    try:
        _remove_drawfuncs_by_name(bpy.types.VIEW3D_MT_context_menu, "_draw_in_view3d_context_menu")
    except Exception:
        pass
    # Outliner menus (names vary by version)
    for mt_name in ("OUTLINER_MT_context_menu", "OUTLINER_MT_collection"):
        mt = getattr(bpy.types, mt_name, None)
        if mt is None:
            continue
        try:
            _remove_drawfuncs_by_name(mt, "_draw_in_outliner_context_menu")
        except Exception:
            pass


# =========================
# Selection helpers
# =========================

def _iter_outliner_selected_ids(context: bpy.types.Context):
    """
    Outliner context menus provide selected_ids / id depending on Blender version
    and which element was right-clicked. Keep this robust and dependency-free.
    """
    # Blender Outliner provides a list of selected datablocks.
    sel_ids = getattr(context, "selected_ids", None) or []
    for _id in sel_ids:
        if _id is not None:
            yield _id

    # Also include the "active" ID if present (often the right-clicked element).
    active_id = getattr(context, "id", None)
    if active_id is not None:
        yield active_id


def _objects_from_id(_id) -> List[bpy.types.Object]:
    out: List[bpy.types.Object] = []
    try:
        if isinstance(_id, bpy.types.Object):
            out.append(_id)
        elif isinstance(_id, bpy.types.Collection):
            # all_objects includes nested collection objects.
            out.extend(list(getattr(_id, "all_objects", None) or []))
    except Exception:
        return []
    return out


def _dedupe_objects(objs: List[bpy.types.Object]) -> List[bpy.types.Object]:
    seen = set()
    out: List[bpy.types.Object] = []
    for o in objs:
        try:
            key = o.as_pointer()
        except Exception:
            key = id(o)
        if key in seen:
            continue
        seen.add(key)
        out.append(o)
    return out


def get_selected_objects(context: bpy.types.Context) -> List[bpy.types.Object]:
    """
    Return selected objects for both:
    - View3D selections (context.selected_objects)
    - Outliner selections where a Collection is selected (context.selected_ids)
    """
    # View3D selection.
    view3d_sel = list(getattr(context, "selected_objects", None) or [])
    if view3d_sel:
        return _dedupe_objects(view3d_sel)

    # Outliner selection: derive objects from selected IDs (Object/Collection).
    outliner_objs: List[bpy.types.Object] = []
    for _id in _iter_outliner_selected_ids(context):
        outliner_objs.extend(_objects_from_id(_id))
    return _dedupe_objects(outliner_objs)


def get_active_object(context: bpy.types.Context) -> Optional[bpy.types.Object]:
    ao = getattr(context, "active_object", None)
    if ao is not None:
        return ao
    # Outliner: if the active ID is an object, treat it as active.
    try:
        active_id = getattr(context, "id", None)
        if isinstance(active_id, bpy.types.Object):
            return active_id
    except Exception:
        pass
    return None


# =========================
# Action loading
# =========================

def _load_action_specs(*, reload_modules: bool) -> List["ActionSpec"]:
    global _LAST_ACTION_LOAD_ERROR, _LAST_ACTION_LOAD_MODULE_ERRORS
    _LAST_ACTION_LOAD_ERROR = ""
    _LAST_ACTION_LOAD_MODULE_ERRORS = []

    _ensure_local_import_path()
    try:
        import sam3_actions
        from sam3_actions import ActionSpec  # noqa: F401
    except Exception as e:
        _LAST_ACTION_LOAD_ERROR = (
            f"Failed to import sam3_actions: {e}. "
            f"Hint: set Preferences>File Paths>Scripts or env {_SAM3_IMPORT_ROOT_ENVVAR}"
        )
        print(f"[SAM3] {_LAST_ACTION_LOAD_ERROR}")
        return []

    specs: List["ActionSpec"] = []
    module_names: List[str] = []
    if _ACTION_MODULES:
        module_names = list(_ACTION_MODULES)
    else:
        # Auto-discover all non-package modules under sam3_actions.
        try:
            module_names = [
                mi.name
                for mi in pkgutil.iter_modules(
                    sam3_actions.__path__, prefix=f"{sam3_actions.__name__}."
                )
                if not mi.ispkg and not mi.name.endswith(".__init__")
            ]
            module_names.sort()
        except Exception:
            module_names = []

    for mod_name in module_names:
        try:
            mod = importlib.import_module(mod_name)
            if reload_modules:
                mod = importlib.reload(mod)
        except Exception as e:
            msg = f"Failed to import action module '{mod_name}': {e}"
            _LAST_ACTION_LOAD_MODULE_ERRORS.append(msg)
            print(f"[SAM3] {msg}")
            continue

        try:
            if hasattr(mod, "get_action_specs"):
                got = mod.get_action_specs()
            else:
                got = getattr(mod, "ACTION_SPECS", [])
            if got:
                specs.extend(list(got))
        except Exception as e:
            msg = f"Failed to load ACTION_SPECS from '{mod_name}': {e}"
            _LAST_ACTION_LOAD_MODULE_ERRORS.append(msg)
            print(f"[SAM3] {msg}")
            continue

    return specs


# =========================
# Context menu integration
# =========================

class SAM3_MT_object_context_menu(bpy.types.Menu):
    """Submenu under the object context menu."""

    bl_label = "SAM3 Quick Tools"
    bl_idname = "SAM3_MT_object_context_menu"

    def draw(self, context: bpy.types.Context):
        layout = self.layout
        sel = get_selected_objects(context)
        active = get_active_object(context)

        # Info line (optional)
        row = layout.row()
        row.enabled = False
        row.label(text=f"Selected: {len(sel)} | Active: {active.name if active else 'None'}")

        specs = _load_action_specs(reload_modules=False)
        if not specs:
            layout.separator()
            row = layout.row()
            row.enabled = False
            row.label(text="No actions loaded")
            # Show a short hint so users don't need to check the console.
            if _LAST_ACTION_LOAD_ERROR:
                row = layout.row()
                row.enabled = False
                row.label(text=_safe_ascii_label(_LAST_ACTION_LOAD_ERROR, fallback="Action import failed (see Console)"))
            elif _LAST_ACTION_LOAD_MODULE_ERRORS:
                row = layout.row()
                row.enabled = False
                row.label(text=_safe_ascii_label(_LAST_ACTION_LOAD_MODULE_ERRORS[0], fallback="Some actions failed (see Console)"))
            return

        layout.separator()
        for sp in specs:
            try:
                if sp.poll is not None and not sp.poll(context):
                    continue
            except Exception:
                continue

            icon = getattr(sp, "icon", "NONE") or "NONE"
            text = _safe_ascii_label(getattr(sp, "menu_label", ""), fallback="Action")
            layout.operator(sp.operator_cls.bl_idname, text=text, icon=icon)


def _draw_in_object_context_menu(self: bpy.types.Menu, context: bpy.types.Context) -> None:
    layout = self.layout
    layout.separator()
    layout.menu(SAM3_MT_object_context_menu.bl_idname, icon="PLUGIN")


def _draw_in_view3d_context_menu(self: bpy.types.Menu, context: bpy.types.Context) -> None:
    """
    Show the submenu in the generic View3D context menu (e.g. right-click on empty space).
    """
    layout = self.layout
    layout.separator()
    layout.menu(SAM3_MT_object_context_menu.bl_idname, icon="PLUGIN")


def _draw_in_outliner_context_menu(self: bpy.types.Menu, context: bpy.types.Context) -> None:
    """
    Show the same submenu in the Outliner right-click context menu.
    This enables displaying the menu when a Collection is selected.
    """
    layout = self.layout
    layout.separator()
    layout.menu(SAM3_MT_object_context_menu.bl_idname, icon="PLUGIN")


def _collect_operator_classes(*, reload_modules: bool) -> List[Type[bpy.types.Operator]]:
    specs = _load_action_specs(reload_modules=reload_modules)
    out: List[Type[bpy.types.Operator]] = []
    for sp in specs:
        try:
            out.append(sp.operator_cls)
        except Exception:
            continue
    return out


def register() -> None:
    # Make registration idempotent even if Blender didn't call unregister()
    # (or when users repeatedly "Run Script" from the Text Editor).
    try:
        _uninstall_previous_installation()
    except Exception:
        pass
    try:
        _remove_all_sam3_menu_draw_callbacks()
    except Exception:
        pass

    # Load/reload action modules when registering to pick up changes.
    action_operator_classes = _collect_operator_classes(reload_modules=True)

    classes_to_register: List[type] = []
    classes_to_register.extend(action_operator_classes)
    classes_to_register.append(SAM3_MT_object_context_menu)

    # Register classes (Operators + Menu).
    for cls in classes_to_register:
        bpy.utils.register_class(cls)

    # Persist the registered class names so we can unregister across repeated script runs.
    try:
        bpy.app.driver_namespace[_DRIVER_NAMESPACE_CLASSNAMES_KEY] = [cls.__name__ for cls in classes_to_register]
    except Exception:
        pass

    # Store the draw function reference in driver_namespace so we can remove it
    # across repeated "Run Script" executions (which create new function objects).
    try:
        bpy.app.driver_namespace[_DRIVER_NAMESPACE_DRAW_KEY] = _draw_in_object_context_menu
    except Exception:
        # driver_namespace should exist, but keep this robust.
        pass
    try:
        bpy.app.driver_namespace[_DRIVER_NAMESPACE_VIEW3D_DRAW_KEY] = _draw_in_view3d_context_menu
    except Exception:
        pass

    # Outliner draw reference (same motivation as above).
    try:
        bpy.app.driver_namespace[_DRIVER_NAMESPACE_OUTLINER_DRAW_KEY] = _draw_in_outliner_context_menu
    except Exception:
        pass

    # Defensive: remove any older duplicated callbacks (same func name).
    _remove_all_sam3_menu_draw_callbacks()

    bpy.types.VIEW3D_MT_object_context_menu.append(_draw_in_object_context_menu)
    # Generic View3D context menu (right-click on empty space).
    try:
        bpy.types.VIEW3D_MT_context_menu.append(_draw_in_view3d_context_menu)
    except Exception:
        pass
    # Also add to Outliner context menu(s), so it appears when selecting Collections.
    for mt_name in ("OUTLINER_MT_context_menu", "OUTLINER_MT_collection"):
        mt = getattr(bpy.types, mt_name, None)
        if mt is None:
            continue
        try:
            mt.append(_draw_in_outliner_context_menu)
        except Exception:
            pass


def unregister() -> None:
    # First, aggressively remove any duplicates by function name (covers cases
    # where driver_namespace was not set, or multiple duplicates accumulated).
    try:
        _remove_all_sam3_menu_draw_callbacks()
    except Exception:
        pass

    # Remove the previously appended menu draw callback.
    # Prefer the persisted reference (works across repeated script runs).
    dn = getattr(bpy.app, "driver_namespace", {})  # type: ignore[assignment]
    old_draw = None
    old_outliner_draw = None
    old_view3d_draw = None
    try:
        old_draw = dn.get(_DRIVER_NAMESPACE_DRAW_KEY)
    except Exception:
        old_draw = None
    try:
        old_view3d_draw = dn.get(_DRIVER_NAMESPACE_VIEW3D_DRAW_KEY)
    except Exception:
        old_view3d_draw = None
    try:
        old_outliner_draw = dn.get(_DRIVER_NAMESPACE_OUTLINER_DRAW_KEY)
    except Exception:
        old_outliner_draw = None

    if old_draw is not None:
        try:
            bpy.types.VIEW3D_MT_object_context_menu.remove(old_draw)
        except Exception:
            pass
        try:
            del dn[_DRIVER_NAMESPACE_DRAW_KEY]
        except Exception:
            pass
    else:
        # Fallback: best-effort remove current reference.
        try:
            bpy.types.VIEW3D_MT_object_context_menu.remove(_draw_in_object_context_menu)
        except Exception:
            pass

    # Remove View3D generic context menu callback.
    if old_view3d_draw is not None:
        try:
            bpy.types.VIEW3D_MT_context_menu.remove(old_view3d_draw)
        except Exception:
            pass
        try:
            del dn[_DRIVER_NAMESPACE_VIEW3D_DRAW_KEY]
        except Exception:
            pass
    else:
        try:
            bpy.types.VIEW3D_MT_context_menu.remove(_draw_in_view3d_context_menu)
        except Exception:
            pass

    # Remove Outliner callbacks (if any).
    if old_outliner_draw is not None:
        for mt_name in ("OUTLINER_MT_context_menu", "OUTLINER_MT_collection"):
            mt = getattr(bpy.types, mt_name, None)
            if mt is None:
                continue
            try:
                mt.remove(old_outliner_draw)
            except Exception:
                pass
        try:
            del dn[_DRIVER_NAMESPACE_OUTLINER_DRAW_KEY]
        except Exception:
            pass
    else:
        for mt_name in ("OUTLINER_MT_context_menu", "OUTLINER_MT_collection"):
            mt = getattr(bpy.types, mt_name, None)
            if mt is None:
                continue
            try:
                mt.remove(_draw_in_outliner_context_menu)
            except Exception:
                pass

    # Unregister classes by stored names (works across repeated script runs).
    class_names: List[str] = []
    try:
        class_names = list(dn.get(_DRIVER_NAMESPACE_CLASSNAMES_KEY) or [])
    except Exception:
        class_names = []

    # Always attempt to unregister the menu as well.
    if "SAM3_MT_object_context_menu" not in class_names:
        class_names.append("SAM3_MT_object_context_menu")

    for name in reversed(class_names):
        old_cls = getattr(bpy.types, name, None)
        if old_cls is None:
            continue
        try:
            bpy.utils.unregister_class(old_cls)
        except Exception:
            pass

    try:
        del dn[_DRIVER_NAMESPACE_CLASSNAMES_KEY]
    except Exception:
        pass


def _uninstall_previous_installation() -> None:
    """
    Best-effort cleanup that works even when this script is executed multiple times
    via Blender's Text Editor (each run creates new Python class/function objects).
    """
    # 0) Remove any duplicated callbacks by name (works even if driver_namespace
    # got overwritten or never populated).
    try:
        _remove_all_sam3_menu_draw_callbacks()
    except Exception:
        pass

    # 1) Remove previously appended menu callback (if any).
    dn = getattr(bpy.app, "driver_namespace", {})  # type: ignore[assignment]
    try:
        old_draw = dn.get(_DRIVER_NAMESPACE_DRAW_KEY)
    except Exception:
        old_draw = None
    try:
        old_view3d_draw = dn.get(_DRIVER_NAMESPACE_VIEW3D_DRAW_KEY)
    except Exception:
        old_view3d_draw = None
    try:
        old_outliner_draw = dn.get(_DRIVER_NAMESPACE_OUTLINER_DRAW_KEY)
    except Exception:
        old_outliner_draw = None
    if old_draw is not None:
        try:
            bpy.types.VIEW3D_MT_object_context_menu.remove(old_draw)
        except Exception:
            pass
        try:
            del dn[_DRIVER_NAMESPACE_DRAW_KEY]
        except Exception:
            pass

    if old_view3d_draw is not None:
        try:
            bpy.types.VIEW3D_MT_context_menu.remove(old_view3d_draw)
        except Exception:
            pass
        try:
            del dn[_DRIVER_NAMESPACE_VIEW3D_DRAW_KEY]
        except Exception:
            pass

    if old_outliner_draw is not None:
        for mt_name in ("OUTLINER_MT_context_menu", "OUTLINER_MT_collection"):
            mt = getattr(bpy.types, mt_name, None)
            if mt is None:
                continue
            try:
                mt.remove(old_outliner_draw)
            except Exception:
                pass
        try:
            del dn[_DRIVER_NAMESPACE_OUTLINER_DRAW_KEY]
        except Exception:
            pass

    # 2) Unregister previously registered classes by stored names.
    class_names: List[str] = []
    try:
        class_names = list(dn.get(_DRIVER_NAMESPACE_CLASSNAMES_KEY) or [])
    except Exception:
        class_names = []

    # Always attempt to unregister the menu as well.
    if "SAM3_MT_object_context_menu" not in class_names:
        class_names.append("SAM3_MT_object_context_menu")

    for name in reversed(class_names):
        old_cls = getattr(bpy.types, name, None)
        if old_cls is None:
            continue
        try:
            bpy.utils.unregister_class(old_cls)
        except Exception:
            pass

    try:
        del dn[_DRIVER_NAMESPACE_CLASSNAMES_KEY]
    except Exception:
        pass


def install_object_context_menu() -> None:
    """
    Framework entry point for script usage.
    Safe to run multiple times: it attempts to unregister first, then registers again.
    """
    # Important: do a robust cleanup that also handles repeated Text Editor runs.
    _uninstall_previous_installation()

    # Also try the standard unregister (covers the add-on case cleanly).
    try:
        unregister()
    except Exception:
        pass
    register()


if __name__ == "__main__":
    # When run directly in Blender's Text Editor, install the menu by default.
    install_object_context_menu()