from __future__ import annotations

from typing import Tuple

import bpy  # type: ignore[import-not-found]
from mathutils import Vector  # type: ignore[import-not-found]

MaskTri = Tuple[
    Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
    Tuple[float, float, float, float],
]


def _pointer_key(obj) -> int:
    try:
        return int(obj.as_pointer())
    except Exception:
        return id(obj)


def dedupe_objects(objs: list[bpy.types.Object]) -> list[bpy.types.Object]:
    seen: set[int] = set()
    out: list[bpy.types.Object] = []
    for o in objs or []:
        if o is None:
            continue
        k = _pointer_key(o)
        if k in seen:
            continue
        seen.add(k)
        out.append(o)
    return out


def get_mask_objects(context: bpy.types.Context) -> list[bpy.types.Object]:
    """
    兼容两种选择来源：
    - View3D：context.selected_objects
    - Outliner：context.selected_ids / context.id（可能是 Object/Collection/LayerCollection）
    """

    def _collect_from_collection(col) -> list[bpy.types.Object]:
        out: list[bpy.types.Object] = []
        seen_cols: set[int] = set()

        def _walk(c) -> None:
            if c is None:
                return
            ck = _pointer_key(c)
            if ck in seen_cols:
                return
            seen_cols.add(ck)
            try:
                out.extend(list(getattr(c, "objects", None) or []))
            except Exception:
                pass
            try:
                for ch in list(getattr(c, "children", None) or []):
                    _walk(ch)
            except Exception:
                pass

        _walk(col)
        return out

    view3d_sel = list(getattr(context, "selected_objects", None) or [])
    if view3d_sel:
        return dedupe_objects(view3d_sel)

    out: list[bpy.types.Object] = []
    sel_ids = list(getattr(context, "selected_ids", None) or [])
    active_id = getattr(context, "id", None)
    if active_id is not None:
        sel_ids.append(active_id)

    for _id in sel_ids:
        try:
            if isinstance(_id, bpy.types.Object):
                out.append(_id)
            elif isinstance(_id, bpy.types.Collection):
                out.extend(_collect_from_collection(_id))
            elif hasattr(bpy.types, "LayerCollection") and isinstance(_id, bpy.types.LayerCollection):
                out.extend(_collect_from_collection(getattr(_id, "collection", None)))
            elif hasattr(_id, "collection") and isinstance(getattr(_id, "collection", None), bpy.types.Collection):
                out.extend(_collect_from_collection(getattr(_id, "collection", None)))
        except Exception:
            continue
    return dedupe_objects(out)


def is_in_mask_collection(obj: bpy.types.Object) -> bool:
    try:
        for col in getattr(obj, "users_collection", []) or []:
            if getattr(col, "name", "").startswith("mask_"):
                return True
    except Exception:
        return False
    return False


def world_bbox_xz_range(obj: bpy.types.Object) -> tuple[float, float, float, float] | None:
    """
    Return (min_x, max_x, min_z, max_z) from object's world-space bound_box.
    """
    bb = getattr(obj, "bound_box", None)
    mw = getattr(obj, "matrix_world", None)
    if not bb or mw is None:
        return None
    try:
        xs: list[float] = []
        zs: list[float] = []
        for co in bb:
            w = mw @ Vector((float(co[0]), float(co[1]), float(co[2])))
            xs.append(float(w.x))
            zs.append(float(w.z))
        if not xs or not zs:
            return None
        return (min(xs), max(xs), min(zs), max(zs))
    except Exception:
        return None


# -------------------------
# 2D geometry helpers (XZ)
# -------------------------
_EPS = 1e-9


def _bbox2_overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    ax0, ax1, az0, az1 = a
    bx0, bx1, bz0, bz1 = b
    return (ax1 >= bx0) and (ax0 <= bx1) and (az1 >= bz0) and (az0 <= bz1)


def _rect_corners_xz(r: tuple[float, float, float, float]) -> list[tuple[float, float]]:
    x0, x1, z0, z1 = r
    return [(x0, z0), (x1, z0), (x1, z1), (x0, z1)]


def _rect_edges(corners: list[tuple[float, float]]) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    return [
        (corners[0], corners[1]),
        (corners[1], corners[2]),
        (corners[2], corners[3]),
        (corners[3], corners[0]),
    ]


def _orient(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    # 2D cross product (b-a) x (c-a)
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a: tuple[float, float], b: tuple[float, float], p: tuple[float, float]) -> bool:
    return (
        min(a[0], b[0]) - _EPS <= p[0] <= max(a[0], b[0]) + _EPS
        and min(a[1], b[1]) - _EPS <= p[1] <= max(a[1], b[1]) + _EPS
    )


def _segments_intersect(
    a1: tuple[float, float],
    a2: tuple[float, float],
    b1: tuple[float, float],
    b2: tuple[float, float],
) -> bool:
    o1 = _orient(a1, a2, b1)
    o2 = _orient(a1, a2, b2)
    o3 = _orient(b1, b2, a1)
    o4 = _orient(b1, b2, a2)

    # Proper intersection
    if (
        ((o1 > _EPS and o2 < -_EPS) or (o1 < -_EPS and o2 > _EPS))
        and ((o3 > _EPS and o4 < -_EPS) or (o3 < -_EPS and o4 > _EPS))
    ):
        return True

    # Colinear / touching cases
    if abs(o1) <= _EPS and _on_segment(a1, a2, b1):
        return True
    if abs(o2) <= _EPS and _on_segment(a1, a2, b2):
        return True
    if abs(o3) <= _EPS and _on_segment(b1, b2, a1):
        return True
    if abs(o4) <= _EPS and _on_segment(b1, b2, a2):
        return True
    return False


def _point_in_rect(p: tuple[float, float], r: tuple[float, float, float, float]) -> bool:
    x0, x1, z0, z1 = r
    return (x0 - _EPS <= p[0] <= x1 + _EPS) and (z0 - _EPS <= p[1] <= z1 + _EPS)


def _point_in_tri(p: tuple[float, float], a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> bool:
    o1 = _orient(a, b, p)
    o2 = _orient(b, c, p)
    o3 = _orient(c, a, p)
    has_neg = (o1 < -_EPS) or (o2 < -_EPS) or (o3 < -_EPS)
    has_pos = (o1 > _EPS) or (o2 > _EPS) or (o3 > _EPS)
    return not (has_neg and has_pos)


def _tri_bbox_xz(tri: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = (tri[0][0], tri[1][0], tri[2][0])
    zs = (tri[0][1], tri[1][1], tri[2][1])
    return (min(xs), max(xs), min(zs), max(zs))


def rect_intersects_triangle_xz(
    r: tuple[float, float, float, float],
    tri: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    tri_bb: tuple[float, float, float, float] | None = None,
) -> bool:
    if tri_bb is None:
        tri_bb = _tri_bbox_xz(tri)
    if not _bbox2_overlap(r, tri_bb):
        return False

    a, b, c = tri

    # Any triangle vertex inside rect
    if _point_in_rect(a, r) or _point_in_rect(b, r) or _point_in_rect(c, r):
        return True

    # Any rect corner inside triangle
    rc = _rect_corners_xz(r)
    if (
        _point_in_tri(rc[0], a, b, c)
        or _point_in_tri(rc[1], a, b, c)
        or _point_in_tri(rc[2], a, b, c)
        or _point_in_tri(rc[3], a, b, c)
    ):
        return True

    # Edge intersection: tri edges vs rect edges
    re = _rect_edges(rc)
    te = [(a, b), (b, c), (c, a)]
    for e1 in te:
        for e2 in re:
            if _segments_intersect(e1[0], e1[1], e2[0], e2[1]):
                return True
    return False


def mask_triangles_xz(
    mask_obj: bpy.types.Object,
    depsgraph,
) -> tuple[list[MaskTri], tuple[float, float, float, float] | None]:
    """
    Return:
    - list of triangles (projected to XZ) and their 2D bboxes
    - overall mask 2D bbox (min_x, max_x, min_z, max_z) for prefilter
    """
    if mask_obj is None or getattr(mask_obj, "type", "") != "MESH":
        return ([], None)

    try:
        obj_eval = mask_obj.evaluated_get(depsgraph) if depsgraph is not None else mask_obj
    except Exception:
        obj_eval = mask_obj

    me = None
    try:
        me = obj_eval.to_mesh()
    except Exception:
        me = None

    if me is None:
        return ([], None)

    tris: list[MaskTri] = []
    overall: tuple[float, float, float, float] | None = None

    try:
        me.calc_loop_triangles()
        mw = getattr(mask_obj, "matrix_world", None)
        if mw is None:
            return ([], None)

        minx = minz = float("inf")
        maxx = maxz = float("-inf")

        verts = getattr(me, "vertices", None)
        if not verts:
            return ([], None)

        for lt in getattr(me, "loop_triangles", []) or []:
            try:
                vi = lt.vertices
                if len(vi) != 3:
                    continue
                v0 = mw @ verts[vi[0]].co
                v1 = mw @ verts[vi[1]].co
                v2 = mw @ verts[vi[2]].co
                tri2d = ((float(v0.x), float(v0.z)), (float(v1.x), float(v1.z)), (float(v2.x), float(v2.z)))
                bb2d = _tri_bbox_xz(tri2d)
                # Skip degenerate triangles in projection
                if (bb2d[1] - bb2d[0] <= _EPS) and (bb2d[3] - bb2d[2] <= _EPS):
                    continue
                tris.append((tri2d, bb2d))
                minx = min(minx, bb2d[0])
                maxx = max(maxx, bb2d[1])
                minz = min(minz, bb2d[2])
                maxz = max(maxz, bb2d[3])
            except Exception:
                continue

        if tris and minx != float("inf"):
            overall = (minx, maxx, minz, maxz)
    finally:
        try:
            obj_eval.to_mesh_clear()
        except Exception:
            pass

    return (tris, overall)


def build_mask_cache(
    context: bpy.types.Context,
    masks: list[bpy.types.Object],
) -> tuple[list[tuple[bpy.types.Object, list[MaskTri], tuple[float, float, float, float] | None]], dict]:
    """
    返回 (mask_cache, stats)
    mask_cache: [(mask_obj, tris, overall_bbox), ...]
    """
    depsgraph = None
    try:
        depsgraph = context.evaluated_depsgraph_get()
    except Exception:
        depsgraph = None

    cache: list[tuple[bpy.types.Object, list[MaskTri], tuple[float, float, float, float] | None]] = []
    total_tris = 0
    mesh_masks = 0
    for m in masks or []:
        tris, overall = mask_triangles_xz(m, depsgraph)
        if tris:
            cache.append((m, tris, overall))
            total_tris += len(tris)
            mesh_masks += 1

    stats = {
        "masks": int(len(masks or [])),
        "mask_meshes": int(mesh_masks),
        "mask_tris": int(total_tris),
    }
    return cache, stats


def iter_visible_candidates(
    context: bpy.types.Context,
    *,
    exclude_objects: list[bpy.types.Object] | None = None,
    exclude_mask_collections: bool = True,
) -> list[bpy.types.Object]:
    visible_objs = list(getattr(context, "visible_objects", None) or [])
    if not visible_objs:
        try:
            visible_objs = list(getattr(context.scene, "objects", []) or [])
        except Exception:
            visible_objs = []

    excludes = set(_pointer_key(o) for o in (exclude_objects or []) if o is not None)

    candidates: list[bpy.types.Object] = []
    for obj in visible_objs:
        if obj is None:
            continue
        if _pointer_key(obj) in excludes:
            continue
        try:
            if hasattr(obj, "visible_get") and not obj.visible_get():
                continue
        except Exception:
            pass
        if exclude_mask_collections and is_in_mask_collection(obj):
            continue
        candidates.append(obj)
    return candidates


def objects_hit_by_mask_cache_xz(
    *,
    context: bpy.types.Context,
    masks: list[bpy.types.Object],
    mask_cache: list[tuple[bpy.types.Object, list[MaskTri], tuple[float, float, float, float] | None]],
    candidates: list[bpy.types.Object] | None = None,
    select_hit_objects: bool = True,
    deselect_first: bool = True,
    set_active: bool = True,
) -> tuple[list[bpy.types.Object], dict]:
    """
    使用预先计算好的 mask_cache，对候选对象做 XZ 相交测试，返回命中的对象列表与统计信息。
    """
    # Ensure we're in Object mode so select_set behaves predictably.
    try:
        if getattr(context, "mode", "") != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
    except Exception:
        pass

    if deselect_first and select_hit_objects:
        try:
            bpy.ops.object.select_all(action="DESELECT")
        except Exception:
            pass
        try:
            context.view_layer.objects.active = None
        except Exception:
            pass

    if candidates is None:
        candidates = iter_visible_candidates(context, exclude_objects=masks, exclude_mask_collections=True)

    considered = 0
    hit_objs: list[bpy.types.Object] = []
    for obj in candidates:
        r = world_bbox_xz_range(obj)
        if r is None:
            continue
        considered += 1
        rect = r

        hit = False
        for _m, tris, overall in mask_cache:
            if overall is not None and not _bbox2_overlap(rect, overall):
                continue
            for tri2d, tri_bb in tris:
                if rect_intersects_triangle_xz(rect, tri2d, tri_bb):
                    hit = True
                    break
            if hit:
                break

        if hit:
            hit_objs.append(obj)
            if select_hit_objects:
                try:
                    obj.select_set(True)
                except Exception:
                    pass

    if set_active and select_hit_objects and hit_objs:
        try:
            context.view_layer.objects.active = hit_objs[0]
        except Exception:
            pass

    stats = {
        "candidates": int(len(candidates)),
        "considered": int(considered),
        "hit": int(len(hit_objs)),
        "masks": int(len(masks or [])),
        "mask_meshes": int(len(mask_cache)),
        "mask_tris": int(sum(len(x[1]) for x in mask_cache)),
    }
    return hit_objs, stats

