# import blender libraries
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import bpy  # type: ignore[import-not-found]
import bmesh  # type: ignore[import-not-found]
from mathutils import Matrix  # type: ignore[import-not-found]


_ROOT_CURVE_COLLECTION_NAME = "mask_curve2D_collection"
_ROOT_POLYGON_COLLECTION_NAME = "mask_polygon_collection"


def _sanitize_name(s: str, *, max_len: int = 63) -> str:
    s2 = re.sub(r"[^0-9A-Za-z_\-]+", "_", str(s)).strip("_")
    if not s2:
        s2 = "unnamed"
    return s2[:max_len]


def _iter_blender_json_files(root: str) -> Iterable[str]:
    for r, _, files in os.walk(root):
        for name in files:
            if name.lower().endswith("_blender.json"):
                yield os.path.join(r, name)


def _guess_tag_from_path(json_path: str) -> str:
    # e.g. .../road/clip_0_blender.json -> road
    return _sanitize_name(os.path.basename(os.path.dirname(json_path)) or "unknown")


def _get_clip_name_from_filename(json_path: str) -> str:
    base = os.path.splitext(os.path.basename(json_path))[0]  # clip_0_blender
    base = re.sub(r"_blender$", "", base, flags=re.IGNORECASE)
    base = _sanitize_name(base)
    # keep "clip_" prefix if present; otherwise prefix for consistency
    return base if base.lower().startswith("clip_") else f"clip_{base}"


def _dedupe_and_close_points(points_xyz: List[List[float]], *, eps: float = 1e-9) -> List[Tuple[float, float, float]]:
    """
    - 去掉连续重复点
    - 若首尾相同（within eps），去掉末尾重复点
    - 返回 (x,y,z) tuples
    """

    def _dist2(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
        return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2

    out: List[Tuple[float, float, float]] = []
    last: Optional[Tuple[float, float, float]] = None
    for p in points_xyz:
        if not isinstance(p, list) or len(p) < 3:
            continue
        cur = (float(p[0]), float(p[1]), float(p[2]))
        if last is not None and _dist2(cur, last) <= eps * eps:
            continue
        out.append(cur)
        last = cur

    if len(out) >= 2 and _dist2(out[0], out[-1]) <= eps * eps:
        out.pop()
    return out


@dataclass(frozen=True)
class _GroupKey:
    tag: str
    clip: str
    mask_index: int
    kind: str  # "include" / "exclude"


@dataclass
class _GroupData:
    tag: str
    clip: str
    mask_index: int
    kind: str
    polys: List[List[Tuple[float, float, float]]] = field(default_factory=list)
    probs: List[float] = field(default_factory=list)


def _load_and_group_polygons(blender_input_path: str) -> Dict[_GroupKey, _GroupData]:
    """
    递归读取 blender_input_path 下所有 *_blender.json，按 (tag, clip, mask_index, kind) 聚合。
    points_xyz 认为已经是 Blender 坐标系 (X,Y,Z)。
    """
    groups: Dict[_GroupKey, _GroupData] = {}
    json_files = list(_iter_blender_json_files(blender_input_path))
    if not json_files:
        print(f"[generate_polygons] 未找到 *_blender.json: {blender_input_path}")
        return groups

    print(f"[generate_polygons] 找到 {len(json_files)} 个 *_blender.json")
    for jp in json_files:
        try:
            with open(jp, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception as e:
            print(f"[generate_polygons] 读取失败: {jp} err={e}")
            continue

        clip = _get_clip_name_from_filename(jp)
        polygons = (obj.get("polygons") or {}) if isinstance(obj, dict) else {}
        if not isinstance(polygons, dict):
            continue

        for kind in ("include", "exclude"):
            items = polygons.get(kind) or []
            if not isinstance(items, list):
                continue
            for it in items:
                if not isinstance(it, dict):
                    continue
                tag = _sanitize_name(str(it.get("tag") or _guess_tag_from_path(jp)))
                try:
                    mask_index = int(it.get("mask_index", -1))
                except Exception:
                    mask_index = -1
                prob = float(it.get("prob", 0.0) or 0.0)
                points_xyz = it.get("points_xyz") or []
                if not isinstance(points_xyz, list) or len(points_xyz) < 3:
                    continue
                pts = _dedupe_and_close_points(points_xyz)
                if len(pts) < 3:
                    continue

                key = _GroupKey(tag=tag, clip=clip, mask_index=mask_index, kind=kind)
                gd = groups.get(key)
                if gd is None:
                    gd = _GroupData(tag=tag, clip=clip, mask_index=mask_index, kind=kind)
                    groups[key] = gd
                gd.polys.append(list(pts))
                gd.probs.append(prob)

    return groups


def _get_or_create_child_collection(parent: bpy.types.Collection, name: str) -> bpy.types.Collection:
    for c in parent.children:
        if c.name == name:
            return c
    col = bpy.data.collections.new(name)
    parent.children.link(col)
    return col


def _get_or_create_root_collection(name: str) -> bpy.types.Collection:
    scene_root = bpy.context.scene.collection
    for c in scene_root.children:
        if c.name == name:
            return c
    col = bpy.data.collections.new(name)
    scene_root.children.link(col)
    return col


def _link_object_to_collection(obj: bpy.types.Object, col: bpy.types.Collection) -> None:
    # Ensure object is in target collection; unlink from master scene collection to avoid duplicates.
    if obj.name not in col.objects:
        col.objects.link(obj)
    try:
        root = bpy.context.scene.collection
        if obj.name in root.objects:
            root.objects.unlink(obj)
    except Exception:
        pass


def _create_curve_object(name: str, polys: List[List[Tuple[float, float, float]]]) -> bpy.types.Object:
    curve_data = bpy.data.curves.new(name=f"{name}_data", type="CURVE")
    curve_data.dimensions = "3D"
    curve_data.resolution_u = 1

    for pts in polys:
        if len(pts) < 3:
            continue
        spline = curve_data.splines.new("POLY")
        spline.points.add(len(pts) - 1)
        for i, (x, y, z) in enumerate(pts):
            spline.points[i].co = (float(x), float(y), float(z), 1.0)
        spline.use_cyclic_u = True

    obj = bpy.data.objects.new(name, curve_data)
    # 让曲线更容易看见
    try:
        obj.show_in_front = True
    except Exception:
        pass
    return obj


def _signed_area_2d(uv: List[Tuple[float, float]]) -> float:
    # Shoelace formula. Positive => CCW.
    if len(uv) < 3:
        return 0.0
    area2 = 0.0
    for i in range(len(uv)):
        x1, y1 = uv[i]
        x2, y2 = uv[(i + 1) % len(uv)]
        area2 += x1 * y2 - x2 * y1
    return 0.5 * area2


def _choose_constant_axis_and_value(
    include_polys: List[List[Tuple[float, float, float]]],
    exclude_polys: List[List[Tuple[float, float, float]]],
) -> Tuple[str, float]:
    """
    这些 mask polygon 一般都在某个平面上（某一轴近似常量）。
    这里通过范围最小的轴来判断“常量轴”，并给出该轴的平均值用于平移。
    返回:
      - axis: "x" / "y" / "z"
      - value: 平移用的常量值（mean）
    """
    xs: List[float] = []
    ys: List[float] = []
    zs: List[float] = []
    for poly in (include_polys or []) + (exclude_polys or []):
        for x, y, z in poly:
            xs.append(float(x))
            ys.append(float(y))
            zs.append(float(z))

    if not xs:
        return ("z", 0.0)

    rx = max(xs) - min(xs)
    ry = max(ys) - min(ys)
    rz = max(zs) - min(zs)
    if rx <= ry and rx <= rz:
        return ("x", sum(xs) / len(xs))
    if ry <= rx and ry <= rz:
        return ("y", sum(ys) / len(ys))
    return ("z", sum(zs) / len(zs))


def _project_to_curve_xy(
    pts: List[Tuple[float, float, float]], const_axis: str
) -> List[Tuple[float, float]]:
    """
    把 3D 点投影到 2D Curve 的 (x,y) 平面坐标:
    - const_axis == "y": 原数据多在 XZ 平面 => (x,z)
    - const_axis == "z": 原数据多在 XY 平面 => (x,y)
    - const_axis == "x": 原数据多在 YZ 平面 => (y,z)
    """
    if const_axis == "y":
        return [(float(x), float(z)) for (x, _y, z) in pts]
    if const_axis == "x":
        return [(float(y), float(z)) for (_x, y, z) in pts]
    # default: const z
    return [(float(x), float(y)) for (x, y, _z) in pts]


def _ensure_winding_2d(
    pts: List[Tuple[float, float, float]], *, const_axis: str, want_ccw: bool
) -> List[Tuple[float, float, float]]:
    # Return possibly reversed point order to match desired winding in projected 2D.
    uv = _project_to_curve_xy(pts, const_axis)
    a = _signed_area_2d(uv)
    if abs(a) < 1e-12:
        return pts
    is_ccw = a > 0.0
    if is_ccw == want_ccw:
        return pts
    return list(reversed(pts))


def _create_filled_curve_object_for_polygons(
    name: str,
    include_polys: List[List[Tuple[float, float, float]]],
    exclude_polys: List[List[Tuple[float, float, float]]],
) -> bpy.types.Object:
    """
    用 2D Curve 的填充能力来做 include - exclude：
    - include 作为外环（统一 CCW）
    - exclude 作为洞（统一 CW）
    然后把 curve convert 成 mesh，得到带洞的面。

    注意：这是针对 mask 多边形（通常都在同一平面/XY）最稳的方式之一。
    """
    curve_data = bpy.data.curves.new(name=f"{name}_data", type="CURVE")
    curve_data.dimensions = "2D"
    curve_data.resolution_u = 1
    # 开启填充：Blender 5.0 的 Python 枚举有个坑（RNA 显示 FULL/HALF，但实际可设的是 BOTH/NONE）。
    # 这里按“优先 BOTH，其次 FULL”做兼容，确保转成 mesh 后有面。
    for v in ("BOTH", "FULL", "BACK", "FRONT"):
        try:
            curve_data.fill_mode = v  # type: ignore[assignment]
            break
        except Exception:
            continue

    const_axis, const_value = _choose_constant_axis_and_value(include_polys, exclude_polys)

    def _add_spline(pts_in: List[Tuple[float, float, float]]) -> None:
        if len(pts_in) < 3:
            return
        uv = _project_to_curve_xy(pts_in, const_axis)
        spline = curve_data.splines.new("POLY")
        spline.points.add(len(uv) - 1)
        for i, (u, v) in enumerate(uv):
            spline.points[i].co = (float(u), float(v), 0.0, 1.0)
        spline.use_cyclic_u = True

    for pts in include_polys:
        _add_spline(_ensure_winding_2d(pts, const_axis=const_axis, want_ccw=True))
    for pts in exclude_polys:
        _add_spline(_ensure_winding_2d(pts, const_axis=const_axis, want_ccw=False))

    obj = bpy.data.objects.new(name, curve_data)
    # 把“2D 曲线所在的 XY 平面”映射回原来的 3D 平面
    if const_axis == "y":
        # local (u,v,0) => world (u, const_y, v)
        rot = Matrix(((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)))
        obj.matrix_world = Matrix.Translation((0.0, float(const_value), 0.0)) @ rot.to_4x4()
    elif const_axis == "x":
        # local (u,v,0) => world (const_x, u, v)
        rot = Matrix(((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
        obj.matrix_world = Matrix.Translation((float(const_value), 0.0, 0.0)) @ rot.to_4x4()
    else:
        # const z: local (u,v,0) => world (u, v, const_z)
        obj.matrix_world = Matrix.Translation((0.0, 0.0, float(const_value)))
    return obj


def _triangulate_mesh_object_inplace(obj: bpy.types.Object) -> None:
    if obj.type != "MESH" or obj.data is None:
        return
    me = obj.data
    bm = bmesh.new()
    try:
        bm.from_mesh(me)
        bm.faces.ensure_lookup_table()
        if bm.faces:
            bmesh.ops.triangulate(bm, faces=list(bm.faces))
        bm.to_mesh(me)
    finally:
        bm.free()


def _create_mesh_object(name: str, polys: List[List[Tuple[float, float, float]]]) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(name=f"{name}_mesh")
    obj = bpy.data.objects.new(name, mesh)

    bm = bmesh.new()
    created_faces: List[bmesh.types.BMFace] = []

    for pi, pts in enumerate(polys):
        if len(pts) < 3:
            continue
        try:
            verts = [bm.verts.new((float(x), float(y), float(z))) for (x, y, z) in pts]
            bm.verts.ensure_lookup_table()
            face = bm.faces.new(verts)
            created_faces.append(face)
        except Exception as e:
            # 自交/退化 poly 可能导致 faces.new 抛错；跳过即可
            print(f"[generate_polygons] mesh face 创建失败: {name} poly#{pi} err={e}")
            continue

    if created_faces:
        try:
            bmesh.ops.triangulate(bm, faces=created_faces)
        except Exception as e:
            print(f"[generate_polygons] triangulate 失败: {name} err={e}")

    bm.to_mesh(mesh)
    bm.free()
    return obj


def generate_polygons_from_blender_clips(blender_input_path: str, output_file: str) -> None:
    """
    读取 blender_input_path 下的所有 `*_blender.json` 文件(由 `map_mask_to_blender` 生成)，并生成 polygons，保存到 output_file（新的 .blend）。

    输出 .blend 的结构如下（按 tag，再按 clip 分组，避免多 clip 重名）：

    mask_curve2D_collection\n
        mask_curve2D_{tag}\n
            {clip}\n
                mask_curve2D_{tag}_{clip}_{mask_index}_include\n
                mask_curve2D_{tag}_{clip}_{mask_index}_exclude\n
                ...\n

    mask_polygon_collection\n
        mask_polygon_{tag}\n
            {clip}\n
                mask_polygon_{tag}_{clip}_{mask_index}\n   (include 已扣除 exclude，不再区分 include/exclude)\n
                ...\n
    """

    blender_input_path = os.path.abspath(blender_input_path)
    output_file = os.path.abspath(output_file)
    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    groups = _load_and_group_polygons(blender_input_path)
    if not groups:
        print("[generate_polygons] 没有可用的 polygon 数据，退出。")
        return

    # collections
    root_curve = _get_or_create_root_collection(_ROOT_CURVE_COLLECTION_NAME)
    root_poly = _get_or_create_root_collection(_ROOT_POLYGON_COLLECTION_NAME)

    created_curve_objects = 0
    created_mesh_objects = 0

    # 1) 仍然按 include/exclude 生成曲线对象（便于排查）
    for key in sorted(groups.keys(), key=lambda k: (k.tag, k.clip, k.mask_index, k.kind)):
        gd = groups[key]
        tag_curve_col = _get_or_create_child_collection(root_curve, f"mask_curve2D_{gd.tag}")
        clip_curve_col = _get_or_create_child_collection(tag_curve_col, gd.clip)
        curve_obj_name = _sanitize_name(
            f"mask_curve2D_{gd.tag}_{gd.clip}_{gd.mask_index}_{gd.kind}", max_len=63
        )
        curve_obj = _create_curve_object(curve_obj_name, gd.polys)
        _link_object_to_collection(curve_obj, clip_curve_col)
        created_curve_objects += 1

    # 2) 在 mask_polygon_collection 中：对每个 (tag, clip, mask_index) 只生成一个 mesh，include 已扣除 exclude
    merged: Dict[Tuple[str, str, int], Dict[str, List[List[Tuple[float, float, float]]]]] = {}
    for key, gd in groups.items():
        base = (key.tag, key.clip, key.mask_index)
        ent = merged.get(base)
        if ent is None:
            ent = {"include": [], "exclude": []}
            merged[base] = ent
        ent[key.kind].extend(gd.polys)

    # deterministic iteration
    for (tag, clip, mask_index) in sorted(merged.keys(), key=lambda t: (t[0], t[1], t[2])):
        include_polys = merged[(tag, clip, mask_index)].get("include") or []
        exclude_polys = merged[(tag, clip, mask_index)].get("exclude") or []
        if not include_polys:
            # 没有 include 就不生成最终面；exclude 单独存在没有意义
            continue

        tag_poly_col = _get_or_create_child_collection(root_poly, f"mask_polygon_{tag}")
        clip_poly_col = _get_or_create_child_collection(tag_poly_col, clip)

        # 先以 2D filled curve 构建 include-with-holes，再 convert 成 mesh
        mesh_obj_name = _sanitize_name(f"mask_polygon_{tag}_{clip}_{mask_index}", max_len=63)
        filled_curve_obj = _create_filled_curve_object_for_polygons(mesh_obj_name, include_polys, exclude_polys)
        _link_object_to_collection(filled_curve_obj, clip_poly_col)

        # convert 需要对象在 view layer 中且被选中/激活
        try:
            bpy.ops.object.select_all(action="DESELECT")
            filled_curve_obj.select_set(True)
            bpy.context.view_layer.objects.active = filled_curve_obj
            bpy.ops.object.convert(target="MESH")
        except Exception as e:
            print(
                f"[generate_polygons] curve->mesh convert 失败: {mesh_obj_name} err={e}，回退为简单 mesh（不扣洞）"
            )
            # 回退：至少生成 include 的 mesh（不扣洞）
            try:
                # 删除失败的 curve 对象，避免留垃圾
                bpy.data.objects.remove(filled_curve_obj, do_unlink=True)
            except Exception:
                pass
            mesh_obj = _create_mesh_object(mesh_obj_name, include_polys)
            _link_object_to_collection(mesh_obj, clip_poly_col)
            created_mesh_objects += 1
            continue

        # convert 后对象仍是同一个引用，但类型/数据变了；统一三角化，便于后续使用
        _triangulate_mesh_object_inplace(filled_curve_obj)
        created_mesh_objects += 1

    print(f"[generate_polygons] 创建完成: curves={created_curve_objects}, meshes={created_mesh_objects}")

    # save blend
    bpy.ops.wm.save_as_mainfile(filepath=output_file)
    print(f"[generate_polygons] 已保存: {output_file}")


def _enable_debugpy(port: int, wait_client: bool) -> None:
    try:
        import debugpy  # type: ignore
    except Exception as e:
        print(f"[debugpy] import 失败（请在 Blender Python 里安装 debugpy）: err={e}")
        return

    try:
        debugpy.listen(("127.0.0.1", int(port)))
        print(f"[debugpy] listen on 127.0.0.1:{port}")
        if wait_client:
            print("[debugpy] 等待客户端 attach...")
            debugpy.wait_for_client()
            print("[debugpy] 客户端已连接，继续执行。")
    except Exception as e:
        print(f"[debugpy] 启动失败: err={e}")


def _get_script_argv() -> List[str]:
    # Blender 会把脚本参数放在 "--" 之后
    argv = sys.argv
    if "--" in argv:
        return argv[argv.index("--") + 1 :]
    return []


def _parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--input", required=True, help="包含 *_blender.json 的目录（会递归扫描）")
    p.add_argument("--output", required=True, help="输出 .blend 文件路径")
    p.add_argument("--no-clean", action="store_true", help="不清空场景（默认会清空以保证可复现）")
    p.add_argument("--debugpy", action="store_true", help="启用 debugpy 远程断点调试")
    p.add_argument("--debugpy-port", type=int, default=5678, help="debugpy 监听端口")
    p.add_argument("--wait-client", action="store_true", help="启动后等待调试器连接")
    return p.parse_args(argv)


def _clean_scene() -> None:
    # 读入一个空场景，避免把当前 .blend 的内容混进输出
    try:
        bpy.ops.wm.read_factory_settings(use_empty=True)
    except Exception as e:
        print(f"[generate_polygons] 清空场景失败（继续执行）: err={e}")


if __name__ == "__main__":
    args = _parse_args(_get_script_argv())
    if args.debugpy:
        _enable_debugpy(args.debugpy_port, args.wait_client)

    if not args.no_clean:
        _clean_scene()

    generate_polygons_from_blender_clips(args.input, args.output)

r"""
后台批处理（推荐）：
blender.exe --background --python e:\sam3_track_seg\blender_scripts\blender_create_polygons.py -- --input e:\sam3_track_seg\test_images_shajing\blender_clips --output e:\sam3_track_seg\output\polygons.blend
断点调试（debugpy attach）：
blender.exe --background --python e:\sam3_track_seg\blender_scripts\blender_create_polygons.py -- --input e:\sam3_track_seg\test_images_shajing\blender_clips --output e:\sam3_track_seg\output\polygons.blend --debugpy --debugpy-port 5678 --wait-client
然后在 Cursor/VSCode 里用 Python “Attach” 连到 127.0.0.1:5678，断点打在 blender_create_polygons.py 里即可。
"""