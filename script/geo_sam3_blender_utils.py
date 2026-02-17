# 将 geo_sam3_image 中产生的 mask json 文件，根据原始数据的 tiles 信息，映射到 Blender 坐标系中。
# 设计目标：不依赖 bpy，在普通 Python 环境中输出可被 Blender 脚本消费的点集。

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Literal


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _iter_tileset_json_files(tiles_json_folder: str) -> Iterable[str]:
    # 优先顶层 tileset.json，其余递归兜底
    top = os.path.join(tiles_json_folder, "tileset.json")
    if os.path.exists(top):
        yield top
    for root, _, files in os.walk(tiles_json_folder):
        for name in files:
            if name.lower() == "tileset.json":
                p = os.path.join(root, name)
                if os.path.abspath(p) != os.path.abspath(top):
                    yield p


def _extract_origin_ecef_from_tileset(tileset: Dict[str, Any]) -> Optional[Tuple[float, float, float]]:
    """
    Cesium 3D Tiles 常见两种可用信息：
    - root.boundingVolume.box 的前三项通常是 box 的中心（在 ECEF 下会是百万级坐标）
    - root.transform（16项，列主序）中 indices 12..14 是平移（ECEF）
    """
    root = tileset.get("root") or {}
    bv = root.get("boundingVolume") or {}
    box = bv.get("box")
    if isinstance(box, list) and len(box) >= 3:
        try:
            x0, y0, z0 = float(box[0]), float(box[1]), float(box[2])
            if all(math.isfinite(v) for v in (x0, y0, z0)):
                return (x0, y0, z0)
        except Exception:
            pass

    tf = root.get("transform")
    if isinstance(tf, list) and len(tf) == 16:
        try:
            # Cesium 使用列主序矩阵，平移在 12..14
            x0, y0, z0 = float(tf[12]), float(tf[13]), float(tf[14])
            if all(math.isfinite(v) for v in (x0, y0, z0)):
                return (x0, y0, z0)
        except Exception:
            pass
    return None


def _get_tiles_origin_ecef(tiles_json_folder: str) -> Tuple[Tuple[float, float, float], str]:
    """
    返回 (origin_ecef, source_description)。
    """
    last_err: Optional[str] = None
    for p in _iter_tileset_json_files(tiles_json_folder):
        try:
            obj = _read_json(p)
            origin = _extract_origin_ecef_from_tileset(obj)
            if origin is not None:
                # 判断来源字段用于 debug
                root = obj.get("root") or {}
                bv = root.get("boundingVolume") or {}
                src = "root.boundingVolume.box" if isinstance(bv.get("box"), list) else "root.transform[12:15]"
                return origin, f"{os.path.relpath(p, tiles_json_folder)}::{src}"
        except Exception as e:
            last_err = f"{p}: {e}"
            continue
    raise RuntimeError(f"无法从 tiles_json_folder 提取 origin_ecef: {tiles_json_folder}. last_error={last_err}")


def _mat4_from_cesium_col_major(tf: List[float]) -> List[List[float]]:
    # Cesium tileset root.transform 是列主序 4x4
    if len(tf) != 16:
        raise ValueError(f"transform 长度必须为16，got={len(tf)}")
    m = [[0.0] * 4 for _ in range(4)]
    for c in range(4):
        for r in range(4):
            m[r][c] = float(tf[c * 4 + r])
    return m


def _mat4_mul_point(m: List[List[float]], p_xyz: Tuple[float, float, float]) -> Tuple[float, float, float]:
    x, y, z = p_xyz
    px = m[0][0] * x + m[0][1] * y + m[0][2] * z + m[0][3] * 1.0
    py = m[1][0] * x + m[1][1] * y + m[1][2] * z + m[1][3] * 1.0
    pz = m[2][0] * x + m[2][1] * y + m[2][2] * z + m[2][3] * 1.0
    pw = m[3][0] * x + m[3][1] * y + m[3][2] * z + m[3][3] * 1.0
    if abs(pw) > 1e-12:
        invw = 1.0 / pw
        return (px * invw, py * invw, pz * invw)
    return (px, py, pz)


def _is_orthonormal_3x3(r: List[List[float]], tol: float = 1e-6) -> bool:
    # 检查 R^T R ≈ I
    def dot(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    c0 = (r[0][0], r[1][0], r[2][0])
    c1 = (r[0][1], r[1][1], r[2][1])
    c2 = (r[0][2], r[1][2], r[2][2])
    d00 = dot(c0, c0)
    d11 = dot(c1, c1)
    d22 = dot(c2, c2)
    d01 = dot(c0, c1)
    d02 = dot(c0, c2)
    d12 = dot(c1, c2)
    return (
        abs(d00 - 1.0) < tol
        and abs(d11 - 1.0) < tol
        and abs(d22 - 1.0) < tol
        and abs(d01) < tol
        and abs(d02) < tol
        and abs(d12) < tol
    )


def _invert_affine_4x4(m: List[List[float]]) -> List[List[float]]:
    """
    仅用于 tileset 的刚体/近似刚体 transform：
    - 若 3x3 近似正交，则用 R^T 快速求逆
    - 否则尝试使用 numpy 求逆（若环境存在）
    """
    r = [[m[0][0], m[0][1], m[0][2]], [m[1][0], m[1][1], m[1][2]], [m[2][0], m[2][1], m[2][2]]]
    t = (m[0][3], m[1][3], m[2][3])

    if _is_orthonormal_3x3(r, tol=1e-5):
        # invR = R^T
        inv = [[0.0] * 4 for _ in range(4)]
        inv[0][0], inv[0][1], inv[0][2] = r[0][0], r[1][0], r[2][0]
        inv[1][0], inv[1][1], inv[1][2] = r[0][1], r[1][1], r[2][1]
        inv[2][0], inv[2][1], inv[2][2] = r[0][2], r[1][2], r[2][2]

        inv[3][3] = 1.0
        # invT = -invR * t
        inv[0][3] = -(inv[0][0] * t[0] + inv[0][1] * t[1] + inv[0][2] * t[2])
        inv[1][3] = -(inv[1][0] * t[0] + inv[1][1] * t[1] + inv[1][2] * t[2])
        inv[2][3] = -(inv[2][0] * t[0] + inv[2][1] * t[1] + inv[2][2] * t[2])
        return inv

    # fallback: numpy
    try:
        import numpy as np  # type: ignore

        a = np.array(m, dtype=float)
        inv_a = np.linalg.inv(a)
        return inv_a.tolist()
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"无法求逆 tileset transform（非正交且无 numpy 可用）: {e}")


def _invert_3x3(a: List[List[float]]) -> Optional[List[List[float]]]:
    # 3x3 逆矩阵（用于 OBB half-axes 矩阵求逆）
    a00, a01, a02 = a[0]
    a10, a11, a12 = a[1]
    a20, a21, a22 = a[2]
    det = (
        a00 * (a11 * a22 - a12 * a21)
        - a01 * (a10 * a22 - a12 * a20)
        + a02 * (a10 * a21 - a11 * a20)
    )
    if abs(det) < 1e-12:
        return None
    inv_det = 1.0 / det
    return [
        [(a11 * a22 - a12 * a21) * inv_det, (a02 * a21 - a01 * a22) * inv_det, (a01 * a12 - a02 * a11) * inv_det],
        [(a12 * a20 - a10 * a22) * inv_det, (a00 * a22 - a02 * a20) * inv_det, (a02 * a10 - a00 * a12) * inv_det],
        [(a10 * a21 - a11 * a20) * inv_det, (a01 * a20 - a00 * a21) * inv_det, (a00 * a11 - a01 * a10) * inv_det],
    ]


def _mat3_mul_vec(a: List[List[float]], v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (
        a[0][0] * v[0] + a[0][1] * v[1] + a[0][2] * v[2],
        a[1][0] * v[0] + a[1][1] * v[1] + a[1][2] * v[2],
        a[2][0] * v[0] + a[2][1] * v[1] + a[2][2] * v[2],
    )


def _point_in_bounding_box_box(p: Tuple[float, float, float], box: List[float], eps: float = 1e-6) -> bool:
    """
    3D Tiles 的 oriented bounding box: [cx,cy,cz, v0x,v0y,v0z, v1x,v1y,v1z, v2x,v2y,v2z]
    判断点 p 是否在盒内（允许少量误差）。
    """
    if len(box) < 12:
        return False
    c = (float(box[0]), float(box[1]), float(box[2]))
    v0 = (float(box[3]), float(box[4]), float(box[5]))
    v1 = (float(box[6]), float(box[7]), float(box[8]))
    v2 = (float(box[9]), float(box[10]), float(box[11]))
    d = (p[0] - c[0], p[1] - c[1], p[2] - c[2])
    # A 的列向量为 half-axes
    a = [
        [v0[0], v1[0], v2[0]],
        [v0[1], v1[1], v2[1]],
        [v0[2], v1[2], v2[2]],
    ]
    inv = _invert_3x3(a)
    if inv is None:
        return False
    s0, s1, s2 = _mat3_mul_vec(inv, d)
    return (abs(s0) <= 1.0 + eps) and (abs(s1) <= 1.0 + eps) and (abs(s2) <= 1.0 + eps)


def _pick_tileset_transform_for_mask(
    tiles_json_folder: str,
    sample_ecef: Tuple[float, float, float],
) -> Optional[Tuple[List[List[float]], str]]:
    """
    在 tiles_json_folder 下挑一个最合适的 tileset root.transform，作为 ECEF->local 的参考：
    - 优先选择“sample 点在该 tileset 的 root.boundingVolume.box 内”的 tileset
    - 若找不到包含关系，就返回第一个带 transform 的 tileset
    """
    first: Optional[Tuple[List[List[float]], str]] = None
    for p in _iter_tileset_json_files(tiles_json_folder):
        try:
            obj = _read_json(p)
        except Exception:
            continue
        root = obj.get("root") or {}
        tf = root.get("transform")
        if not (isinstance(tf, list) and len(tf) == 16):
            continue
        m = _mat4_from_cesium_col_major([float(x) for x in tf])
        inv = _invert_affine_4x4(m)
        local = _mat4_mul_point(inv, sample_ecef)
        if first is None:
            first = (inv, os.path.relpath(p, tiles_json_folder))
        bv = root.get("boundingVolume") or {}
        box = bv.get("box")
        if isinstance(box, list) and len(box) >= 12:
            try:
                if _point_in_bounding_box_box(local, [float(x) for x in box]):
                    return inv, os.path.relpath(p, tiles_json_folder)
            except Exception:
                pass
    return first


_WGS84_A = 6378137.0
_WGS84_F = 1.0 / 298.257223563
_WGS84_B = _WGS84_A * (1.0 - _WGS84_F)
_WGS84_E2 = _WGS84_F * (2.0 - _WGS84_F)  # first eccentricity squared
_WGS84_EP2 = (_WGS84_A * _WGS84_A - _WGS84_B * _WGS84_B) / (_WGS84_B * _WGS84_B)  # second eccentricity squared


def geodetic_to_ecef(lon_deg: float, lat_deg: float, h_m: float = 0.0) -> Tuple[float, float, float]:
    """
    WGS84: (lon, lat, h) -> ECEF (X,Y,Z), meters.
    lon/lat in degrees, height in meters.
    """
    lon = math.radians(float(lon_deg))
    lat = math.radians(float(lat_deg))
    h = float(h_m)

    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)

    n = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)
    x = (n + h) * cos_lat * cos_lon
    y = (n + h) * cos_lat * sin_lon
    z = (n * (1.0 - _WGS84_E2) + h) * sin_lat
    return (x, y, z)


def ecef_to_geodetic(x_m: float, y_m: float, z_m: float) -> Tuple[float, float, float]:
    """
    WGS84: ECEF (X,Y,Z) -> (lon,lat,h).
    Returns lon/lat in degrees, height in meters.

    使用 Bowring 近似（对绝大多数场景足够稳定/精确）。
    """
    x = float(x_m)
    y = float(y_m)
    z = float(z_m)

    lon = math.atan2(y, x)
    p = math.sqrt(x * x + y * y)
    # 防止极点处除零
    if p < 1e-12:
        lat = math.copysign(math.pi / 2.0, z)
        n = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * (math.sin(lat) ** 2))
        h = abs(z) - n * (1.0 - _WGS84_E2)
        return (math.degrees(lon), math.degrees(lat), h)

    theta = math.atan2(z * _WGS84_A, p * _WGS84_B)
    sin_t = math.sin(theta)
    cos_t = math.cos(theta)
    lat = math.atan2(
        z + _WGS84_EP2 * _WGS84_B * sin_t * sin_t * sin_t,
        p - _WGS84_E2 * _WGS84_A * cos_t * cos_t * cos_t,
    )

    sin_lat = math.sin(lat)
    n = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)
    h = p / max(1e-12, math.cos(lat)) - n
    return (math.degrees(lon), math.degrees(lat), h)


def _enu_basis(lon_rad: float, lat_rad: float) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]:
    """
    在 (lon0, lat0) 处构造 ENU 三个单位向量（ECEF 坐标系下表示）。
    """
    sin_lon = math.sin(lon_rad)
    cos_lon = math.cos(lon_rad)
    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)

    e = (-sin_lon, cos_lon, 0.0)
    n = (-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat)
    u = (cos_lat * cos_lon, cos_lat * sin_lon, sin_lat)
    return e, n, u


def _dot(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def ecef_to_enu(
    x_m: float,
    y_m: float,
    z_m: float,
    origin_ecef: Tuple[float, float, float],
    origin_lon_deg: float,
    origin_lat_deg: float,
) -> Tuple[float, float, float]:
    """
    将 ECEF 点转换到以 origin 为原点的 ENU 局部坐标（米）。
    """
    ox, oy, oz = origin_ecef
    d = (float(x_m) - float(ox), float(y_m) - float(oy), float(z_m) - float(oz))
    e, n, u = _enu_basis(math.radians(origin_lon_deg), math.radians(origin_lat_deg))
    return (_dot(e, d), _dot(n, d), _dot(u, d))


@dataclass
class TilesetTransformInfo:
    """Coordinate transform context from a 3D Tiles tileset.

    Encapsulates the ECEF origin, geodetic origin, and the inverse
    tileset transform needed to convert WGS84 coordinates to Blender-local
    coordinates.  Obtain via ``get_tileset_transform()``.
    """
    origin_ecef: Tuple[float, float, float]
    origin_lon: float
    origin_lat: float
    origin_h: float
    origin_src: str
    effective_mode: str  # "enu" or "tileset_local"
    inv_transform: Optional[List[List[float]]]  # 4x4 inverse transform matrix
    tf_source: Optional[str]  # tileset file relative path


def get_tileset_transform(
    tiles_json_folder: str,
    sample_geo_xy: Optional[Tuple[float, float]] = None,
    frame_mode: Literal["auto", "enu", "tileset_local"] = "auto",
) -> TilesetTransformInfo:
    """Get tileset coordinate transform info (one-time call).

    Extracts the ECEF origin from the tileset, optionally picks the best
    tileset transform using a sample geographic point, and returns all
    information needed to convert WGS84 coordinates to Blender-local coords.

    Args:
        tiles_json_folder: Directory with tileset.json file(s).
        sample_geo_xy: Optional ``(lon, lat)`` to help select the correct
            tileset when there are multiple.
        frame_mode: ``"auto"`` (prefer tileset_local, fallback enu),
            ``"enu"`` (force ENU), or ``"tileset_local"`` (force tileset).
    """
    origin_ecef, origin_src = _get_tiles_origin_ecef(tiles_json_folder)
    origin_lon, origin_lat, origin_h = ecef_to_geodetic(*origin_ecef)

    sample_ecef: Optional[Tuple[float, float, float]] = None
    if sample_geo_xy is not None:
        sample_ecef = geodetic_to_ecef(sample_geo_xy[0], sample_geo_xy[1], 0.0)

    chosen_inv_tf: Optional[List[List[float]]] = None
    chosen_tf_src: Optional[str] = None
    if frame_mode in ("auto", "tileset_local") and sample_ecef is not None:
        picked = _pick_tileset_transform_for_mask(tiles_json_folder, sample_ecef)
        if picked is not None:
            chosen_inv_tf, chosen_tf_src = picked

    effective_mode: str
    if frame_mode == "enu":
        effective_mode = "enu"
    elif frame_mode == "tileset_local":
        if chosen_inv_tf is None:
            raise RuntimeError(
                "frame_mode='tileset_local' but no tileset.json with root.transform found"
            )
        effective_mode = "tileset_local"
    else:
        effective_mode = "tileset_local" if chosen_inv_tf is not None else "enu"

    return TilesetTransformInfo(
        origin_ecef=origin_ecef,
        origin_lon=origin_lon,
        origin_lat=origin_lat,
        origin_h=origin_h,
        origin_src=origin_src,
        effective_mode=effective_mode,
        inv_transform=chosen_inv_tf,
        tf_source=chosen_tf_src,
    )


def geo_points_to_blender_xyz(
    geo_xy: List[List[float]],
    tf_info: TilesetTransformInfo,
    z_mode: Literal["zero", "enu", "const"] = "zero",
    z_value: Optional[float] = None,
) -> List[List[float]]:
    """Convert WGS84 coordinate list to Blender XYZ coordinates.

    Uses the same axis remapping as ``map_mask_to_blender``:

    - tileset_local: ``(blender_X, blender_Y, blender_Z) = (local_x, local_z, local_y)``
    - enu: ``(blender_X, blender_Y, blender_Z) = (E, U, N)``

    Args:
        geo_xy: List of ``[lon, lat]`` pairs in WGS84.
        tf_info: ``TilesetTransformInfo`` from ``get_tileset_transform()``.
        z_mode: ``"zero"`` (force 0), ``"enu"`` (real height), ``"const"`` (fixed).
        z_value: Height value when ``z_mode="const"``.

    Returns:
        List of ``[bx, by, bz]`` in Blender coordinate space.
    """
    points_xyz: List[List[float]] = []
    for pt in geo_xy:
        if len(pt) < 2:
            continue
        lon, lat = float(pt[0]), float(pt[1])
        x, y, z = geodetic_to_ecef(lon, lat, 0.0)

        if tf_info.effective_mode == "tileset_local" and tf_info.inv_transform is not None:
            lx, ly, lz = _mat4_mul_point(tf_info.inv_transform, (x, y, z))
            if z_mode == "zero":
                lz_out = 0.0
            elif z_mode == "const":
                lz_out = float(z_value)  # type: ignore[arg-type]
            else:
                lz_out = float(lz)
            # Axis remap: Blender (X,Y,Z) = (local_x, local_z, local_y)
            points_xyz.append([float(lx), float(lz_out), float(ly)])
        else:
            e, n, u = ecef_to_enu(
                x, y, z, tf_info.origin_ecef, tf_info.origin_lon, tf_info.origin_lat
            )
            if z_mode == "zero":
                u_out = 0.0
            elif z_mode == "const":
                u_out = float(z_value)  # type: ignore[arg-type]
            else:
                u_out = float(u)
            # Axis remap: Blender (X,Y,Z) = (E, U, N)
            points_xyz.append([float(e), float(u_out), float(n)])

    return points_xyz


def map_mask_to_blender(
    mask_json_file: str,
    tiles_json_folder: str,
    *,
    z_mode: Literal["zero", "enu", "const"] = "zero",
    z_value: Optional[float] = None,
    frame_mode: Literal["auto", "enu", "tileset_local"] = "auto",
) -> Dict[str, Any]:
    """
    将 geo_sam3_image 产生的 mask json（包含 polygons.geo_xy）映射到 Blender 使用的局部 ENU 坐标系中。

    - 不依赖 bpy
    - 输出：一组 include 多边形与一组 exclude 多边形（每个多边形为 3D 点集），并带 tag/prob/index 信息

    输出格式（返回 Dict[str, Any]）：

    - **origin**: 参考原点信息（用于 debug/对齐）
      - `ecef`: `[x,y,z]`（米）
      - `lonlat`: `[lon,lat]`（度）
      - `h`: 高程（米）
      - `source`: origin 来源说明（例如 `tileset.json::root.boundingVolume.box`）
    - **frame**: 实际使用的坐标系
      - `mode`: `"tileset_local"` 或 `"enu"`
      - `tileset_transform_source`: 选中的 tileset 文件相对路径（仅 tileset_local 时有意义）
    - **polygons**:
      - `include`: `List[poly]`
      - `exclude`: `List[poly]`
      - 其中 `poly` 结构为：
        - `kind`: `"include"` / `"exclude"`
        - `mask_index`: mask 序号（来自 *_masks.json）
        - `poly_index`: 多边形序号（在该 mask 的 include/exclude 内的索引）
        - `tag`: 标签（如 "road"）
        - `prob`: 置信度
        - `points_xyz`: `[[x,y,z], ...]`（Blender 局部坐标，单位米）

          注意：当前项目中 **本方法内部的平面坐标 (x,y)** 对应到 Blender 的 **(X,Z)**，
          也就是 `BlenderZ = local_y`，因此实际输出会做一次轴重排：

          - `BlenderX = local_x`
          - `BlenderY = local_z`（由 z_mode 决定：0/const/enu）
          - `BlenderZ = local_y`
        - `geo_xy`: `[[lon,lat], ...]`（原始经纬度，便于核对）

    Args:
        mask_json_file: geo_sam3_image 导出的 *_masks.json
        tiles_json_folder: 3D Tiles 根目录（包含 tileset.json）
        z_mode:
            - "zero": 强制输出 z=0（更适合在 Blender 中再投射/贴地）
            - "enu": 使用 ENU 的 U 作为 z（由椭球面 h=0 推导，通常接近 0）
            - "const": 强制使用 z_value 作为 z（手动指定一个常量高度）
        z_value: 当 z_mode="const" 时生效，指定输出 z 的常量值（米）
        frame_mode:
            - "auto": 优先使用 tileset_local（与 Blender 导入 tileset 的局部坐标对齐），失败则回退 enu
            - "enu": 强制使用 ENU 局部坐标
            - "tileset_local": 强制使用 tileset root.transform 的逆变换得到局部坐标
    """
    mask_obj = _read_json(mask_json_file)
    meta = mask_obj.get("meta") or {}
    geo_meta = meta.get("geo") or {}
    crs = geo_meta.get("crs")
    if crs not in (None, "EPSG:4326"):
        # 当前实现只明确支持经纬度（WGS84）。如果后续需要投影坐标，可在这里扩展。
        raise ValueError(f"暂不支持的 CRS: {crs}. 需要 EPSG:4326")

    origin_ecef, origin_src = _get_tiles_origin_ecef(tiles_json_folder)
    origin_lon, origin_lat, origin_h = ecef_to_geodetic(*origin_ecef)

    if z_mode not in ("zero", "enu", "const"):
        raise ValueError(f"z_mode 必须是 'zero'/'enu'/'const'，当前={z_mode!r}")
    if z_mode == "const":
        if z_value is None or not math.isfinite(float(z_value)):
            raise ValueError("当 z_mode='const' 时，必须提供有限的 z_value（米）")
    if frame_mode not in ("auto", "enu", "tileset_local"):
        raise ValueError(f"frame_mode 必须是 'auto'/'enu'/'tileset_local'，当前={frame_mode!r}")

    # 选一个 sample 点用于挑选正确的 tileset transform（用 meta bounds 的左上角）
    bounds = (geo_meta.get("bounds") or {}) if isinstance(geo_meta, dict) else {}
    left_val = bounds.get("left") if isinstance(bounds, dict) else None
    top_val = bounds.get("top") if isinstance(bounds, dict) else None
    sample_lon = float(left_val) if left_val is not None else None
    sample_lat = float(top_val) if top_val is not None else None
    sample_ecef: Optional[Tuple[float, float, float]] = None
    if sample_lon is not None and sample_lat is not None:
        sample_ecef = geodetic_to_ecef(sample_lon, sample_lat, 0.0)

    chosen_inv_tf: Optional[List[List[float]]] = None
    chosen_tf_src: Optional[str] = None
    if frame_mode in ("auto", "tileset_local") and sample_ecef is not None:
        picked = _pick_tileset_transform_for_mask(tiles_json_folder, sample_ecef)
        if picked is not None:
            chosen_inv_tf, chosen_tf_src = picked

    effective_mode: Literal["enu", "tileset_local"]
    if frame_mode == "enu":
        effective_mode = "enu"
    elif frame_mode == "tileset_local":
        if chosen_inv_tf is None:
            raise RuntimeError("frame_mode='tileset_local' 但未找到任何带 root.transform 的 tileset.json")
        effective_mode = "tileset_local"
    else:
        # auto：优先 tileset_local，否则回退到 ENU
        effective_mode = "tileset_local" if chosen_inv_tf is not None else "enu"

    include_out: List[Dict[str, Any]] = []
    exclude_out: List[Dict[str, Any]] = []

    masks = mask_obj.get("masks") or []
    if not isinstance(masks, list):
        raise ValueError(f"mask json 格式错误: masks 应为 list, got={type(masks)}")

    for m in masks:
        if not isinstance(m, dict):
            continue
        mask_index = int(m.get("index", -1))
        tag = m.get("tag", None)
        prob = float(m.get("prob", 0.0))
        polys = (m.get("polygons") or {})
        inc_list = polys.get("include") or []
        exc_list = polys.get("exclude") or []

        def _convert_poly_list(poly_list: Any, out_list: List[Dict[str, Any]], kind: str) -> None:
            if not isinstance(poly_list, list):
                return
            for poly_index, poly in enumerate(poly_list):
                if not isinstance(poly, dict):
                    continue
                geo_xy = poly.get("geo_xy") or []
                if not isinstance(geo_xy, list) or len(geo_xy) == 0:
                    continue

                points_xyz: List[List[float]] = []
                for pt in geo_xy:
                    if not isinstance(pt, list) or len(pt) < 2:
                        continue
                    lon = float(pt[0])
                    lat = float(pt[1])
                    x, y, z = geodetic_to_ecef(lon, lat, 0.0)
                    if effective_mode == "tileset_local" and chosen_inv_tf is not None:
                        lx, ly, lz = _mat4_mul_point(chosen_inv_tf, (x, y, z))
                        if z_mode == "zero":
                            lz_out = 0.0
                        elif z_mode == "const":
                            lz_out = float(z_value)  # type: ignore[arg-type]
                        else:
                            lz_out = float(lz)
                        # 轴重排：Blender (X,Y,Z) = (local_x, local_z, local_y)
                        points_xyz.append([float(lx), float(lz_out), float(ly)])
                    else:
                        e, n, u = ecef_to_enu(x, y, z, origin_ecef, origin_lon, origin_lat)
                        if z_mode == "zero":
                            u_out = 0.0
                        elif z_mode == "const":
                            u_out = float(z_value)  # type: ignore[arg-type]
                        else:
                            u_out = float(u)
                        # 轴重排：Blender (X,Y,Z) = (E, U, N)
                        points_xyz.append([float(e), float(u_out), float(n)])

                if not points_xyz:
                    continue

                out_list.append(
                    {
                        "kind": kind,
                        "mask_index": mask_index,
                        "poly_index": int(poly_index),
                        "tag": tag,
                        "prob": prob,
                        "points_xyz": points_xyz,
                        # 可选：保留原始 geo 便于核对
                        "geo_xy": geo_xy,
                    }
                )

        _convert_poly_list(inc_list, include_out, "include")
        _convert_poly_list(exc_list, exclude_out, "exclude")

    return {
        "origin": {
            "ecef": [float(origin_ecef[0]), float(origin_ecef[1]), float(origin_ecef[2])],
            "lonlat": [float(origin_lon), float(origin_lat)],
            "h": float(origin_h),
            "source": origin_src,
        },
        "frame": {
            "mode": effective_mode,
            "tileset_transform_source": chosen_tf_src,
        },
        "source_mask_json": os.path.abspath(mask_json_file),
        "source_tiles_folder": os.path.abspath(tiles_json_folder),
        "polygons": {
            "include": include_out,
            "exclude": exclude_out,
        },
    }


def consolidate_clips_by_tag(blender_clips_folder: str, output_folder: Optional[str] = None) -> Dict[str, str]:
    """
    Merge all per-clip ``*_blender.json`` files into consolidated per-tag files.

    For every unique tag found across the blender clip files, produces a single
    consolidated file named ``{tag}_clip.json`` that contains **all** polygons
    (both include and exclude) for that material type.  The consolidated file
    uses the same ``*_blender.json`` schema so it can be consumed by
    ``blender_create_polygons.py`` without changes.

    Args:
        blender_clips_folder: Root folder containing ``*_blender.json`` files
            (searched recursively).
        output_folder: Where to write the consolidated files.  Defaults to
            *blender_clips_folder* itself when ``None``.

    Returns:
        A dict mapping ``tag`` -> absolute path of the written consolidated file.
    """
    if output_folder is None:
        output_folder = blender_clips_folder
    os.makedirs(output_folder, exist_ok=True)

    # Collect all blender json files
    blender_json_files: List[str] = []
    for root, _, files in os.walk(blender_clips_folder):
        for name in files:
            if name.lower().endswith("_blender.json"):
                blender_json_files.append(os.path.join(root, name))

    if not blender_json_files:
        print(f"[consolidate_clips_by_tag] No *_blender.json found in {blender_clips_folder}")
        return {}

    # tag -> {"include": [...], "exclude": [...]}
    tag_polygons: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    # Keep the first origin/frame seen per tag for metadata
    tag_meta: Dict[str, Dict[str, Any]] = {}

    for jp in sorted(blender_json_files):
        try:
            with open(jp, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception as e:
            print(f"[consolidate_clips_by_tag] Failed to read {jp}: {e}")
            continue

        if not isinstance(obj, dict):
            continue

        polygons = obj.get("polygons") or {}
        if not isinstance(polygons, dict):
            continue

        for kind in ("include", "exclude"):
            items = polygons.get(kind) or []
            if not isinstance(items, list):
                continue
            for poly in items:
                if not isinstance(poly, dict):
                    continue
                tag = str(poly.get("tag") or "unknown").strip()
                if not tag:
                    tag = "unknown"

                if tag not in tag_polygons:
                    tag_polygons[tag] = {"include": [], "exclude": []}
                tag_polygons[tag][kind].append(poly)

                if tag not in tag_meta:
                    tag_meta[tag] = {
                        "origin": obj.get("origin"),
                        "frame": obj.get("frame"),
                    }

    # Write one consolidated file per tag
    written: Dict[str, str] = {}
    for tag in sorted(tag_polygons.keys()):
        consolidated = {
            "origin": (tag_meta.get(tag) or {}).get("origin"),
            "frame": (tag_meta.get(tag) or {}).get("frame"),
            "source_tag": tag,
            "polygons": tag_polygons[tag],
        }
        out_path = os.path.join(output_folder, f"{tag}_clip.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(consolidated, f, indent=2, ensure_ascii=False)
        inc_cnt = len(tag_polygons[tag].get("include") or [])
        exc_cnt = len(tag_polygons[tag].get("exclude") or [])
        print(f"[consolidate_clips_by_tag] {tag}: include={inc_cnt}, exclude={exc_cnt} -> {out_path}")
        written[tag] = os.path.abspath(out_path)

    return written


if __name__ == "__main__":
    tiles_folder = "E:\\sam3_track_seg\\test_images_shajing\\b3dm"
    mask_json_file = "E:\\sam3_track_seg\\test_images_shajing\\clips\\road\\clip_18_masks.json"

    # 先验证能否读到 tiles 原点（ECEF）
    origin, src = _get_tiles_origin_ecef(tiles_folder)
    print(f"tiles origin (ECEF) = {origin} from {src}")

    # 再跑一次映射，输出简单统计
    out = map_mask_to_blender(mask_json_file, tiles_folder, z_mode="zero", frame_mode="auto")
    inc_cnt = len(out.get("polygons", {}).get("include", []) or [])
    exc_cnt = len(out.get("polygons", {}).get("exclude", []) or [])
    print(f"mapped polygons: include={inc_cnt}, exclude={exc_cnt}")
    print(f"frame: {out.get('frame')}")

    # 简单可视化：在 Blender 的 X-Z 平面画出多边形轮廓（需要 matplotlib）
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ModuleNotFoundError:
        plt = None  # type: ignore

    if plt is None:
        print("matplotlib 未安装，跳过可视化。你可以：pip install matplotlib")
    else:
        fig, ax = plt.subplots(figsize=(8, 8))

        def _plot_polys(polys: List[Dict[str, Any]], color: str, label: str) -> None:
            shown = False
            for poly in polys:
                pts = poly.get("points_xyz") or []
                if not isinstance(pts, list) or len(pts) < 2:
                    continue
                # points_xyz 已是 Blender 坐标 (X,Y,Z)，这里画 X-Z 平面
                xs = [float(p[0]) for p in pts if isinstance(p, list) and len(p) >= 3]
                zs = [float(p[2]) for p in pts if isinstance(p, list) and len(p) >= 3]
                if len(xs) < 2 or len(zs) < 2:
                    continue
                # 闭合
                xs2 = xs + [xs[0]]
                zs2 = zs + [zs[0]]
                ax.plot(xs2, zs2, color=color, linewidth=1.0, alpha=0.9, label=(label if not shown else None))
                shown = True

        include_polys = (out.get("polygons") or {}).get("include") or []
        exclude_polys = (out.get("polygons") or {}).get("exclude") or []
        _plot_polys(include_polys, color="tab:blue", label="include")
        _plot_polys(exclude_polys, color="tab:red", label="exclude")

        ax.set_title(f"Mapped polygons in Blender X-Z plane (mode={out.get('frame', {}).get('mode')})")
        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Z (meters)")
        ax.axis("equal")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
        ax.legend()
        plt.tight_layout()
        plt.show()