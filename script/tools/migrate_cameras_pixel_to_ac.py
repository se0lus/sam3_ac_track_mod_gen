#!/usr/bin/env python3
"""Migrate cameras.ini from pixel-space coordinates to tileset_local AC coordinates.

When the root tileset has an identity transform but child tilesets share a common
root.transform, the camera_editor previously stored coordinates in pixel space
(origin = image top-left, 1 unit ≈ 1 pixel ≈ 1.07m). This script converts those
coordinates to tileset_local AC space (consistent with Blender / game engine).

Conversion chain (POSITION):
  pixel (x_ac, y_ac, z_ac) → ECEF via pixel affine
  ECEF → tileset_local via inverse tileset transform
  tileset_local → AC: new_x_ac = lx, new_y_ac = -lz, new_z_ac = ly

Conversion chain (FORWARD / UP direction vectors):
  pixel_dir → ECEF_dir via pixel affine rotation
  ECEF_dir → local_dir via inverse tileset rotation
  local_dir → AC_dir: new_dx = dlx, new_dy = -dlz, new_dz = dly, normalize

Usage:
  python -m script.tools.migrate_cameras_pixel_to_ac [--layout default] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys

# Add project root to path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
sys.path.insert(0, _REPO_ROOT)

from script.geo_sam3_blender_utils import (
    geodetic_to_ecef,
    _mat4_from_cesium_col_major,
    _invert_affine_4x4,
    _mat4_mul_point,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MANUAL_TRACK_INFO_DIR = os.path.join(_REPO_ROOT, "output", "11a_manual_track_info")
_02_RESULT_DIR = os.path.join(_REPO_ROOT, "output", "02_result")
_MASK_FULL_MAP_DIR = os.path.join(_REPO_ROOT, "output", "02_mask_full_map")
_CONFIG_JSON = os.path.join(_REPO_ROOT, "output", "webtools_config.json")


# ---------------------------------------------------------------------------
# INI parsing (mirrors run_webtools logic)
# ---------------------------------------------------------------------------

def _parse_cameras_ini(text: str) -> dict:
    header = {}
    cameras = []
    current_section = None
    current_cam = {}

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(";") or line.startswith("#"):
            continue
        m = re.match(r'^\[(.+)\]$', line)
        if m:
            section = m.group(1)
            if current_section and current_section.startswith("CAMERA_") and current_cam:
                cameras.append(current_cam)
                current_cam = {}
            current_section = section
            continue
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()
        if current_section == "HEADER":
            if key in ("VERSION", "CAMERA_COUNT"):
                header[key] = int(val)
            else:
                header[key] = val
        elif current_section and current_section.startswith("CAMERA_"):
            if "," in val:
                parts = [p.strip() for p in val.split(",")]
                try:
                    current_cam[key] = [float(p) for p in parts]
                except ValueError:
                    current_cam[key] = val
            else:
                try:
                    if "." in val or "e" in val or "E" in val:
                        current_cam[key] = float(val)
                    else:
                        current_cam[key] = int(val)
                except ValueError:
                    current_cam[key] = val

    if current_section and current_section.startswith("CAMERA_") and current_cam:
        cameras.append(current_cam)

    return {"header": header, "cameras": cameras}


def _fmt_cam_float(val: float) -> str:
    """Format a float for cameras.ini, avoiding scientific notation."""
    if val == int(val):
        return str(int(val))
    s = f"{val}"
    if "e" in s or "E" in s:
        s = f"{val:.10f}".rstrip("0").rstrip(".")
    return s


def _serialize_cameras_ini(data: dict) -> str:
    lines = []
    header = data.get("header", {})
    cameras = data.get("cameras", [])
    header["CAMERA_COUNT"] = len(cameras)

    lines.append("[HEADER]")
    lines.append(f"VERSION={header.get('VERSION', 3)}")
    lines.append(f"CAMERA_COUNT={header['CAMERA_COUNT']}")
    lines.append(f"SET_NAME={header.get('SET_NAME', 'TV1')}")
    lines.append("")

    field_order = [
        "NAME", "POSITION", "FORWARD", "UP",
        "MIN_FOV", "MAX_FOV", "IN_POINT", "OUT_POINT",
        "SHADOW_SPLIT0", "SHADOW_SPLIT1", "SHADOW_SPLIT2",
        "NEAR_PLANE", "FAR_PLANE", "MIN_EXPOSURE", "MAX_EXPOSURE",
        "DOF_FACTOR", "DOF_RANGE", "DOF_FOCUS", "DOF_MANUAL",
        "SPLINE", "SPLINE_ROTATION", "FOV_GAMMA",
        "SPLINE_ANIMATION_LENGTH", "IS_FIXED",
    ]
    for i, cam in enumerate(cameras):
        lines.append(f"[CAMERA_{i}]")
        written = set()
        for key in field_order:
            if key in cam:
                val = cam[key]
                if isinstance(val, list):
                    val_str = " ,".join(
                        f"{v:.3f}" if isinstance(v, float) else str(v) for v in val
                    )
                elif isinstance(val, float):
                    val_str = _fmt_cam_float(val)
                else:
                    val_str = str(val)
                lines.append(f"{key}={val_str}")
                written.add(key)
        for key, val in cam.items():
            if key not in written:
                if isinstance(val, list):
                    val_str = " ,".join(
                        f"{v:.3f}" if isinstance(v, float) else str(v) for v in val
                    )
                elif isinstance(val, float):
                    val_str = _fmt_cam_float(val)
                else:
                    val_str = str(val)
                lines.append(f"{key}={val_str}")
        lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_webtools_config() -> dict:
    if os.path.isfile(_CONFIG_JSON):
        with open(_CONFIG_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "tiles_dir": os.path.join(_REPO_ROOT, "test_images_shajing", "b3dm"),
    }


def _find_child_tileset_transform(tiles_dir: str):
    """Find first child tileset.json with a root.transform (16-element col-major)."""
    for root_dir, _dirs, files in os.walk(tiles_dir):
        for f in files:
            if f.lower() == "tileset.json":
                p = os.path.join(root_dir, f)
                try:
                    with open(p, "r", encoding="utf-8") as fh:
                        obj = json.load(fh)
                    tf = (obj.get("root") or {}).get("transform")
                    if isinstance(tf, list) and len(tf) == 16:
                        return [float(x) for x in tf]
                except Exception:
                    continue
    return None


def _find_geo_metadata() -> dict | None:
    """Find geo_metadata.json with corner coordinates."""
    for candidate in [
        os.path.join(_02_RESULT_DIR, "geo_metadata.json"),
        os.path.join(_MASK_FULL_MAP_DIR, "geo_metadata.json"),
    ]:
        if os.path.isfile(candidate):
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                continue
    return None


def _wgs84_up_vector(lon_deg: float, lat_deg: float):
    """WGS84 surface normal (up direction) at a given lon/lat."""
    lon = math.radians(lon_deg)
    lat = math.radians(lat_deg)
    cos_lat = math.cos(lat)
    return (
        cos_lat * math.cos(lon),
        cos_lat * math.sin(lon),
        math.sin(lat),
    )


def _vec_sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _vec_scale(v, s):
    return (v[0] * s, v[1] * s, v[2] * s)


def _vec_add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _vec_len(v):
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _vec_normalize(v):
    l = _vec_len(v)
    if l < 1e-12:
        return (0.0, 0.0, 0.0)
    return (v[0] / l, v[1] / l, v[2] / l)


def _mat3_mul_vec(m3, v):
    """Multiply 3x3 rotation (row-major list-of-lists) by vector."""
    return (
        m3[0][0] * v[0] + m3[0][1] * v[1] + m3[0][2] * v[2],
        m3[1][0] * v[0] + m3[1][1] * v[1] + m3[1][2] * v[2],
        m3[2][0] * v[0] + m3[2][1] * v[1] + m3[2][2] * v[2],
    )


def _extract_rotation_3x3(m4):
    """Extract 3x3 rotation from 4x4 row-major matrix."""
    return [
        [m4[0][0], m4[0][1], m4[0][2]],
        [m4[1][0], m4[1][1], m4[1][2]],
        [m4[2][0], m4[2][1], m4[2][2]],
    ]


# ---------------------------------------------------------------------------
# Build pixel→ECEF affine (mirrors camera_editor.js logic)
# ---------------------------------------------------------------------------

def _build_pixel_affine(geo_meta: dict):
    """Build pixel-to-ECEF rotation columns and origin.

    Returns (vecX, vecY, upVec, origin) where each is an (x,y,z) tuple.
    vecX = per-pixel step along image X axis (→ east)
    vecY = per-pixel step along image Y axis (↓ south)
    upVec = WGS84 surface normal at TL
    origin = TL corner ECEF
    """
    corners = geo_meta.get("corners", {})
    img_w = geo_meta.get("image_width", 1)
    img_h = geo_meta.get("image_height", 1)
    tl = corners.get("top_left")     # [lat, lon]
    tr = corners.get("top_right")    # [lat, lon]
    bl = corners.get("bottom_left")  # [lat, lon]
    if not (tl and tr and bl):
        raise ValueError("geo_metadata missing corner coordinates")

    tl_ecef = geodetic_to_ecef(tl[1], tl[0], 0)
    tr_ecef = geodetic_to_ecef(tr[1], tr[0], 0)
    bl_ecef = geodetic_to_ecef(bl[1], bl[0], 0)

    vecX = _vec_scale(_vec_sub(tr_ecef, tl_ecef), 1.0 / img_w)
    vecY = _vec_scale(_vec_sub(bl_ecef, tl_ecef), 1.0 / img_h)
    upVec = _wgs84_up_vector(tl[1], tl[0])

    return vecX, vecY, upVec, tl_ecef


def _pixel_ac_to_ecef(pos, vecX, vecY, upVec, origin):
    """Convert pixel AC coords (x_ac, y_ac, z_ac) to ECEF.

    ACToGeoTransform maps: x_b = x_ac, y_b = z_ac, z_b = -y_ac
    Then: ecef = origin + col0*x_b + col1*y_b + col2*z_b
    Where: col0=vecX, col1=vecY, col2=upVec
    """
    x_ac, y_ac, z_ac = pos
    x_b = x_ac
    y_b = z_ac
    z_b = -y_ac

    return _vec_add(
        origin,
        _vec_add(
            _vec_scale(vecX, x_b),
            _vec_add(_vec_scale(vecY, y_b), _vec_scale(upVec, z_b)),
        ),
    )


def _pixel_dir_to_ecef(d, vecX, vecY, upVec):
    """Convert pixel AC direction (dx, dy, dz) to ECEF direction (rotation only)."""
    dx_b = d[0]
    dy_b = d[2]
    dz_b = -d[1]

    return _vec_add(
        _vec_scale(vecX, dx_b),
        _vec_add(_vec_scale(vecY, dy_b), _vec_scale(upVec, dz_b)),
    )


# ---------------------------------------------------------------------------
# Main migration
# ---------------------------------------------------------------------------

def migrate(layout: str = "default", dry_run: bool = False):
    """Migrate cameras.ini from pixel space to tileset_local AC space."""
    # 1. Find child tileset transform
    cfg = _load_webtools_config()
    tiles_dir = cfg.get("tiles_dir", "")
    if not tiles_dir or not os.path.isdir(tiles_dir):
        print(f"ERROR: tiles_dir not found: {tiles_dir}")
        return False

    child_tf = _find_child_tileset_transform(tiles_dir)
    if not child_tf:
        print("ERROR: No child tileset with root.transform found")
        return False

    print(f"Child tileset transform found (first 4): {child_tf[:4]}")

    # 2. Find geo_metadata for pixel affine
    geo_meta = _find_geo_metadata()
    if not geo_meta:
        print("ERROR: geo_metadata.json not found")
        return False

    vecX, vecY, upVec, origin = _build_pixel_affine(geo_meta)
    print(f"Pixel affine origin (ECEF): ({origin[0]:.2f}, {origin[1]:.2f}, {origin[2]:.2f})")

    # 3. Build inverse tileset transform
    ts_mat = _mat4_from_cesium_col_major(child_tf)
    ts_inv = _invert_affine_4x4(ts_mat)
    ts_inv_rot = _extract_rotation_3x3(ts_inv)

    # 4. Read cameras.ini
    ini_path = os.path.join(_MANUAL_TRACK_INFO_DIR, layout, "cameras.ini")
    if not os.path.isfile(ini_path):
        print(f"ERROR: cameras.ini not found: {ini_path}")
        return False

    with open(ini_path, "r", encoding="utf-8") as f:
        ini_text = f.read()

    data = _parse_cameras_ini(ini_text)
    cameras = data.get("cameras", [])
    if not cameras:
        print("WARNING: No cameras found in ini file")
        return True

    print(f"\nMigrating {len(cameras)} cameras from pixel to tileset_local AC space...\n")

    # 5. Convert each camera
    for i, cam in enumerate(cameras):
        name = cam.get("NAME", f"CAMERA_{i}")

        # --- POSITION ---
        pos = cam.get("POSITION")
        if isinstance(pos, list) and len(pos) == 3:
            old_pos = list(pos)
            # pixel AC → ECEF
            ecef = _pixel_ac_to_ecef(pos, vecX, vecY, upVec, origin)
            # ECEF → tileset_local
            local = _mat4_mul_point(ts_inv, ecef)
            # tileset_local → AC: x_ac = lx, y_ac = -lz, z_ac = ly
            new_pos = [local[0], -local[2], local[1]]
            cam["POSITION"] = new_pos
            print(f"  [{name}] POSITION: ({old_pos[0]:.1f}, {old_pos[1]:.1f}, {old_pos[2]:.1f})"
                  f" → ({new_pos[0]:.1f}, {new_pos[1]:.1f}, {new_pos[2]:.1f})")

        # --- FORWARD ---
        fwd = cam.get("FORWARD")
        if isinstance(fwd, list) and len(fwd) == 3:
            old_fwd = list(fwd)
            # pixel dir → ECEF dir
            ecef_dir = _pixel_dir_to_ecef(fwd, vecX, vecY, upVec)
            # ECEF dir → local dir (rotation only)
            local_dir = _mat3_mul_vec(ts_inv_rot, ecef_dir)
            # local → AC dir, normalize
            ac_dir = _vec_normalize((local_dir[0], -local_dir[2], local_dir[1]))
            cam["FORWARD"] = list(ac_dir)
            print(f"  [{name}] FORWARD: ({old_fwd[0]:.3f}, {old_fwd[1]:.3f}, {old_fwd[2]:.3f})"
                  f" → ({ac_dir[0]:.3f}, {ac_dir[1]:.3f}, {ac_dir[2]:.3f})")

        # --- UP ---
        up = cam.get("UP")
        if isinstance(up, list) and len(up) == 3:
            old_up = list(up)
            ecef_dir = _pixel_dir_to_ecef(up, vecX, vecY, upVec)
            local_dir = _mat3_mul_vec(ts_inv_rot, ecef_dir)
            ac_dir = _vec_normalize((local_dir[0], -local_dir[2], local_dir[1]))
            cam["UP"] = list(ac_dir)
            print(f"  [{name}] UP: ({old_up[0]:.3f}, {old_up[1]:.3f}, {old_up[2]:.3f})"
                  f" → ({ac_dir[0]:.3f}, {ac_dir[1]:.3f}, {ac_dir[2]:.3f})")

    # 6. Write output
    if dry_run:
        print("\n[DRY RUN] No files written.")
        return True

    # Backup original
    backup_path = ini_path + ".pixel_backup"
    if not os.path.isfile(backup_path):
        shutil.copy2(ini_path, backup_path)
        print(f"\nBackup saved: {backup_path}")
    else:
        print(f"\nBackup already exists: {backup_path}")

    # Write migrated
    new_text = _serialize_cameras_ini(data)
    with open(ini_path, "w", encoding="utf-8") as f:
        f.write(new_text)
    print(f"Migrated cameras.ini written: {ini_path}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Migrate cameras.ini from pixel-space to tileset_local AC coordinates"
    )
    parser.add_argument("--layout", default="default", help="Layout name (default: 'default')")
    parser.add_argument("--dry-run", action="store_true", help="Print conversions without writing")
    args = parser.parse_args()

    ok = migrate(layout=args.layout, dry_run=args.dry_run)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
