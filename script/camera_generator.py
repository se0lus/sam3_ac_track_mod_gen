"""Camera generation module — generate AC cameras.ini from centerline.

Converts pixel-space centerline coordinates to AC world coordinates and
places cameras along the track with configurable count, height, FOV, and
side offset.

Coordinate chain:
  pixel (from centerline.json) → WGS84 (via geo_metadata bounds)
  → Blender XYZ (via tileset transform) → AC (axis negation)

Pure Python — no bpy dependency.
"""
from __future__ import annotations

import json
import logging
import math
import os
from typing import List, Optional, Tuple

logger = logging.getLogger("sam3_pipeline.camera_generator")


# ---------------------------------------------------------------------------
# Coordinate conversion: pixel → Blender → AC
# ---------------------------------------------------------------------------

def _find_dsm(tiles_dir: str) -> Optional[str]:
    """Auto-detect dsm.tif as a sibling of the tiles directory."""
    parent = os.path.dirname(os.path.abspath(tiles_dir))
    for candidate in (
        os.path.join(parent, "map", "dsm.tif"),
        os.path.join(parent, "dsm.tif"),
    ):
        if os.path.isfile(candidate):
            return candidate
    return None


def _sample_dsm_heights(
    geo_xy: list,
    dsm_path: str,
) -> Optional[List[float]]:
    """Sample terrain elevation from a DSM GeoTIFF at WGS84 lon/lat points.

    Returns list of elevation values (metres), or None on failure.
    """
    try:
        import rasterio
        from rasterio.warp import transform as warp_transform
    except ImportError:
        logger.warning("rasterio not available — cannot sample DSM elevation")
        return None

    try:
        with rasterio.open(dsm_path) as src:
            lons = [float(p[0]) for p in geo_xy]
            lats = [float(p[1]) for p in geo_xy]

            # Transform WGS84 → DSM CRS if needed
            dsm_crs = str(src.crs) if src.crs else ""
            if dsm_crs and dsm_crs != "EPSG:4326":
                xs, ys = warp_transform("EPSG:4326", src.crs, lons, lats)
            else:
                xs, ys = lons, lats

            coords = list(zip(xs, ys))
            elevations = [float(v[0]) for v in src.sample(coords)]
        logger.info("Sampled DSM elevation for %d points from %s", len(elevations), dsm_path)
        return elevations
    except Exception as exc:
        logger.warning("Failed to sample DSM: %s", exc)
        return None


def _load_centerline_ac(
    centerline_path: str,
    geo_metadata_path: str,
    tiles_dir: str,
    dsm_path: Optional[str] = None,
) -> Optional[List[Tuple[float, float, float]]]:
    """Load centerline.json pixel coords and convert to AC world coordinates.

    When *dsm_path* is provided (or auto-detected next to *tiles_dir*),
    terrain elevation is sampled from the DSM so that AC Y reflects the
    real model surface.

    Returns list of (ac_x, ac_y, ac_z) tuples, or None on failure.
    AC mapping from Blender: ac_x = -bx, ac_y = -by, ac_z = bz
    """
    from geo_sam3_blender_utils import get_tileset_transform, geo_points_to_blender_xyz

    if not os.path.isfile(centerline_path) or not os.path.isfile(geo_metadata_path):
        logger.warning("Missing centerline or geo_metadata: %s / %s",
                       centerline_path, geo_metadata_path)
        return None

    with open(centerline_path, "r", encoding="utf-8") as f:
        cl_data = json.load(f)
    with open(geo_metadata_path, "r", encoding="utf-8") as f:
        geo_meta = json.load(f)

    pixel_points = cl_data.get("points") or cl_data.get("centerline") or []
    if not pixel_points:
        logger.warning("No points in centerline.json")
        return None

    # Pixel → WGS84
    w = geo_meta["image_width"]
    h = geo_meta["image_height"]
    bounds = geo_meta["bounds"]
    geo_xy = []
    for pt in pixel_points:
        lon = bounds["west"] + float(pt[0]) * (bounds["east"] - bounds["west"]) / w
        lat = bounds["north"] - float(pt[1]) * (bounds["north"] - bounds["south"]) / h
        geo_xy.append([lon, lat])

    # Sample terrain elevation from DSM
    if not dsm_path:
        dsm_path = _find_dsm(tiles_dir)
    altitudes = _sample_dsm_heights(geo_xy, dsm_path) if dsm_path else None
    if altitudes:
        logger.info("Using DSM surface elevation for centerline Y")
    else:
        logger.info("No DSM available — centerline Y will be 0 (flat)")

    # WGS84 → Blender via tileset transform
    sample_geo = geo_xy[len(geo_xy) // 2] if geo_xy else None
    sample_tuple = tuple(sample_geo) if sample_geo else None
    tf_info = get_tileset_transform(tiles_dir, sample_geo_xy=sample_tuple)
    blender_pts = geo_points_to_blender_xyz(
        geo_xy, tf_info, z_mode="enu", altitudes=altitudes,
    )

    # Blender → AC: ac_x = -bx, ac_y = -by, ac_z = bz
    ac_pts = [(-float(p[0]), -float(p[1]), float(p[2])) for p in blender_pts]
    logger.info("Loaded centerline: %d points → AC coords", len(ac_pts))
    return ac_pts


# ---------------------------------------------------------------------------
# Camera placement algorithm
# ---------------------------------------------------------------------------

def _cumulative_lengths(points: List[Tuple[float, float, float]]) -> List[float]:
    """Compute cumulative arc-length in the XZ plane (AC horizontal)."""
    cum = [0.0]
    for i in range(1, len(points)):
        dx = points[i][0] - points[i - 1][0]
        dz = points[i][2] - points[i - 1][2]
        cum.append(cum[-1] + math.sqrt(dx * dx + dz * dz))
    return cum


def _local_curvature(points: List[Tuple[float, float, float]], idx: int,
                     window: int = 10) -> float:
    """Estimate curvature at index via angle change over a window.

    Returns curvature in radians / metre (higher = tighter bend).
    """
    n = len(points)
    if n < 3:
        return 0.0
    half = max(1, window // 2)
    i_prev = max(0, idx - half)
    i_next = min(n - 1, idx + half)
    if i_prev == i_next:
        return 0.0

    # Tangent at prev
    dx1 = points[min(i_prev + 1, n - 1)][0] - points[i_prev][0]
    dz1 = points[min(i_prev + 1, n - 1)][2] - points[i_prev][2]
    # Tangent at next
    dx2 = points[i_next][0] - points[max(i_next - 1, 0)][0]
    dz2 = points[i_next][2] - points[max(i_next - 1, 0)][2]

    len1 = math.sqrt(dx1 * dx1 + dz1 * dz1)
    len2 = math.sqrt(dx2 * dx2 + dz2 * dz2)
    if len1 < 1e-9 or len2 < 1e-9:
        return 0.0

    # Angle between tangents
    cos_a = max(-1.0, min(1.0, (dx1 * dx2 + dz1 * dz2) / (len1 * len2)))
    angle = math.acos(cos_a)

    # Arc length over the window
    arc = 0.0
    for j in range(i_prev, i_next):
        ddx = points[j + 1][0] - points[j][0]
        ddz = points[j + 1][2] - points[j][2]
        arc += math.sqrt(ddx * ddx + ddz * ddz)
    if arc < 1e-9:
        return 0.0
    return angle / arc


def generate_cameras_ini(
    centerline_path: str,
    geo_metadata_path: str,
    tiles_dir: str,
    num_cameras: int = 5,
    height_range: Tuple[float, float] = (6.0, 12.0),
    fov_range: Tuple[float, float] = (10.0, 60.0),
    side_offset: float = 15.0,
    dsm_path: Optional[str] = None,
) -> str:
    """Generate AC cameras.ini content from centerline with proper coordinate conversion.

    Args:
        centerline_path: Path to centerline.json (pixel-space coordinates).
        geo_metadata_path: Path to geo_metadata.json (image bounds).
        tiles_dir: Path to tileset directory for coordinate transform.
        num_cameras: Number of cameras to generate.
        height_range: (min_height, max_height) in metres above surface.
        fov_range: (min_fov, max_fov) — camera field of view range.
        side_offset: Perpendicular offset from track centreline in metres.
        dsm_path: Path to DSM GeoTIFF for terrain elevation sampling.
            Auto-detected from tiles_dir sibling if not provided.

    Returns:
        cameras.ini content string.
    """
    if num_cameras < 1:
        return "[HEADER]\nVERSION=3\nCAMERA_COUNT=0\nSET_NAME=TV1\n\n"

    ac_points = _load_centerline_ac(
        centerline_path, geo_metadata_path, tiles_dir, dsm_path=dsm_path,
    )
    if not ac_points or len(ac_points) < 2:
        logger.warning("Cannot generate cameras: insufficient centerline points")
        return "[HEADER]\nVERSION=3\nCAMERA_COUNT=0\nSET_NAME=TV1\n\n"

    if num_cameras > len(ac_points):
        num_cameras = len(ac_points)

    total_points = len(ac_points)
    step = total_points / num_cameras
    min_h, max_h = height_range
    min_fov, max_fov = fov_range

    lines = [
        "[HEADER]",
        "VERSION=3",
        f"CAMERA_COUNT={num_cameras}",
        "SET_NAME=TV1",
        "",
    ]

    for i in range(num_cameras):
        idx = int(i * step) % total_points
        px, py, pz = ac_points[idx]

        # Forward direction from current to a point ahead
        look_ahead = max(1, total_points // 50)
        next_idx = (idx + look_ahead) % total_points
        nx, ny, nz = ac_points[next_idx]
        dx, dz = nx - px, nz - pz
        length = math.sqrt(dx * dx + dz * dz)
        if length > 1e-9:
            dx /= length
            dz /= length
        else:
            dx, dz = 1.0, 0.0

        # Curvature-dependent height: tighter bends → higher camera
        curv = _local_curvature(ac_points, idx, window=max(10, total_points // 20))
        # Normalize curvature: 0 → min_h, high curvature → max_h
        curv_factor = min(1.0, curv * 50.0)
        height = min_h + curv_factor * (max_h - min_h)

        # Side offset: perpendicular to track direction in XZ plane
        # Perpendicular = (-dz, dx) in XZ
        cam_x = px + (-dz) * side_offset
        cam_z = pz + dx * side_offset
        # Camera Y: surface elevation py minus height (AC: negative Y = up)
        cam_y = py - height

        # Track coverage segments — evenly divide [0, 1] across cameras
        in_point = round(i / num_cameras, 3)
        out_point = round((i + 1) / num_cameras, 3)
        # Clamp to valid range
        in_point = min(in_point, 1.0)
        out_point = max(out_point, in_point)
        out_point = min(out_point, 1.0)

        lines.extend([
            f"[CAMERA_{i}]",
            f"NAME={i + 1}",
            f"POSITION={cam_x:.3f} ,{cam_y:.3f} ,{cam_z:.3f}",
            f"FORWARD={dx:.6f} ,-0.15 ,{dz:.6f}",
            "UP=0 ,1 ,0",
            f"MIN_FOV={min_fov:.0f}",
            f"MAX_FOV={max_fov:.0f}",
            f"IN_POINT={in_point}",
            f"OUT_POINT={out_point}",
            "SHADOW_SPLIT0=1.8",
            "SHADOW_SPLIT1=20",
            "SHADOW_SPLIT2=180",
            "NEAR_PLANE=0.1",
            "FAR_PLANE=5000",
            "MIN_EXPOSURE=0",
            "MAX_EXPOSURE=10000",
            "DOF_FACTOR=10",
            "DOF_RANGE=10000",
            "DOF_FOCUS=0",
            "DOF_MANUAL=0",
            "SPLINE=",
            "SPLINE_ROTATION=0",
            "FOV_GAMMA=0",
            "SPLINE_ANIMATION_LENGTH=15",
            "IS_FIXED=0",
            "",
        ])

    return "\n".join(lines)


def compute_track_bounds_ac(
    centerline_path: str,
    geo_metadata_path: str,
    tiles_dir: str,
    dsm_path: Optional[str] = None,
) -> Optional[Tuple[float, float, float, float]]:
    """Compute track bounding box in AC coordinates: (min_x, min_z, max_x, max_z).

    Used for map.ini generation.
    """
    ac_points = _load_centerline_ac(
        centerline_path, geo_metadata_path, tiles_dir, dsm_path=dsm_path,
    )
    if not ac_points:
        return None
    xs = [p[0] for p in ac_points]
    zs = [p[2] for p in ac_points]
    return (min(xs), min(zs), max(xs), max(zs))
