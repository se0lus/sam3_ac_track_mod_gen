"""
Surface extraction utilities for projecting 2D mask polygons onto 3D surfaces.

Pure Python module -- no bpy dependency. All Blender-specific logic lives in
``blender_scripts/sam3_actions/surface_extractor.py``.

Key functions
-------------
- ``generate_sampling_grid``  -- fill a polygon with a regular point grid
- ``triangulate_points``      -- Delaunay triangulation with boundary enforcement
- ``generate_collision_name`` -- Assetto Corsa naming convention
- ``load_clip_polygons``      -- read consolidated clip JSON into polygon lists
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import numpy as np  # type: ignore[import-untyped]
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


# ---------------------------------------------------------------------------
# Collision naming (TODO-6)
# ---------------------------------------------------------------------------

#: Recognised material types and their AC collision prefixes.
MATERIAL_PREFIXES: Dict[str, str] = {
    "wall": "1WALL",
    "road": "1ROAD",
    "road2": "2ROAD",
    "sand": "1SAND",
    "kerb": "1KERB",
    "grass": "1GRASS",
}

#: Each surface tag → its own collision collection name.
#: Each surface tag → its own collision collection name.
COLLISION_COLLECTION_MAP: Dict[str, str] = {
    "road": "collision_road",
    "road2": "collision_road2",
    "grass": "collision_grass",
    "sand": "collision_sand",
    "kerb": "collision_kerb",
    "wall": "collision_walls",
}


def generate_collision_name(material_type: str, index: int) -> str:
    """Return an Assetto Corsa collision object name.

    >>> generate_collision_name("road", 0)
    '1ROAD_0'
    >>> generate_collision_name("grass", 3)
    '1GRASS_3'
    """
    key = material_type.strip().lower()
    prefix = MATERIAL_PREFIXES.get(key)
    if prefix is None:
        raise ValueError(
            f"Unknown material type: {material_type!r}. "
            f"Expected one of {sorted(MATERIAL_PREFIXES.keys())}"
        )
    return f"{prefix}_{int(index)}"


# ---------------------------------------------------------------------------
# Point-in-polygon (ray casting)
# ---------------------------------------------------------------------------

def _point_in_polygon_2d(px: float, pz: float, polygon: List[Tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon test on the XZ plane."""
    n = len(polygon)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, zi = polygon[i]
        xj, zj = polygon[j]
        if ((zi > pz) != (zj > pz)) and (px < (xj - xi) * (pz - zi) / (zj - zi) + xi):
            inside = not inside
        j = i
    return inside


def _pip_numpy_batch(
    px: "np.ndarray", pz: "np.ndarray",
    polygon: List[Tuple[float, float]],
) -> "np.ndarray":
    """Vectorised ray-casting PIP on numpy arrays.  Returns bool array."""
    n = len(polygon)
    if n < 3:
        return np.zeros(len(px), dtype=bool)
    inside = np.zeros(len(px), dtype=bool)
    j = n - 1
    for i in range(n):
        xi, zi = float(polygon[i][0]), float(polygon[i][1])
        xj, zj = float(polygon[j][0]), float(polygon[j][1])
        dz = zj - zi
        if abs(dz) > 1e-12:
            cond = (zi > pz) != (zj > pz)
            x_cross = (xj - xi) * (pz - zi) / dz + xi
            inside ^= cond & (px < x_cross)
        j = i
    return inside


def _polygon_area(polygon: List[Tuple[float, float]]) -> float:
    """Absolute area via shoelace formula."""
    n = len(polygon)
    a = 0.0
    for i in range(n):
        j = (i + 1) % n
        a += polygon[i][0] * polygon[j][1]
        a -= polygon[j][0] * polygon[i][1]
    return abs(a) / 2.0


def classify_contours(
    contours: List[List[Tuple[float, float]]],
) -> List[Tuple[List[Tuple[float, float]], List[List[Tuple[float, float]]]]]:
    """Classify boundary contours into ``(outer, holes)`` groups.

    Returns a list of groups.  Each group is ``(outer_polygon, [hole_polygons])``.
    Disconnected regions that are not inside the main outer become separate
    groups with no holes.
    """
    if not contours:
        return []
    if len(contours) == 1:
        return [(contours[0], [])]

    # Sort by area, largest first
    with_area = [(c, _polygon_area(c)) for c in contours]
    with_area.sort(key=lambda x: -x[1])

    outer = with_area[0][0]
    holes: List[List[Tuple[float, float]]] = []
    extra: List[List[Tuple[float, float]]] = []

    for c, _area in with_area[1:]:
        # Centroid-based containment test
        cx = sum(p[0] for p in c) / len(c)
        cz = sum(p[1] for p in c) / len(c)
        if _point_in_polygon_2d(cx, cz, outer):
            holes.append(c)
        else:
            extra.append(c)

    groups: List[Tuple[List[Tuple[float, float]], List[List[Tuple[float, float]]]]] = [
        (outer, holes),
    ]
    for c in extra:
        groups.append((c, []))
    return groups


# ---------------------------------------------------------------------------
# Sampling grid generation
# ---------------------------------------------------------------------------

def generate_sampling_grid(
    polygon_xz: List[Tuple[float, float]],
    density: float,
    *,
    holes: Optional[List[List[Tuple[float, float]]]] = None,
    include_boundary: bool = True,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Tuple[List[Tuple[float, float]], List[int]]:
    """Generate a regular grid of sample points inside *polygon_xz*.

    Parameters
    ----------
    polygon_xz :
        Closed polygon vertices as ``(x, z)`` pairs (Blender XZ plane).
        The last point should **not** duplicate the first.
    density :
        Approximate spacing (in metres) between adjacent grid samples.
    holes :
        Optional list of hole polygons.  Grid points that fall inside any
        hole are excluded.  Hole boundary vertices are included when
        *include_boundary* is ``True``.
    include_boundary :
        When ``True`` (default), the polygon boundary vertices (and hole
        boundary vertices) are prepended to the output so that the final
        mesh edges align precisely with the mask outline.
    progress_callback :
        Optional callable receiving a status message string.  Used by the
        Blender operator to stream progress to the log.

    Returns
    -------
    points :
        All sample points as ``(x, z)`` tuples.  Boundary points come first
        when *include_boundary* is ``True``.
    boundary_indices :
        Indices (into *points*) that belong to the polygon boundary.
    """
    if density <= 0:
        raise ValueError(f"density must be positive, got {density}")
    if len(polygon_xz) < 3:
        raise ValueError(f"polygon must have >= 3 vertices, got {len(polygon_xz)}")

    points: List[Tuple[float, float]] = []
    boundary_indices: List[int] = []

    # 1) Boundary vertices (outer polygon + holes)
    if include_boundary:
        for pt in polygon_xz:
            boundary_indices.append(len(points))
            points.append((float(pt[0]), float(pt[1])))
        if holes:
            for hole in holes:
                for pt in hole:
                    boundary_indices.append(len(points))
                    points.append((float(pt[0]), float(pt[1])))

    # 2) Interior grid -- bounding box
    xs = [p[0] for p in polygon_xz]
    zs = [p[1] for p in polygon_xz]
    min_x, max_x = min(xs), max(xs)
    min_z, max_z = min(zs), max(zs)

    # De-duplication set (snap to grid tolerance)
    eps = density * 0.1
    existing: set = set()
    for pt in points:
        existing.add((_snap(pt[0], eps), _snap(pt[1], eps)))

    if _HAS_NUMPY:
        _generate_grid_numpy(
            polygon_xz, density, min_x, max_x, min_z, max_z,
            points, existing, eps, progress_callback, holes,
        )
    else:
        _generate_grid_python(
            polygon_xz, density, min_x, max_x, min_z, max_z,
            points, existing, eps, progress_callback, holes,
        )

    return points, boundary_indices


def _snap(v: float, eps: float) -> float:
    """Snap *v* to a grid with resolution *eps* for de-duplication."""
    return round(v / eps) * eps


# ---------------------------------------------------------------------------
# Numpy-accelerated grid generation
# ---------------------------------------------------------------------------

def _generate_grid_numpy(
    polygon_xz: List[Tuple[float, float]],
    density: float,
    min_x: float, max_x: float,
    min_z: float, max_z: float,
    points: List[Tuple[float, float]],
    existing: set,
    eps: float,
    cb: Optional[Callable[[str], None]],
    holes: Optional[List[List[Tuple[float, float]]]] = None,
) -> None:
    """Fill *points* with interior grid samples using numpy-vectorised PIP.

    Points inside *polygon_xz* but outside any *hole* are accepted.
    Memory is kept bounded by processing columns in chunks (~4 M points each).
    """
    x_coords = np.arange(min_x, max_x + 1e-9, density)
    z_coords = np.arange(min_z, max_z + 1e-9, density)
    nx, nz = len(x_coords), len(z_coords)
    total = nx * nz
    n_holes = len(holes) if holes else 0

    if cb:
        cb(f"  grid: {nx}x{nz} = {total:,} candidates "
           f"(numpy, {len(polygon_xz)} edges, {n_holes} holes)")

    # Process in column-chunks to limit memory (~32 MB per chunk)
    chunk_cols = max(1, 4_000_000 // max(nz, 1))
    n_inside_total = 0
    t0 = time.monotonic()

    for col_start in range(0, nx, chunk_cols):
        col_end = min(col_start + chunk_cols, nx)
        x_chunk = x_coords[col_start:col_end]

        gx, gz = np.meshgrid(x_chunk, z_coords, indexing="ij")
        gx_flat = gx.ravel()
        gz_flat = gz.ravel()

        # Inside outer polygon
        inside = _pip_numpy_batch(gx_flat, gz_flat, polygon_xz)

        # Exclude points inside any hole
        if holes:
            for hole in holes:
                in_hole = _pip_numpy_batch(gx_flat, gz_flat, hole)
                inside &= ~in_hole

        # Collect inside points (dedup against boundary)
        ix = gx_flat[inside]
        iz = gz_flat[inside]
        n_chunk_inside = len(ix)
        for x_val, z_val in zip(ix.tolist(), iz.tolist()):
            key = (_snap(x_val, eps), _snap(z_val, eps))
            if key not in existing:
                existing.add(key)
                points.append((x_val, z_val))

        n_inside_total += n_chunk_inside

        if cb and col_end < nx:
            pct = col_end * 100 // nx
            elapsed = time.monotonic() - t0
            cb(f"  grid: {pct}% ({col_end}/{nx} cols, "
               f"{n_inside_total:,} inside, {elapsed:.1f}s)")

    elapsed = time.monotonic() - t0
    if cb:
        cb(f"  grid done: {n_inside_total:,} inside / {total:,} total "
           f"({elapsed:.2f}s, numpy)")


# ---------------------------------------------------------------------------
# Pure-Python fallback grid generation
# ---------------------------------------------------------------------------

def _generate_grid_python(
    polygon_xz: List[Tuple[float, float]],
    density: float,
    min_x: float, max_x: float,
    min_z: float, max_z: float,
    points: List[Tuple[float, float]],
    existing: set,
    eps: float,
    cb: Optional[Callable[[str], None]],
    holes: Optional[List[List[Tuple[float, float]]]] = None,
) -> None:
    """Fill *points* with interior grid samples using pure-Python PIP.

    Fallback for environments without numpy.  Much slower for large grids
    but produces identical results.
    """
    xs_range: List[float] = []
    x = min_x
    while x <= max_x + 1e-9:
        xs_range.append(x)
        x += density

    zs_range: List[float] = []
    z = min_z
    while z <= max_z + 1e-9:
        zs_range.append(z)
        z += density

    total = len(xs_range) * len(zs_range)
    if cb:
        cb(f"  grid: {len(xs_range)}x{len(zs_range)} = {total:,} "
           f"candidates (pure Python)")

    t0 = time.monotonic()
    done = 0
    last_pct = -1

    for x_val in xs_range:
        for z_val in zs_range:
            done += 1
            if _point_in_polygon_2d(x_val, z_val, polygon_xz):
                # Exclude points inside holes
                if holes and any(
                    _point_in_polygon_2d(x_val, z_val, h) for h in holes
                ):
                    continue
                key = (_snap(x_val, eps), _snap(z_val, eps))
                if key not in existing:
                    existing.add(key)
                    points.append((x_val, z_val))
        # Progress per completed column
        pct = done * 100 // total
        if cb and pct >= last_pct + 5:
            last_pct = pct
            elapsed = time.monotonic() - t0
            if pct > 0:
                eta = elapsed / pct * (100 - pct)
                cb(f"  grid: {pct}% ({done:,}/{total:,}), "
                   f"elapsed {elapsed:.1f}s, ETA {eta:.1f}s")

    if cb:
        elapsed = time.monotonic() - t0
        cb(f"  grid done: {len(points):,} total points ({elapsed:.1f}s, "
           f"pure Python)")


# ---------------------------------------------------------------------------
# Triangulation
# ---------------------------------------------------------------------------

def triangulate_points(
    points_3d: List[Tuple[float, float, float]],
    boundary_indices: Optional[List[int]] = None,
) -> List[Tuple[int, int, int]]:
    """Delaunay triangulation projected onto XZ, returning face index triples.

    Uses ``scipy.spatial.Delaunay`` when available, otherwise falls back to a
    simple ear-clipping fan (only useful for convex-ish shapes).

    Parameters
    ----------
    points_3d :
        Vertices as ``(x, y, z)`` -- triangulation is done on ``(x, z)``.
    boundary_indices :
        Optional indices of boundary vertices (currently unused but reserved
        for constrained Delaunay in the future).

    Returns
    -------
    faces :
        List of ``(i, j, k)`` index triples referencing *points_3d*.
    """
    if len(points_3d) < 3:
        return []

    pts_2d = [(p[0], p[2]) for p in points_3d]

    try:
        from scipy.spatial import Delaunay  # type: ignore[import-untyped]
        try:
            tri = Delaunay(pts_2d)
            return [tuple(int(v) for v in simplex) for simplex in tri.simplices]  # type: ignore[misc]
        except Exception:
            # Degenerate input (collinear, duplicate points, etc.) -- fall through
            pass
    except ImportError:
        pass

    # Fallback: simple fan triangulation (centroid-based)
    cx = sum(p[0] for p in pts_2d) / len(pts_2d)
    cz = sum(p[1] for p in pts_2d) / len(pts_2d)

    # Sort by angle from centroid
    indexed = list(range(len(pts_2d)))
    indexed.sort(key=lambda i: math.atan2(pts_2d[i][1] - cz, pts_2d[i][0] - cx))

    faces: List[Tuple[int, int, int]] = []
    for i in range(len(indexed) - 2):
        faces.append((indexed[0], indexed[i + 1], indexed[i + 2]))
    return faces


# ---------------------------------------------------------------------------
# Triangle post-filtering
# ---------------------------------------------------------------------------

def filter_triangles_by_polygon(
    points_3d: List[Tuple[float, float, float]],
    faces: List[Tuple[int, int, int]],
    polygon_xz: List[Tuple[float, float]],
    holes: Optional[List[List[Tuple[float, float]]]] = None,
) -> List[Tuple[int, int, int]]:
    """Remove Delaunay triangles whose centroids are outside the polygon or
    inside any hole.

    After unconstrained Delaunay, some faces span across holes or extend
    beyond the polygon boundary.  This filter keeps only faces whose XZ
    centroid lies inside *polygon_xz* and outside all *holes*.
    """
    if not faces:
        return faces

    if _HAS_NUMPY:
        pts = np.array(points_3d)
        fa = np.array(faces)
        # Centroids in XZ (Blender: x=0, y=1, z=2)
        cx = (pts[fa[:, 0], 0] + pts[fa[:, 1], 0] + pts[fa[:, 2], 0]) / 3.0
        cz = (pts[fa[:, 0], 2] + pts[fa[:, 1], 2] + pts[fa[:, 2], 2]) / 3.0

        keep = _pip_numpy_batch(cx, cz, polygon_xz)
        if holes:
            for hole in holes:
                keep &= ~_pip_numpy_batch(cx, cz, hole)

        return [faces[i] for i in range(len(faces)) if keep[i]]

    # Pure-Python fallback
    filtered: List[Tuple[int, int, int]] = []
    for f in faces:
        cx = (points_3d[f[0]][0] + points_3d[f[1]][0] + points_3d[f[2]][0]) / 3.0
        cz = (points_3d[f[0]][2] + points_3d[f[1]][2] + points_3d[f[2]][2]) / 3.0
        if not _point_in_polygon_2d(cx, cz, polygon_xz):
            continue
        if holes and any(_point_in_polygon_2d(cx, cz, h) for h in holes):
            continue
        filtered.append(f)
    return filtered


# ---------------------------------------------------------------------------
# Clip JSON loading
# ---------------------------------------------------------------------------

def load_clip_polygons(clip_json_path: str) -> Dict[str, Any]:
    """Load a consolidated ``{tag}_clip.json`` and return parsed data.

    Returns
    -------
    dict with keys:
        ``tag``  -- material tag (str)
        ``include`` -- list of polygon dicts (each has ``points_xyz``)
        ``exclude`` -- list of polygon dicts
    """
    with open(clip_json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    tag = str(obj.get("source_tag", "unknown")).strip() or "unknown"
    polygons = obj.get("polygons") or {}
    include = polygons.get("include") or []
    exclude = polygons.get("exclude") or []

    return {
        "tag": tag,
        "include": include,
        "exclude": exclude,
    }


def extract_polygon_xz(poly_dict: Dict[str, Any]) -> List[Tuple[float, float]]:
    """Extract (x, z) pairs from a polygon dict's ``points_xyz`` field.

    ``points_xyz`` stores Blender coordinates ``[X, Y, Z]`` where the
    mask polygon lies in the XZ plane (Y is the "up" axis).
    """
    pts = poly_dict.get("points_xyz") or []
    out: List[Tuple[float, float]] = []
    for p in pts:
        if isinstance(p, (list, tuple)) and len(p) >= 3:
            out.append((float(p[0]), float(p[2])))
    return out
