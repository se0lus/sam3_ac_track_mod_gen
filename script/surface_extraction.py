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
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Collision naming (TODO-6)
# ---------------------------------------------------------------------------

#: Recognised material types and their AC collision prefixes.
MATERIAL_PREFIXES: Dict[str, str] = {
    "wall": "1WALL",
    "road": "1ROAD",
    "sand": "1SAND",
    "kerb": "1KERB",
    "grass": "1GRASS",
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


# ---------------------------------------------------------------------------
# Sampling grid generation
# ---------------------------------------------------------------------------

def generate_sampling_grid(
    polygon_xz: List[Tuple[float, float]],
    density: float,
    *,
    include_boundary: bool = True,
) -> Tuple[List[Tuple[float, float]], List[int]]:
    """Generate a regular grid of sample points inside *polygon_xz*.

    Parameters
    ----------
    polygon_xz :
        Closed polygon vertices as ``(x, z)`` pairs (Blender XZ plane).
        The last point should **not** duplicate the first.
    density :
        Approximate spacing (in metres) between adjacent grid samples.
    include_boundary :
        When ``True`` (default), the polygon boundary vertices are prepended
        to the output so that the final mesh edges align precisely with the
        mask outline.

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

    # 1) Boundary vertices
    if include_boundary:
        for pt in polygon_xz:
            boundary_indices.append(len(points))
            points.append((float(pt[0]), float(pt[1])))

    # 2) Interior grid
    xs = [p[0] for p in polygon_xz]
    zs = [p[1] for p in polygon_xz]
    min_x, max_x = min(xs), max(xs)
    min_z, max_z = min(zs), max(zs)

    # Build a set of existing points for de-duplication (snap to grid tolerance)
    eps = density * 0.1
    existing = set()
    for pt in points:
        existing.add((_snap(pt[0], eps), _snap(pt[1], eps)))

    x = min_x
    while x <= max_x + 1e-9:
        z = min_z
        while z <= max_z + 1e-9:
            if _point_in_polygon_2d(x, z, polygon_xz):
                key = (_snap(x, eps), _snap(z, eps))
                if key not in existing:
                    existing.add(key)
                    points.append((x, z))
            z += density
        x += density

    return points, boundary_indices


def _snap(v: float, eps: float) -> float:
    """Snap *v* to a grid with resolution *eps* for de-duplication."""
    return round(v / eps) * eps


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
