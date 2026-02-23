"""
Blender headless export script — Stage 10.

Cleans, splits, renames, batches, and exports FBX files from final_track.blend.

Usage::

    blender --background --python blender_export.py -- \\
        --blend-input output/09_result/final_track.blend \\
        --output-dir output/10_model_export \\
        --tiles-dir test_images_shajing/b3dm \\
        --max-vertices 21000 \\
        --max-batch-mb 100 \\
        --fbx-scale 0.01

Steps:
  1. Cleanup — remove mask collections + other non-game data
  2. Split oversized meshes (road via centerline, others via optimal XZ)
  3. Rename collision objects with AC conventions
  4. Organise into export batches (auto-detect tile levels from scene)
  5. Final cleanup — remove everything NOT in an export batch
  6. Save intermediate .blend + export FBX per batch (no embedded textures)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# sys.path setup
# ---------------------------------------------------------------------------
_this_dir = os.path.dirname(os.path.realpath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

_script_dir = os.path.join(os.path.dirname(_this_dir), "script")
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

import bpy  # type: ignore[import-not-found]
import bmesh  # type: ignore[import-not-found]
from mathutils import Vector  # type: ignore[import-not-found]


def _emit_progress(pct, msg=""):
    """Emit structured progress line for webtools dashboard."""
    print(f"@@PROGRESS@@ {max(0,min(100,int(pct)))} {msg}".rstrip(), flush=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("blender_export")


# ===================================================================
# Constants
# ===================================================================
from surface_extraction import (
    MATERIAL_PREFIXES,
    COLLISION_COLLECTION_MAP,
    generate_collision_name,
)

# Reverse map: collection_name -> material_tag
_COLLECTION_TO_TAG: Dict[str, str] = {v: k for k, v in COLLISION_COLLECTION_MAP.items()}

# KN5 format uses 16-bit vertex indices → hard limit of 65535 per mesh.
# Game engines typically don't share vertices across triangles, so the worst-
# case KN5 vertex count = triangle_count × 3 (no vertex reuse at all).
_KN5_MAX_VERTICES = 65535

# Mask collection names to delete
_MASK_COLLECTIONS = ["mask_polygon_collection", "mask_curve2D_collection"]

# Known game-relevant collection names (besides L{N} tile collections)
_GAME_COLLECTIONS: Set[str] = {
    "collision", "collision_road", "collision_kerb", "collision_grass",
    "collision_sand", "collision_road2", "collision_walls", "game_objects",
}


# ===================================================================
# Helpers
# ===================================================================

def _read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_collection(name: str) -> Optional[bpy.types.Collection]:
    """Find a collection by name anywhere in the scene hierarchy."""
    def _search(col: bpy.types.Collection) -> Optional[bpy.types.Collection]:
        if col.name == name:
            return col
        for child in col.children:
            found = _search(child)
            if found is not None:
                return found
        return None
    return _search(bpy.context.scene.collection)


def _delete_collection_recursive(col: bpy.types.Collection) -> int:
    """Delete a collection and all its children/objects. Returns count deleted."""
    count = 0
    for child in list(col.children):
        count += _delete_collection_recursive(child)
    for obj in list(col.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
        count += 1
    bpy.data.collections.remove(col)
    return count


def _ensure_collection(name: str) -> bpy.types.Collection:
    """Get or create a root-level collection."""
    col = _find_collection(name)
    if col is not None:
        return col
    col = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(col)
    return col


def _get_object_collections(obj: bpy.types.Object) -> List[bpy.types.Collection]:
    """Return all collections an object belongs to."""
    result = []
    for col in bpy.data.collections:
        if obj.name in col.objects:
            result.append(col)
    return result


def _mesh_vertex_count(obj: bpy.types.Object) -> int:
    if obj.type != "MESH" or obj.data is None:
        return 0
    return len(obj.data.vertices)


def _mesh_kn5_vertex_estimate(obj: bpy.types.Object) -> int:
    """Worst-case KN5 vertex count: ``triangles × 3`` (no vertex sharing).

    Game engines / KN5 may treat every triangle corner as a unique vertex
    (no index reuse).  Blender quads become 2 triangles on export, so for
    each polygon with *N* vertices we get ``(N − 2)`` triangles × 3 corners.
    This is always ≥ ``len(obj.data.loops)``.
    """
    if obj.type != "MESH" or obj.data is None:
        return 0
    mesh = obj.data
    tri_count = sum(p.loop_total - 2 for p in mesh.polygons)
    return tri_count * 3


def _detect_tile_levels() -> Tuple[List[int], Optional[int]]:
    """Auto-detect L{N} tile collections in the scene.

    Returns (sorted list of all levels, base_level or None).
    Base level is the lowest detected level.
    """
    levels: List[int] = []
    for col in bpy.data.collections:
        m = re.match(r"^L(\d+)$", col.name)
        if m and len(list(col.objects)) > 0:
            levels.append(int(m.group(1)))
    levels.sort()
    base = levels[0] if levels else None
    return levels, base


# ===================================================================
# Step 1: Cleanup — delete mask collections and non-game data
# ===================================================================

def step1_cleanup() -> None:
    """Remove mask collections and other non-game helper data."""
    log.info("Step 1/6: Cleanup — removing mask collections")
    total = 0
    for name in _MASK_COLLECTIONS:
        col = _find_collection(name)
        if col is not None:
            n = _delete_collection_recursive(col)
            log.info("  Deleted '%s' (%d objects)", name, n)
            total += n
        else:
            log.info("  '%s' not found, skipping", name)

    bpy.ops.outliner.orphans_purge(do_recursive=True)
    log.info("  Cleanup done (%d objects removed, orphans purged)", total)


# ===================================================================
# Step 2: Mesh splitting
# ===================================================================

def _bisect_mesh_object(
    obj: bpy.types.Object,
    plane_co: Tuple[float, float, float],
    plane_no: Tuple[float, float, float],
) -> Tuple[Optional[bpy.types.Object], Optional[bpy.types.Object]]:
    """Bisect a mesh object into two halves along a plane.

    Returns (inner_obj, outer_obj). Either may be None if one side is empty.
    """
    collections = _get_object_collections(obj)
    if not collections:
        collections = [bpy.context.scene.collection]

    outer_obj = obj.copy()
    outer_obj.data = obj.data.copy()
    outer_obj.name = obj.name + "_split"
    for col in collections:
        col.objects.link(outer_obj)

    # Inner side (clear_outer=True)
    bm_inner = bmesh.new()
    bm_inner.from_mesh(obj.data)
    geom = bm_inner.verts[:] + bm_inner.edges[:] + bm_inner.faces[:]
    bmesh.ops.bisect_plane(
        bm_inner, geom=geom, dist=0.0001,
        plane_co=Vector(plane_co),
        plane_no=Vector(plane_no),
        clear_outer=True, clear_inner=False,
    )
    bm_inner.to_mesh(obj.data)
    bm_inner.free()
    obj.data.update()

    # Outer side (clear_inner=True)
    bm_outer = bmesh.new()
    bm_outer.from_mesh(outer_obj.data)
    geom = bm_outer.verts[:] + bm_outer.edges[:] + bm_outer.faces[:]
    bmesh.ops.bisect_plane(
        bm_outer, geom=geom, dist=0.0001,
        plane_co=Vector(plane_co),
        plane_no=Vector(plane_no),
        clear_outer=False, clear_inner=True,
    )
    bm_outer.to_mesh(outer_obj.data)
    bm_outer.free()
    outer_obj.data.update()

    inner_verts = len(obj.data.vertices)
    outer_verts = len(outer_obj.data.vertices)

    inner_result = obj if inner_verts > 0 else None
    outer_result = outer_obj if outer_verts > 0 else None

    if inner_verts == 0:
        bpy.data.objects.remove(obj, do_unlink=True)
    if outer_verts == 0:
        bpy.data.objects.remove(outer_obj, do_unlink=True)

    return inner_result, outer_result


def _find_optimal_split(
    obj: bpy.types.Object,
    max_vertices: int,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Find optimal bisection plane via vertex distribution analysis.

    Picks the axis (X or Z) with greater spread, cuts at the median
    vertex position to equalise both sides and minimise recursion.
    """
    mesh = obj.data
    mat = obj.matrix_world
    n_verts = len(mesh.vertices)

    xs = [0.0] * n_verts
    zs = [0.0] * n_verts
    for i, v in enumerate(mesh.vertices):
        co = mat @ v.co
        xs[i] = co.x
        zs[i] = co.z

    xs.sort()
    zs.sort()

    if (xs[-1] - xs[0]) >= (zs[-1] - zs[0]):
        cut_val = xs[n_verts // 2]
        return (cut_val, 0.0, 0.0), (1.0, 0.0, 0.0)
    else:
        cut_val = zs[n_verts // 2]
        return (0.0, 0.0, cut_val), (0.0, 0.0, 1.0)


def _split_recursive(
    obj: bpy.types.Object,
    max_vertices: int,
    depth: int = 0,
) -> List[bpy.types.Object]:
    """Recursively split until all pieces <= max_vertices."""
    verts = _mesh_vertex_count(obj)
    if verts <= max_vertices:
        return [obj]

    if depth >= 10:
        log.warning("  Max recursion depth for '%s' (%d verts)", obj.name, verts)
        return [obj]

    plane_co, plane_no = _find_optimal_split(obj, max_vertices)
    inner, outer = _bisect_mesh_object(obj, plane_co, plane_no)

    results = []
    if inner is not None:
        results.extend(_split_recursive(inner, max_vertices, depth + 1))
    if outer is not None:
        results.extend(_split_recursive(outer, max_vertices, depth + 1))
    return results if results else [obj]


def _load_centerline_blender(
    centerline_json: str,
    geo_metadata_path: str,
    tiles_dir: str,
) -> Optional[List[Tuple[float, float, float]]]:
    """Load centerline.json and convert pixel coords to Blender XYZ."""
    try:
        from geo_sam3_blender_utils import get_tileset_transform, geo_points_to_blender_xyz
    except ImportError:
        log.warning("geo_sam3_blender_utils not available, cannot load centerline")
        return None

    if not os.path.isfile(centerline_json) or not os.path.isfile(geo_metadata_path):
        return None

    try:
        cl_data = _read_json(centerline_json)
        geo_meta = _read_json(geo_metadata_path)
    except Exception as e:
        log.warning("Failed to read centerline/metadata: %s", e)
        return None

    pixel_points = cl_data.get("points") or cl_data.get("centerline") or []
    if not pixel_points:
        log.warning("No points in centerline.json")
        return None

    w = geo_meta["image_width"]
    h = geo_meta["image_height"]
    bounds = geo_meta["bounds"]
    geo_xy = []
    for pt in pixel_points:
        lon = bounds["west"] + float(pt[0]) * (bounds["east"] - bounds["west"]) / w
        lat = bounds["north"] - float(pt[1]) * (bounds["north"] - bounds["south"]) / h
        geo_xy.append([lon, lat])

    sample_geo = geo_xy[len(geo_xy) // 2] if geo_xy else None
    sample_tuple = tuple(sample_geo) if sample_geo else None
    tf_info = get_tileset_transform(tiles_dir, sample_geo_xy=sample_tuple)
    blender_pts = geo_points_to_blender_xyz(geo_xy, tf_info, z_mode="zero")

    result = [(float(p[0]), float(p[1]), float(p[2])) for p in blender_pts]
    log.info("  Loaded centerline: %d points -> Blender coords", len(result))
    return result


def _centerline_cumulative_lengths(
    centerline: List[Tuple[float, float, float]],
) -> List[float]:
    """Compute cumulative arc-length along the centerline (XZ plane)."""
    cum = [0.0]
    for i in range(1, len(centerline)):
        dx = centerline[i][0] - centerline[i - 1][0]
        dz = centerline[i][2] - centerline[i - 1][2]
        cum.append(cum[-1] + math.sqrt(dx * dx + dz * dz))
    return cum


def _project_faces_to_arc_length(
    obj: bpy.types.Object,
    centerline: List[Tuple[float, float, float]],
    cum_lengths: List[float],
) -> List[float]:
    """Project each polygon's centroid onto the centerline via nearest-point.

    For each face, finds the closest centerline point in the XZ plane and
    returns its arc-length.  This follows the road shape naturally and never
    produces infinite cut planes that could cross a bend.

    Processes in batches of 5000 faces for memory efficiency.
    """
    import numpy as np

    mat = obj.matrix_world
    mesh = obj.data
    n_cl = len(centerline)
    n_faces = len(mesh.polygons)

    # Centerline XZ as numpy array
    cl_xz = np.array([[centerline[i][0], centerline[i][2]] for i in range(n_cl)])
    cum_arr = np.array(cum_lengths)

    # Collect face centroids in world space
    centroids_xz = np.empty((n_faces, 2), dtype=np.float64)
    for fi, poly in enumerate(mesh.polygons):
        c = mat @ poly.center
        centroids_xz[fi, 0] = c.x
        centroids_xz[fi, 1] = c.z

    # Batched nearest-centerline-point lookup
    face_arcs = np.empty(n_faces, dtype=np.float64)
    batch_size = 5000
    for start in range(0, n_faces, batch_size):
        end = min(start + batch_size, n_faces)
        batch = centroids_xz[start:end]  # (B, 2)
        # Distance to each centerline point: (B, n_cl)
        diffs = batch[:, np.newaxis, :] - cl_xz[np.newaxis, :, :]
        dists_sq = np.sum(diffs ** 2, axis=2)
        nearest_idx = np.argmin(dists_sq, axis=1)
        face_arcs[start:end] = cum_arr[nearest_idx]

    return face_arcs.tolist()


def _split_road_by_arc_proximity(
    obj: bpy.types.Object,
    centerline: List[Tuple[float, float, float]],
    max_vertices: int,
) -> List[bpy.types.Object]:
    """Split a road mesh into segments by centerline arc-length proximity.

    Instead of using infinite bisect planes (which cross bends), this
    approach:

    1. Projects every face centroid onto the nearest centerline point
       to get an arc-length parameter.
    2. Sorts faces by arc-length and partitions them into N segments
       of roughly equal face count.
    3. Creates a separate mesh for each segment by deleting the faces
       belonging to other segments.

    Each face is assigned to exactly one segment → no gaps, no overlaps,
    and the splits follow the road shape through curves.
    """
    verts = _mesh_vertex_count(obj)
    n_segments = math.ceil(verts / max_vertices)
    if n_segments <= 1:
        return [obj]

    n_pts = len(centerline)
    if n_pts < 2:
        return _split_recursive(obj, max_vertices)

    log.info("  Road '%s': %d verts, splitting into %d arc-length segments",
             obj.name, verts, n_segments)

    cum_lengths = _centerline_cumulative_lengths(centerline)
    face_arcs = _project_faces_to_arc_length(obj, centerline, cum_lengths)

    n_faces = len(face_arcs)

    # Find arc-length thresholds that divide faces into roughly equal groups
    sorted_arcs = sorted(face_arcs)
    thresholds: List[float] = []
    for i in range(1, n_segments):
        idx = int(n_faces * i / n_segments)
        thresholds.append(sorted_arcs[min(idx, n_faces - 1)])

    # Assign each face to a segment
    def _get_seg(arc: float) -> int:
        s = 0
        for t in thresholds:
            if arc > t:
                s += 1
        return s

    face_segment = [_get_seg(a) for a in face_arcs]

    # Create one object per segment
    collections = _get_object_collections(obj)
    if not collections:
        collections = [bpy.context.scene.collection]

    results: List[bpy.types.Object] = []
    original_name = obj.name

    for seg_idx in range(n_segments):
        keep_faces: Set[int] = {fi for fi, s in enumerate(face_segment) if s == seg_idx}
        if not keep_faces:
            continue

        new_obj = obj.copy()
        new_obj.data = obj.data.copy()
        new_obj.name = f"{original_name}_s{seg_idx}"
        for col in collections:
            col.objects.link(new_obj)

        bm = bmesh.new()
        bm.from_mesh(new_obj.data)
        bm.faces.ensure_lookup_table()

        # Delete faces NOT in this segment
        to_del = [f for f in bm.faces if f.index not in keep_faces]
        if to_del:
            bmesh.ops.delete(bm, geom=to_del, context="FACES")

        # Remove orphaned vertices (no connected faces)
        isolated = [v for v in bm.verts if not v.link_faces]
        if isolated:
            bmesh.ops.delete(bm, geom=isolated, context="VERTS")

        bm.to_mesh(new_obj.data)
        bm.free()
        new_obj.data.update()

        seg_verts = len(new_obj.data.vertices)
        if seg_verts > 0:
            results.append(new_obj)
            log.info("    segment %d: %d faces, %d verts",
                     seg_idx, len(new_obj.data.polygons), seg_verts)
        else:
            bpy.data.objects.remove(new_obj, do_unlink=True)

    # Remove original object
    bpy.data.objects.remove(obj, do_unlink=True)

    # Any segment still exceeding max_vertices → recursive XZ fallback
    final: List[bpy.types.Object] = []
    for piece in results:
        if _mesh_vertex_count(piece) > max_vertices:
            log.info("    '%s' still has %d verts, applying XZ fallback",
                     piece.name, _mesh_vertex_count(piece))
            final.extend(_split_recursive(piece, max_vertices))
        else:
            final.append(piece)

    return final


def step2_split_meshes(
    max_vertices: int,
    centerline: Optional[List[Tuple[float, float, float]]] = None,
) -> None:
    """Split all oversized MESH objects.

    Checks both the Blender vertex count (against *max_vertices*) and the
    worst-case KN5 vertex count (against ``_KN5_MAX_VERTICES``, 65535).

    KN5 / game engines may treat every triangle corner as a unique vertex
    (no index reuse), so the effective vertex count = ``triangles × 3``.
    Blender quads fan-out into 2 triangles on FBX export, further inflating
    the count.  A mesh that looks fine in Blender may still exceed the KN5
    limit after triangulation and vertex expansion.
    """
    log.info("Step 2/6: Mesh splitting (max %d verts, KN5 limit %d tri-verts)",
             max_vertices, _KN5_MAX_VERTICES)

    to_split: List[Tuple[bpy.types.Object, bool, int]] = []
    for obj in list(bpy.data.objects):
        if obj.type != "MESH":
            continue
        verts = _mesh_vertex_count(obj)
        kn5_verts = _mesh_kn5_vertex_estimate(obj)
        exceeds_user = verts > max_vertices
        exceeds_kn5 = kn5_verts > _KN5_MAX_VERTICES
        if not exceeds_user and not exceeds_kn5:
            continue
        is_road = any(c.name == "collision_road" for c in _get_object_collections(obj))
        to_split.append((obj, is_road, kn5_verts))

    if not to_split:
        log.info("  No meshes exceed limits, skipping")
        return

    log.info("  Found %d meshes to split", len(to_split))

    total_pieces = 0
    for obj, is_road, kn5_verts in to_split:
        verts_before = _mesh_vertex_count(obj)
        name = obj.name

        # Compute effective max_vertices that also satisfies the KN5 limit.
        # KN5 vertex estimate = triangles × 3 (no sharing).
        # inflation = kn5_verts / blender_verts → how many KN5 verts per Blender vert.
        # effective_max = KN5_MAX / inflation
        effective_max = max_vertices
        if kn5_verts > _KN5_MAX_VERTICES:
            inflation = kn5_verts / max(verts_before, 1)
            kn5_safe = int(_KN5_MAX_VERTICES / inflation)
            # Extra 10% safety margin to account for splitting imprecision
            kn5_safe = int(kn5_safe * 0.9)
            effective_max = min(max_vertices, max(kn5_safe, 100))
            log.info("  '%s': %d blender-verts, %d kn5-verts (%.1fx inflation) -> "
                     "effective max %d blender-verts for KN5 safety",
                     name, verts_before, kn5_verts, inflation, effective_max)

        if is_road and centerline is not None and len(centerline) >= 2:
            pieces = _split_road_by_arc_proximity(obj, centerline, effective_max)
        else:
            pieces = _split_recursive(obj, effective_max)

        total_pieces += len(pieces)
        verts_after = sum(_mesh_vertex_count(p) for p in pieces)
        kn5_after = sum(_mesh_kn5_vertex_estimate(p) for p in pieces)
        log.info("  '%s': %d verts/%d kn5-verts -> %d pieces (%d verts, %d kn5-verts)",
                 name, verts_before, kn5_verts, len(pieces), verts_after, kn5_after)

        # Verify no piece exceeds KN5 limit
        for p in pieces:
            p_kn5 = _mesh_kn5_vertex_estimate(p)
            if p_kn5 > _KN5_MAX_VERTICES:
                log.warning("  WARNING: '%s' has %d kn5-verts (> %d KN5 limit)!",
                            p.name, p_kn5, _KN5_MAX_VERTICES)

    log.info("  Split complete: %d objects -> %d pieces", len(to_split), total_pieces)


# ===================================================================
# Step 3: Collision object renaming
# ===================================================================

def step3_rename_collision() -> None:
    """Rename collision objects to Assetto Corsa convention.

    Uses a two-pass approach to avoid Blender's automatic ``.001`` suffix
    when the target name is already taken by another object in the scene.
    """
    log.info("Step 3/6: Renaming collision objects")

    renamed = 0
    for tag, col_name in COLLISION_COLLECTION_MAP.items():
        col = _find_collection(col_name)
        if col is None:
            continue

        mesh_objs = sorted(
            [obj for obj in col.objects if obj.type == "MESH"],
            key=lambda o: o.name,
        )

        # Pass 1: rename to temporary names to free all target names
        for idx, obj in enumerate(mesh_objs):
            tmp = f"__tmp_{tag}_{idx}"
            obj.name = tmp
            if obj.data:
                obj.data.name = tmp

        # Pass 2: rename to final collision names (no conflicts now)
        for idx, obj in enumerate(mesh_objs):
            new_name = generate_collision_name(tag, idx)
            log.info("  %s -> %s", obj.name, new_name)
            obj.name = new_name
            if obj.data:
                obj.data.name = new_name
            renamed += 1

    log.info("  Renamed %d collision objects", renamed)


# ===================================================================
# Step 4: Batch organisation (auto-detect tile levels)
# ===================================================================

def _estimate_geometry_bytes(obj: bpy.types.Object) -> int:
    """Estimate FBX geometry size (no textures).  ~40 B/vert + ~20 B/face."""
    if obj.type != "MESH" or obj.data is None:
        return 0
    return len(obj.data.vertices) * 40 + len(obj.data.polygons) * 20


def step4_batch_organise(max_batch_mb: int) -> List[str]:
    """Organise objects into export batch collections.

    Auto-detects tile levels from ``L{N}`` collections in the scene.

    Priority groups:
      1. Road collision (track_core)
      2. Game objects — per-layout (``go_{LayoutName}``) or single (``game_objects``)
      3. Other collision (kerb, grass, sand, road2, walls)
      4. Fine terrain tiles — label ``L{min}-L{max}`` (e.g. ``L18-L19``)
      5. Base terrain tiles — label ``L{N}`` (e.g. ``L17``)
      6. Custom user objects — anything not collected above (label ``custom``)
    """
    log.info("Step 4/6: Batch organisation (max %d MB per batch)", max_batch_mb)

    max_batch_bytes = max_batch_mb * 1024 * 1024

    # Auto-detect tile levels
    tile_levels, base_level = _detect_tile_levels()
    if tile_levels:
        log.info("  Auto-detected tile levels: %s (base=%d)", tile_levels, base_level)
    else:
        log.info("  No tile level collections found")

    groups: List[Tuple[str, List[bpy.types.Object]]] = []

    # Group 1: Road collision only (no game objects)
    g1: List[bpy.types.Object] = []
    road_col = _find_collection("collision_road")
    if road_col:
        g1.extend(o for o in road_col.objects if o.type == "MESH")
    if g1:
        groups.append(("track_core", g1))

    # Game objects — per-layout or single
    go_root = _find_collection("game_objects")
    if go_root:
        layout_cols = [c for c in go_root.children
                       if c.name.startswith("game_objects_")]
        if layout_cols:
            # Multi-layout: one group per layout
            for lc in sorted(layout_cols, key=lambda c: c.name):
                layout_name = lc.name.replace("game_objects_", "", 1)
                objs = list(lc.all_objects)
                if objs:
                    groups.append((f"go_{layout_name}", objs))
                    log.info("  Game objects layout '%s': %d objects",
                             layout_name, len(objs))
        else:
            # Single layout: all game objects as one group
            objs = list(go_root.objects)
            if objs:
                groups.append(("game_objects", objs))

    # Group 2: Other collision
    g2: List[bpy.types.Object] = []
    for tag, col_name in COLLISION_COLLECTION_MAP.items():
        if tag == "road":
            continue
        col = _find_collection(col_name)
        if col:
            g2.extend(o for o in col.objects if o.type == "MESH")
    if g2:
        groups.append(("collision_other", g2))

    # Group 3: Fine terrain tiles (everything above base level)
    fine_levels = [lv for lv in tile_levels if lv != base_level] if base_level is not None else []
    g3: List[bpy.types.Object] = []
    for lv in fine_levels:
        col = _find_collection(f"L{lv}")
        if col:
            g3.extend(o for o in col.objects if o.type == "MESH")
    if g3:
        fine_label = f"L{min(fine_levels)}-L{max(fine_levels)}" if len(fine_levels) > 1 else f"L{fine_levels[0]}"
        groups.append((fine_label, g3))

    # Group 4: Base terrain tiles
    if base_level is not None:
        base_col = _find_collection(f"L{base_level}")
        g4: List[bpy.types.Object] = []
        if base_col:
            g4.extend(o for o in base_col.objects if o.type == "MESH")
        if g4:
            groups.append((f"L{base_level}", g4))

    # Group 5: Custom user objects (not already assigned to any group)
    assigned_objects: Set[str] = set()
    for _, objs in groups:
        for o in objs:
            assigned_objects.add(o.name)

    custom_objs: List[bpy.types.Object] = []
    for obj in bpy.data.objects:
        if obj.name in assigned_objects:
            continue
        if obj.type not in ("MESH", "EMPTY"):
            continue
        cols = _get_object_collections(obj)
        skip = False
        for c in cols:
            if c.name in _MASK_COLLECTIONS or c.name.startswith("export_batch"):
                skip = True
                break
            # Skip objects belonging to tile, collision or game_objects collections
            # These should have been assigned by Groups 1-4; if they weren't,
            # they belong to a known group and must not leak into custom.
            if re.match(r"^L\d+$", c.name):
                log.warning("  '%s' in tile collection '%s' but not assigned "
                            "by tile groups — skipping from custom", obj.name, c.name)
                skip = True
                break
            if c.name in _GAME_COLLECTIONS or c.name.startswith("game_objects_"):
                log.warning("  '%s' in game collection '%s' but not assigned "
                            "by earlier groups — skipping from custom", obj.name, c.name)
                skip = True
                break
        if skip:
            continue
        custom_objs.append(obj)

    if custom_objs:
        groups.append(("custom", custom_objs))
        log.info("  Custom user objects: %d", len(custom_objs))

    # Track which objects are assigned to batches
    batched_objects: Set[str] = set()

    # Create batch collections
    batch_names: List[str] = []
    batch_idx = 0

    for group_label, objs in groups:
        if not objs:
            continue
        # De-duplicate (an object might appear in both collision and a parent)
        objs = list({o.name: o for o in objs}.values())

        total_bytes = sum(_estimate_geometry_bytes(o) for o in objs)
        total_mb = total_bytes / (1024 * 1024)
        log.info("  Group '%s': %d objects, ~%.1f MB", group_label, len(objs), total_mb)

        if total_bytes <= max_batch_bytes:
            batch_name = f"export_batch_{batch_idx:02d}_{group_label}"
            batch_col = _ensure_collection(batch_name)
            for obj in objs:
                if obj.name not in batch_col.objects:
                    batch_col.objects.link(obj)
                batched_objects.add(obj.name)
            batch_names.append(batch_name)
            batch_idx += 1
        else:
            current_objs: List[bpy.types.Object] = []
            current_bytes = 0
            for obj in objs:
                obj_bytes = _estimate_geometry_bytes(obj)
                if current_bytes + obj_bytes > max_batch_bytes and current_objs:
                    batch_name = f"export_batch_{batch_idx:02d}_{group_label}"
                    batch_col = _ensure_collection(batch_name)
                    for o in current_objs:
                        if o.name not in batch_col.objects:
                            batch_col.objects.link(o)
                        batched_objects.add(o.name)
                    batch_names.append(batch_name)
                    batch_idx += 1
                    current_objs = []
                    current_bytes = 0
                current_objs.append(obj)
                current_bytes += obj_bytes
            if current_objs:
                batch_name = f"export_batch_{batch_idx:02d}_{group_label}"
                batch_col = _ensure_collection(batch_name)
                for o in current_objs:
                    if o.name not in batch_col.objects:
                        batch_col.objects.link(o)
                    batched_objects.add(o.name)
                batch_names.append(batch_name)
                batch_idx += 1

    log.info("  Organised %d batches: %s", len(batch_names), batch_names)
    log.info("  Total batched objects: %d", len(batched_objects))
    return batch_names


# ===================================================================
# Step 5: Final cleanup — remove everything not in an export batch
# ===================================================================

def step5_final_cleanup(batch_names: List[str]) -> None:
    """Delete all collections and objects not belonging to any export batch.

    After this, the scene contains ONLY export_batch_NN collections and
    the Scene Collection root.
    """
    log.info("Step 5/6: Final cleanup — removing non-export data")

    # Collect all objects that belong to at least one export batch
    export_objects: Set[str] = set()
    export_col_names: Set[str] = set(batch_names)
    for bn in batch_names:
        col = _find_collection(bn)
        if col:
            for obj in col.objects:
                export_objects.add(obj.name)

    # Delete objects not in any export batch
    removed_objs = 0
    for obj in list(bpy.data.objects):
        if obj.name not in export_objects:
            bpy.data.objects.remove(obj, do_unlink=True)
            removed_objs += 1

    # Delete collections that are not export batches (or Scene Collection)
    removed_cols = 0
    for col in list(bpy.data.collections):
        if col.name not in export_col_names:
            try:
                bpy.data.collections.remove(col)
                removed_cols += 1
            except Exception:
                pass  # might already be removed as child

    bpy.ops.outliner.orphans_purge(do_recursive=True)
    log.info("  Removed %d objects, %d collections", removed_objs, removed_cols)


# ===================================================================
# FBX INI generation — called per-batch from step6
# ===================================================================

def _classify_batch(batch_name: str) -> str:
    """Classify batch type: 'collision', 'game_objects', or 'terrain'.

    Labels like ``L17``, ``L18-L19``, ``custom``, ``terrain_fine``,
    ``terrain_base`` all fall through to the ``"terrain"`` default.
    """
    suffix = batch_name.split("_", 3)[-1] if "_" in batch_name else batch_name
    if "collision" in suffix or suffix == "track_core":
        return "collision"
    if suffix.startswith("go_") or suffix == "game_objects":
        return "game_objects"
    return "terrain"


def _collect_materials_terrain(
    objects: List[bpy.types.Object],
) -> List[Dict[str, str]]:
    """Collect unique materials from terrain objects.

    Returns list of dicts: {"name": material_name, "texture": filename}.
    """
    seen: Dict[str, str] = {}  # mat_name -> texture_filename
    for obj in objects:
        if obj.type != "MESH" or not obj.data:
            continue
        for slot in obj.material_slots:
            mat = slot.material
            if mat is None or mat.name in seen:
                continue
            texture = "NULL.dds"
            if mat.use_nodes and mat.node_tree:
                for node in mat.node_tree.nodes:
                    if node.type == "TEX_IMAGE" and node.image:
                        texture = os.path.basename(node.image.filepath)
                        break
            seen[mat.name] = texture
    return [{"name": n, "texture": t} for n, t in seen.items()]


def _write_material_section(
    lines: List[str],
    idx: int,
    name: str,
    texture: str,
    ks_ambient: float,
    ks_diffuse: float,
    ks_emissive: float,
) -> None:
    """Append a [MATERIAL_N] section to lines."""
    em = f"{ks_emissive},{ks_emissive},{ks_emissive}"
    lines.append(f"[MATERIAL_{idx}]")
    lines.append(f"NAME={name}")
    lines.append("SHADER=ksPerPixel")
    lines.append("ALPHABLEND=0")
    lines.append("ALPHATEST=0")
    lines.append("DEPTHMODE=0")
    lines.append("VARCOUNT=6")
    lines.append("VAR_0_NAME=ksAmbient")
    lines.append(f"VAR_0_FLOAT1={ks_ambient}")
    lines.append("VAR_0_FLOAT2=0,0")
    lines.append("VAR_0_FLOAT3=0,0,0")
    lines.append("VAR_0_FLOAT4=0,0,0,0")
    lines.append("VAR_1_NAME=ksDiffuse")
    lines.append(f"VAR_1_FLOAT1={ks_diffuse}")
    lines.append("VAR_1_FLOAT2=0,0")
    lines.append("VAR_1_FLOAT3=0,0,0")
    lines.append("VAR_1_FLOAT4=0,0,0,0")
    lines.append("VAR_2_NAME=ksSpecular")
    lines.append("VAR_2_FLOAT1=0")
    lines.append("VAR_2_FLOAT2=0,0")
    lines.append("VAR_2_FLOAT3=0,0,0")
    lines.append("VAR_2_FLOAT4=0,0,0,0")
    lines.append("VAR_3_NAME=ksSpecularEXP")
    lines.append("VAR_3_FLOAT1=0")
    lines.append("VAR_3_FLOAT2=0,0")
    lines.append("VAR_3_FLOAT3=0,0,0")
    lines.append("VAR_3_FLOAT4=0,0,0,0")
    lines.append("VAR_4_NAME=ksEmissive")
    lines.append("VAR_4_FLOAT1=0")
    lines.append("VAR_4_FLOAT2=0,0")
    lines.append(f"VAR_4_FLOAT3={em}")
    lines.append("VAR_4_FLOAT4=0,0,0,0")
    lines.append("VAR_5_NAME=ksAlphaRef")
    lines.append("VAR_5_FLOAT1=0")
    lines.append("VAR_5_FLOAT2=0,0")
    lines.append("VAR_5_FLOAT3=0,0,0")
    lines.append("VAR_5_FLOAT4=0,0,0,0")
    lines.append("RESCOUNT=1")
    lines.append("RES_0_NAME=txDiffuse")
    lines.append("RES_0_SLOT=0")
    lines.append(f"RES_0_TEXTURE={texture}")
    lines.append("")


def _generate_fbx_ini(
    fbx_path: str,
    batch_name: str,
    objects: List[bpy.types.Object],
    ks_ambient: float,
    ks_diffuse: float,
    ks_emissive: float,
) -> str:
    """Generate a .fbx.ini config file for ksEditor FBX→KN5 conversion.

    Returns the path to the written INI file.
    """
    from datetime import datetime

    ini_path = fbx_path + ".ini"
    fbx_filename = os.path.basename(fbx_path)
    batch_type = _classify_batch(batch_name)

    lines: List[str] = []

    # Header
    lines.append("[HEADER]")
    lines.append("VERSION=4")
    lines.append("DLC_KEY=0")
    lines.append("USERNAME=")
    now = datetime.now()
    if sys.platform == "win32":
        date_str = now.strftime("%#m/%#d/%Y %#I:%M:%S %p")
    else:
        date_str = now.strftime("%-m/%-d/%Y %-I:%M:%S %p")
    lines.append(f"DATE={date_str}")
    lines.append("MD5_HASH=00000000000000000000000000000000")
    lines.append("")

    # Material list
    if batch_type == "collision":
        # Collect unique collision materials (hidden, NULL.dds)
        mat_names: Set[str] = set()
        for obj in objects:
            if obj.type != "MESH" or not obj.data:
                continue
            for slot in obj.material_slots:
                if slot.material:
                    mat_names.add(slot.material.name)
        if not mat_names:
            mat_names.add("hidden")
        mat_list = sorted(mat_names)
        lines.append("[MATERIAL_LIST]")
        lines.append(f"COUNT={len(mat_list)}")
        lines.append("")
        for i, mname in enumerate(mat_list):
            _write_material_section(
                lines, i, mname, "NULL.dds",
                ks_ambient=0.0509, ks_diffuse=0.0782, ks_emissive=0.0,
            )
    elif batch_type == "terrain":
        # Collect real materials with textures
        mats = _collect_materials_terrain(objects)
        lines.append("[MATERIAL_LIST]")
        lines.append(f"COUNT={len(mats)}")
        lines.append("")
        for i, m in enumerate(mats):
            _write_material_section(
                lines, i, m["name"], m["texture"],
                ks_ambient=ks_ambient,
                ks_diffuse=ks_diffuse,
                ks_emissive=ks_emissive,
            )
    else:
        # game_objects — no materials
        lines.append("[MATERIAL_LIST]")
        lines.append("COUNT=0")
        lines.append("")

    # Node entries
    # Root node
    lines.append(f"[model_FBX: {fbx_filename}]")
    lines.append("ACTIVE=1")
    lines.append("PRIORITY=0")
    lines.append("")

    renderable = 0 if batch_type == "collision" else 1

    for obj in sorted(objects, key=lambda o: o.name):
        # Object node
        lines.append(f"[model_FBX: {fbx_filename}_{obj.name}]")
        lines.append("ACTIVE=1")
        lines.append("PRIORITY=0")
        lines.append("")

        # Mesh data leaf node (only for MESH objects)
        if obj.type == "MESH" and obj.data:
            mesh_name = obj.data.name
            lines.append(f"[model_FBX: {fbx_filename}_{obj.name}_{mesh_name}]")
            lines.append("ACTIVE=1")
            lines.append("PRIORITY=0")
            lines.append("VISIBLE=1")
            lines.append("TRANSPARENT=0")
            lines.append("CAST_SHADOWS=1")
            lines.append("LOD_IN=0")
            lines.append("LOD_OUT=0")
            lines.append(f"RENDERABLE={renderable}")
            lines.append("")

    with open(ini_path, "w", encoding="utf-8", newline="\r\n") as f:
        f.write("\n".join(lines))

    log.info("  INI: %s (%s, %d materials)",
             os.path.basename(ini_path), batch_type,
             len([l for l in lines if l.startswith("[MATERIAL_")]))
    return ini_path


# ===================================================================
# Step 6: Save blend + export FBX
# ===================================================================

def step6_save_and_export(
    batch_names: List[str],
    output_dir: str,
    fbx_scale: float,
    ks_ambient: float = 0.5,
    ks_diffuse: float = 0.1,
    ks_emissive: float = 0.1,
) -> List[str]:
    """Save intermediate .blend then export each batch as FBX (no textures)."""
    # Save intermediate blend
    blend_path = os.path.join(output_dir, "organized_track.blend")
    log.info("Step 6/6: Save + FBX export (%d batches)", len(batch_names))
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    log.info("  Saved: %s", blend_path)

    exported: List[str] = []

    for batch_name in batch_names:
        col = _find_collection(batch_name)
        if col is None:
            log.warning("  Batch '%s' not found, skipping", batch_name)
            continue

        bpy.ops.object.select_all(action="DESELECT")
        obj_count = 0
        for obj in col.objects:
            obj.select_set(True)
            obj_count += 1

        if obj_count == 0:
            log.info("  '%s': empty, skipping", batch_name)
            continue

        # Strip layout prefix before FBX export for game objects batches.
        # "go_" prefix = per-layout batch, "game_objects" = single-layout batch.
        is_go_batch = batch_name.split("_", 3)[-1].startswith("go_") or \
                      batch_name.endswith("game_objects")
        renamed: Dict[bpy.types.Object, str] = {}
        if is_go_batch:
            for obj in col.objects:
                if "__" in obj.name:
                    original = obj.name.split("__", 1)[1]
                    renamed[obj] = obj.name
                    obj.name = original
            if renamed:
                log.info("  '%s': stripped layout prefix from %d objects",
                         batch_name, len(renamed))

        # Ensure mesh data names match object names for ksEditorAt INI matching.
        # Some creators (boolean_mesh_generator) add _mesh suffix to data names,
        # which causes RENDERABLE=0 to not be applied during FBX->KN5 conversion.
        for obj in col.objects:
            if obj.type == "MESH" and obj.data and obj.data.name != obj.name:
                obj.data.name = obj.name

        fbx_path = os.path.join(output_dir, f"{batch_name}.fbx")

        bpy.ops.export_scene.fbx(
            filepath=fbx_path,
            use_selection=True,
            global_scale=fbx_scale,
            apply_scale_options="FBX_SCALE_ALL",
            axis_forward="-Y",
            axis_up="Z",
            path_mode="AUTO",
            embed_textures=False,
            mesh_smooth_type="OFF",
            use_mesh_modifiers=True,
        )

        if os.path.isfile(fbx_path):
            size_mb = os.path.getsize(fbx_path) / (1024 * 1024)
            log.info("  '%s': %d objects -> %.1f MB (%s)",
                     batch_name, obj_count, size_mb, fbx_path)
            exported.append(fbx_path)
            # Generate companion .fbx.ini (before name restore — names must match FBX)
            _generate_fbx_ini(
                fbx_path, batch_name, list(col.objects),
                ks_ambient, ks_diffuse, ks_emissive,
            )
        else:
            log.warning("  '%s': FBX export failed", batch_name)

        # Restore prefixed names after export + INI (blend file may be saved again)
        for obj, old_name in renamed.items():
            obj.name = old_name

    log.info("  Exported %d FBX files", len(exported))
    return exported


# ===================================================================
# Main
# ===================================================================

def _parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    p = argparse.ArgumentParser(description="Blender Stage 10: Model export")
    p.add_argument("--blend-input", required=True, help="Input .blend file")
    p.add_argument("--output-dir", required=True, help="FBX output directory")
    p.add_argument("--tiles-dir", default="", help="Tileset directory for coordinate transform")
    p.add_argument("--centerline-json", default="", help="Centerline JSON for road splitting")
    p.add_argument("--geo-metadata", default="", help="geo_metadata.json for coordinate conversion")
    p.add_argument("--max-vertices", type=int, default=21000, help="Max vertices per mesh")
    p.add_argument("--max-batch-mb", type=int, default=100, help="Max FBX batch size MB")
    p.add_argument("--fbx-scale", type=float, default=0.01, help="FBX export scale")
    p.add_argument("--ks-ambient", type=float, default=0.5, help="ksAmbient for visible materials")
    p.add_argument("--ks-diffuse", type=float, default=0.1, help="ksDiffuse for visible materials")
    p.add_argument("--ks-emissive", type=float, default=0.1, help="ksEmissive for visible materials")
    return p.parse_args(argv)


def main() -> None:
    args = _parse_args()
    log.info("=" * 60)
    log.info("Stage 10: Model Export")
    log.info("  Input:  %s", args.blend_input)
    log.info("  Output: %s", args.output_dir)
    log.info("  Max vertices: %d", args.max_vertices)
    log.info("  Max batch MB: %d", args.max_batch_mb)
    log.info("  FBX scale: %.4f", args.fbx_scale)
    log.info("=" * 60)

    bpy.ops.wm.open_mainfile(filepath=os.path.abspath(args.blend_input))
    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Cleanup masks
    _emit_progress(5, "Cleanup masks...")
    step1_cleanup()

    # Step 2: Split oversized meshes
    _emit_progress(15, "Splitting oversized meshes...")
    centerline = None
    if args.centerline_json and args.geo_metadata and args.tiles_dir:
        centerline = _load_centerline_blender(
            args.centerline_json, args.geo_metadata, args.tiles_dir,
        )
    step2_split_meshes(args.max_vertices, centerline)

    # Step 3: Rename collision objects
    _emit_progress(40, "Renaming collision objects...")
    step3_rename_collision()

    # Step 4: Batch organisation (auto-detects tile levels)
    _emit_progress(55, "Batch organisation...")
    batch_names = step4_batch_organise(args.max_batch_mb)

    # Step 5: Remove all non-export data
    _emit_progress(65, "Final cleanup...")
    step5_final_cleanup(batch_names)

    # Step 6: Save + export FBX (with INI generation)
    _emit_progress(75, "Saving and exporting FBX...")
    exported = step6_save_and_export(
        batch_names, args.output_dir, args.fbx_scale,
        ks_ambient=args.ks_ambient,
        ks_diffuse=args.ks_diffuse,
        ks_emissive=args.ks_emissive,
    )

    _emit_progress(100, "Export complete")
    log.info("=" * 60)
    log.info("Stage 10 complete: %d FBX files exported to %s", len(exported), args.output_dir)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
