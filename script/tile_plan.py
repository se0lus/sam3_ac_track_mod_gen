"""
Pre-compute tile loading plan — pure Python, no Blender dependency.

Given polygon AABBs from Stage 8 and a CTile tree from tileset.json,
determines exactly which tiles to load at which level, avoiding
over-refinement of tiles that don't overlap any polygon.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Import CTile — works both inside Blender (sam3_actions package) and
# standalone (direct module import, avoids bpy dependency in __init__.py)
# ---------------------------------------------------------------------------
_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_this_dir)

try:
    from sam3_actions.c_tiles import CTile  # inside Blender
except (ImportError, ModuleNotFoundError):
    # Standalone: import c_tiles.py directly (bypasses __init__.py which needs bpy)
    _actions_dir = os.path.join(_project_root, "blender_scripts", "sam3_actions")
    if _actions_dir not in sys.path:
        sys.path.insert(0, _actions_dir)
    from c_tiles import CTile  # type: ignore[no-redef]  # noqa: E402

log = logging.getLogger("tile_plan")

# Type alias: (min_x, max_x, min_z, max_z)
AABB2D = Tuple[float, float, float, float]


# ---------------------------------------------------------------------------
# 2D geometry helpers
# ---------------------------------------------------------------------------

def bbox2_overlap(a: AABB2D, b: AABB2D) -> bool:
    """Test if two 2D AABBs overlap. Format: (min_x, max_x, min_z, max_z)."""
    return (a[1] >= b[0]) and (a[0] <= b[1]) and (a[3] >= b[2]) and (a[2] <= b[3])


def tile_aabb_xz(tile: CTile, axis_map: str) -> Optional[AABB2D]:
    """Convert CTile bounding volume to Blender XZ AABB.

    Returns ``(min_x, max_x, min_z, max_z)`` or ``None`` if bbox is invalid.
    When ``None`` is returned, the caller should treat the tile conservatively
    (assume overlap).
    """
    bb = tile.boxBoundingVolume
    if not bb or len(bb) < 12 or all(v == 0 for v in bb):
        return None

    cx, cy, cz = bb[0], bb[1], bb[2]
    # Half-extents: handle general OBB case conservatively
    hx = abs(bb[3]) + abs(bb[4]) + abs(bb[5])
    hy = abs(bb[6]) + abs(bb[7]) + abs(bb[8])
    hz = abs(bb[9]) + abs(bb[10]) + abs(bb[11])

    if axis_map == "XnY":
        return (cx - hx, cx + hx, -cy - hy, -cy + hy)
    elif axis_map == "XZ":
        return (cx - hx, cx + hx, cz - hz, cz + hz)
    elif axis_map == "XnZ":
        return (cx - hx, cx + hx, -cz - hz, -cz + hz)
    else:  # "XY" or default
        return (cx - hx, cx + hx, cy - hy, cy + hy)


# ---------------------------------------------------------------------------
# Polygon AABB extraction from Stage 8 JSON
# ---------------------------------------------------------------------------

def extract_polygon_aabbs(json_path: str, padding_m: float = 0.5) -> List[AABB2D]:
    """Extract XZ AABBs from a Stage 8 ``*_merged_blender.json`` file.

    Each ``mesh_group`` in the JSON has ``points_xyz`` in Blender coords
    (tileset-local). We extract the XZ bounding box of each group.

    Args:
        json_path: Path to Stage 8 ``{tag}_merged_blender.json``
        padding_m: Padding in metres around each polygon AABB

    Returns:
        List of ``(min_x, max_x, min_z, max_z)`` tuples
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    aabbs: List[AABB2D] = []
    for mg in data.get("mesh_groups", []):
        pts = mg.get("points_xyz", [])
        if len(pts) < 3:
            continue
        xs = [p[0] for p in pts]
        zs = [p[2] for p in pts]
        aabbs.append((
            min(xs) - padding_m,
            max(xs) + padding_m,
            min(zs) - padding_m,
            max(zs) + padding_m,
        ))
    return aabbs


def extract_polygon_aabbs_for_tags(
    polygon_dir: str,
    tags: List[str],
    padding_m: float = 0.5,
) -> List[AABB2D]:
    """Extract polygon AABBs for multiple tags from Stage 8 gap_filled output.

    Args:
        polygon_dir: Path to gap_filled directory
            (e.g., ``output/08_blender_polygons/gap_filled``)
        tags: List of tags to include (e.g., ``["road"]``)
        padding_m: Padding in metres

    Returns:
        Combined list of AABBs from all matching tags
    """
    all_aabbs: List[AABB2D] = []
    for tag in tags:
        json_path = os.path.join(polygon_dir, tag, f"{tag}_merged_blender.json")
        if os.path.isfile(json_path):
            aabbs = extract_polygon_aabbs(json_path, padding_m)
            all_aabbs.extend(aabbs)
            log.info("  tag '%s': %d polygon AABBs from %s", tag, len(aabbs), json_path)
        else:
            log.warning("  tag '%s': JSON not found at %s", tag, json_path)
    return all_aabbs


# ---------------------------------------------------------------------------
# Tileset tree loading (mirrors blender_automate._load_tileset_tree)
# ---------------------------------------------------------------------------

def load_tileset_tree(tiles_dir: str) -> CTile:
    """Load CTile tree from *tiles_dir*.

    Tries ``tiles_dir/tileset.json`` first.  If absent, walks subdirectories
    for block-level ``tileset.json`` files and assembles under a virtual root.
    """
    top_json = os.path.join(tiles_dir, "tileset.json")
    root_tile = CTile()
    if os.path.isfile(top_json):
        root_tile.loadFromRootJson(top_json)
        return root_tile

    found = 0
    for dirpath, _dirnames, filenames in os.walk(tiles_dir):
        for fname in filenames:
            if fname.lower() == "tileset.json":
                child = CTile()
                child.loadFromRootJson(os.path.join(dirpath, fname))
                if child.children or child.hasMesh:
                    root_tile.children.append(child)
                    child.parent = root_tile
                    found += 1
    if found:
        root_tile.canRefine = True
        log.info("Assembled virtual root from %d block tileset(s)", found)
    else:
        log.warning("No tileset.json found in %s or subdirectories", tiles_dir)
    return root_tile


# ---------------------------------------------------------------------------
# Axis-map detection (no Blender needed)
# ---------------------------------------------------------------------------

def _collect_mesh_tiles(tile: CTile, out: List[CTile], max_count: int = 20) -> None:
    """Collect up to *max_count* mesh tiles from the tree (BFS-ish)."""
    if len(out) >= max_count:
        return
    if tile.hasMesh:
        out.append(tile)
    for child in tile.children:
        if len(out) >= max_count:
            return
        _collect_mesh_tiles(child, out, max_count)


def detect_axis_map(
    tiles_dir: str,
    polygon_aabbs: Optional[List[AABB2D]] = None,
) -> str:
    """Detect axis mapping from tileset metadata and polygon overlap.

    Tries each candidate mapping and picks the one where the most sampled
    tiles overlap the polygon extent.  Falls back to ``"XY"`` (standard for
    ``gltfUpAxis: "Z"`` tilesets).

    Returns: ``"XY"``, ``"XnY"``, ``"XZ"``, or ``"XnZ"``
    """
    if polygon_aabbs is None or len(polygon_aabbs) == 0:
        return "XY"

    # Compute polygon overall AABB
    poly_min_x = min(a[0] for a in polygon_aabbs)
    poly_max_x = max(a[1] for a in polygon_aabbs)
    poly_min_z = min(a[2] for a in polygon_aabbs)
    poly_max_z = max(a[3] for a in polygon_aabbs)
    poly_overall: AABB2D = (poly_min_x, poly_max_x, poly_min_z, poly_max_z)

    root = load_tileset_tree(tiles_dir)
    sample_tiles: List[CTile] = []
    _collect_mesh_tiles(root, sample_tiles, max_count=30)

    if not sample_tiles:
        return "XY"

    candidates = ["XY", "XnY", "XZ", "XnZ"]
    best_map = "XY"
    best_count = 0

    for candidate in candidates:
        overlap_count = 0
        for tile in sample_tiles:
            aabb = tile_aabb_xz(tile, candidate)
            if aabb is not None and bbox2_overlap(aabb, poly_overall):
                overlap_count += 1
        if overlap_count > best_count:
            best_count = overlap_count
            best_map = candidate

    log.info("Axis map detection: %s (overlap=%d/%d tiles)",
             best_map, best_count, len(sample_tiles))
    return best_map


# ---------------------------------------------------------------------------
# Core algorithm: compute tile load plan
# ---------------------------------------------------------------------------

def compute_tile_load_plan(
    root: CTile,
    polygon_aabbs: List[AABB2D],
    base_level: int,
    target_level: int,
    axis_map: str = "XY",
) -> Dict[int, List[CTile]]:
    """Compute which tiles to load at which level.

    Walk the CTile tree top-down.  For each mesh tile:

    - Tiles below *base_level* → always recurse (refine to at least base)
    - Tiles at/above *base_level* that don't overlap any polygon → load as-is
    - Tiles that overlap polygons → recurse until *target_level*
    - Tiles at *target_level* or that can't refine further → load as-is

    This ensures:

    1. Complete coverage (no holes) — every area has a tile
    2. Only polygon-adjacent areas are refined to *target_level*
    3. Non-polygon areas stay at *base_level*

    Args:
        root: Root CTile tree
        polygon_aabbs: List of polygon AABBs to refine around
        base_level: Minimum level to load (e.g., 17)
        target_level: Target refinement level for overlapping areas (e.g., 19)
        axis_map: Axis mapping string

    Returns:
        Dict mapping level → list of CTile to load
    """
    plan: Dict[int, List[CTile]] = {}

    def _overlaps_any(tile: CTile) -> bool:
        aabb = tile_aabb_xz(tile, axis_map)
        if aabb is None:
            return True  # Can't determine → conservative (assume overlap)
        for poly_aabb in polygon_aabbs:
            if bbox2_overlap(aabb, poly_aabb):
                return True
        return False

    def _walk(tile: CTile) -> bool:
        """Walk tile tree, return True if at least one tile was added."""
        if tile.hasMesh:
            # Reached target level or can't refine → load directly.
            # If the tree skips levels (e.g. L18→L20 with no L19),
            # meshLevel may exceed target — that's correct: mask areas
            # require AT LEAST target_level quality.
            if tile.meshLevel >= target_level or not tile.canRefine or not tile.children:
                plan.setdefault(tile.meshLevel, []).append(tile)
                return True

            # At or above base level and no polygon overlap → stop
            if tile.meshLevel >= base_level and not _overlaps_any(tile):
                plan.setdefault(tile.meshLevel, []).append(tile)
                return True

            # Below base level OR overlapping → recurse into children
            any_child = False
            for child in tile.children:
                if _walk(child):
                    any_child = True
            if not any_child:
                # Children produced nothing → fallback to this tile
                plan.setdefault(tile.meshLevel, []).append(tile)
            return True

        # Non-mesh node (virtual root, JSON redirect) → recurse.
        # Must propagate True if any descendant was added to the plan,
        # otherwise the parent mesh node will incorrectly add itself
        # as a fallback (causing parent-child overlap).
        if tile.canRefine and tile.children:
            any_child = False
            for child in tile.children:
                if _walk(child):
                    any_child = True
            return any_child
        return False

    _walk(root)
    return plan


# ---------------------------------------------------------------------------
# High-level convenience interface
# ---------------------------------------------------------------------------

def compute_plan_from_config(
    tiles_dir: str,
    polygon_dir: str,
    tags: List[str],
    base_level: int,
    target_level: int,
    padding_m: float = 0.5,
) -> Dict[int, List[CTile]]:
    """Compute tile load plan from pipeline paths and config.

    Args:
        tiles_dir: Path to tileset directory
        polygon_dir: Path to Stage 8 gap_filled directory
        tags: Tags to refine (e.g., ``["road"]``)
        base_level: Base tile level (e.g., 17)
        target_level: Target refinement level (e.g., 19)
        padding_m: AABB padding in metres (default 0.5)

    Returns:
        Dict mapping level → list of CTile to load
    """
    log.info("Computing tile load plan...")
    log.info("  tiles_dir: %s", tiles_dir)
    log.info("  polygon_dir: %s", polygon_dir)
    log.info("  tags: %s, base=%d, target=%d, padding=%.1fm",
             tags, base_level, target_level, padding_m)

    # 1. Extract polygon AABBs
    polygon_aabbs = extract_polygon_aabbs_for_tags(polygon_dir, tags, padding_m)
    log.info("  Total polygon AABBs: %d", len(polygon_aabbs))

    # 2. Load tile tree
    root = load_tileset_tree(tiles_dir)

    # 3. Detect axis mapping
    axis_map = detect_axis_map(tiles_dir, polygon_aabbs)
    log.info("  Axis map: %s", axis_map)

    # 4. Compute plan
    plan = compute_tile_load_plan(root, polygon_aabbs, base_level, target_level, axis_map)

    # 5. Summary
    total = 0
    for level in sorted(plan.keys()):
        count = len(plan[level])
        total += count
        log.info("  Level %d: %d tiles", level, count)
    log.info("  Total tiles to load: %d", total)

    # 6. Validate — no ancestor should coexist with descendant
    overlap_errors = validate_plan_no_ancestor_overlap(plan)
    if overlap_errors:
        log.warning("TILE PLAN VALIDATION: %d ancestor overlap(s) detected!", len(overlap_errors))
        for err in overlap_errors[:20]:
            log.warning("  %s", err)

    return plan


def log_plan_summary(plan: Dict[int, List[CTile]]) -> None:
    """Print a summary of the load plan to the logger."""
    total = 0
    for level in sorted(plan.keys()):
        count = len(plan[level])
        total += count
        log.info("  Level %d: %d tiles", level, count)
    log.info("  Total tiles to load: %d", total)


def validate_plan_no_ancestor_overlap(plan: Dict[int, List[CTile]]) -> List[str]:
    """Check that no tile in the plan is an ancestor of another.

    If a tile A is in the plan *and* A is an ancestor of tile B which is
    also in the plan, they cover the same geographic area → the coarser
    tile (A) should have been replaced by the finer one (B), not loaded
    alongside it.

    Returns a list of error descriptions (empty means valid).
    """
    all_tiles = []
    for level in plan:
        for tile in plan[level]:
            all_tiles.append(tile)

    # Build set of tile ids for O(1) membership check
    plan_ids = {id(t) for t in all_tiles}

    errors: List[str] = []
    for tile in all_tiles:
        # Walk up the parent chain — if any ancestor is also in the plan,
        # that's an overlap error
        ancestor = tile.parent
        while ancestor is not None:
            if id(ancestor) in plan_ids:
                errors.append(
                    f"Overlap: '{tile.content}' (L{tile.meshLevel}) and "
                    f"ancestor '{ancestor.content}' (L{ancestor.meshLevel}) "
                    f"are both in the plan"
                )
                break
            ancestor = ancestor.parent

    return errors
