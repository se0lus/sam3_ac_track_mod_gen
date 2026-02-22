"""
Track centerline extraction, curvature analysis, bend detection, and timing point generation.

Pipeline: road mask (binary image) -> skeleton -> ordered centerline -> curvature ->
composite bends -> AC_TIME_N_L / AC_TIME_N_R pairs.

Pure Python -- no Blender dependency.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from scipy.ndimage import label as ndimage_label
from scipy.signal import savgol_filter
from skimage.morphology import skeletonize

logger = logging.getLogger("sam3_pipeline.centerline")


# ---------------------------------------------------------------------------
# 1. Centerline extraction
# ---------------------------------------------------------------------------

def extract_centerline(
    road_mask: np.ndarray,
    pixel_size_m: float = 0.3,
    morph_close_gap_m: float = 1.5,
    min_branch_length_m: float = 15.0,
    resample_spacing_m: float = 2.5,
    smooth_window_m: float = 120.0,
    simplify_tolerance: float = 0.0,
) -> np.ndarray:
    """Binary road mask -> ordered centerline coordinates (N, 2) in pixel space.

    All algorithmic parameters are specified in physical units (meters) and
    converted to pixels at runtime using *pixel_size_m*.

    Args:
        road_mask: HxW uint8 image (>127 = road).
        pixel_size_m: Meters per pixel (from GeoTIFF metadata).
        morph_close_gap_m: Morphological close kernel radius in meters.
        min_branch_length_m: Prune skeleton branches shorter than this (meters).
        resample_spacing_m: Uniform resampling interval (meters) before smoothing.
        smooth_window_m: Savitzky-Golay smoothing window length in meters.
        simplify_tolerance: Douglas-Peucker epsilon (0 = no simplification, in pixels).

    Returns:
        (N, 2) float64 array of [x, y] centerline points, ordered along the track.
    """
    # --- Convert physical units to pixels ---
    morph_px = max(3, int(round(morph_close_gap_m / pixel_size_m)))
    ksize = morph_px | 1  # ensure odd
    min_branch_px = max(5, int(round(min_branch_length_m / pixel_size_m)))
    resample_px = max(1.0, resample_spacing_m / pixel_size_m)
    smooth_pts = max(7, int(round(smooth_window_m / resample_spacing_m)))
    smooth_pts = smooth_pts | 1  # ensure odd
    min_cycle_px = max(20, int(round(15.0 / pixel_size_m)))  # 15 m

    logger.debug("extract_centerline: pixel_size_m=%.3f, morph_px=%d, "
                 "min_branch_px=%d, resample_px=%.1f, smooth_pts=%d",
                 pixel_size_m, ksize, min_branch_px, resample_px, smooth_pts)

    # Binarize
    binary = (road_mask > 127).astype(np.uint8)

    # Morphological close to fill small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Skeletonize
    skel = skeletonize(binary > 0).astype(np.uint8)

    # Prune short branches
    skel = _prune_branches(skel, min_length=min_branch_px)

    # Extract ordered path from skeleton
    path = _order_skeleton_path(skel, min_cycle_length=min_cycle_px)
    if path is None or len(path) < 10:
        logger.warning("Skeleton path too short (%d points), returning raw skeleton pixels",
                        len(path) if path is not None else 0)
        ys, xs = np.where(skel > 0)
        if len(xs) == 0:
            return np.zeros((0, 2), dtype=np.float64)
        return np.column_stack([xs, ys]).astype(np.float64)

    # Uniform resampling
    path = _resample_path(path, spacing=resample_px)

    # Savitzky-Golay smoothing
    if len(path) > 15:
        window = min(len(path) // 2 * 2 - 1, smooth_pts)  # must be odd, capped
        if window >= 7:
            path[:, 0] = savgol_filter(path[:, 0], window, 3, mode="wrap")
            path[:, 1] = savgol_filter(path[:, 1], window, 3, mode="wrap")

    # Optional Douglas-Peucker simplification (disabled by default for curvature accuracy)
    if simplify_tolerance > 0 and len(path) > 20:
        contour = path.reshape(-1, 1, 2).astype(np.float32)
        approx = cv2.approxPolyDP(contour, simplify_tolerance, closed=True)
        path = approx.reshape(-1, 2).astype(np.float64)

    return path


def _prune_branches(skel: np.ndarray, min_length: int = 50) -> np.ndarray:
    """Iteratively remove short branches (degree-1 endpoints) from skeleton."""
    skel = skel.copy()
    changed = True
    while changed:
        changed = False
        # Find endpoints (degree == 1 in 8-connectivity)
        endpoints = _find_endpoints(skel)
        for ey, ex in endpoints:
            branch = _trace_branch(skel, ey, ex)
            if len(branch) < min_length:
                for by, bx in branch:
                    skel[by, bx] = 0
                changed = True
    return skel


def _find_endpoints(skel: np.ndarray) -> List[Tuple[int, int]]:
    """Find skeleton pixels with exactly 1 neighbor (8-connectivity)."""
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(skel, -1, kernel)
    ys, xs = np.where((skel > 0) & (neighbor_count == 1))
    return list(zip(ys.tolist(), xs.tolist()))


def _trace_branch(skel: np.ndarray, start_y: int, start_x: int) -> List[Tuple[int, int]]:
    """Trace from an endpoint along the skeleton until hitting a junction (degree > 2)."""
    h, w = skel.shape
    visited = set()
    branch = []
    cy, cx = start_y, start_x

    while True:
        visited.add((cy, cx))
        branch.append((cy, cx))

        neighbors = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and skel[ny, nx] > 0 and (ny, nx) not in visited:
                    neighbors.append((ny, nx))

        if len(neighbors) == 0:
            break
        if len(neighbors) > 1:
            # Reached junction — stop (don't delete junction pixel)
            break
        cy, cx = neighbors[0]

    return branch


def _order_skeleton_path(skel: np.ndarray, min_cycle_length: int = 50) -> Optional[np.ndarray]:
    """Order skeleton pixels into the main track loop.

    Strategy: build a junction graph (nodes = junctions/endpoints, edges =
    pixel paths between them), then find the longest cycle via DFS on the
    simplified graph.  Finally concatenate edge pixel paths to reconstruct
    the full-resolution centerline.

    Args:
        skel: Binary skeleton image.
        min_cycle_length: Minimum cycle length in pixels to be considered valid.

    Returns (N, 2) array of [x, y] coordinates.
    """
    ys, xs = np.where(skel > 0)
    if len(xs) == 0:
        return None

    # Build pixel adjacency (row, col)
    pixel_set = set(zip(ys.tolist(), xs.tolist()))
    adj: Dict[Tuple[int, int], List[Tuple[int, int]]] = {p: [] for p in pixel_set}
    for (py, px) in pixel_set:
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                nb = (py + dy, px + dx)
                if nb in pixel_set:
                    adj[(py, px)].append(nb)

    # Largest connected component
    visited_global: set = set()
    components: List[List[Tuple[int, int]]] = []
    for p in pixel_set:
        if p in visited_global:
            continue
        comp: List[Tuple[int, int]] = []
        stack = [p]
        while stack:
            node = stack.pop()
            if node in visited_global:
                continue
            visited_global.add(node)
            comp.append(node)
            for nb in adj[node]:
                if nb not in visited_global:
                    stack.append(nb)
        components.append(comp)

    largest_comp = max(components, key=len)
    comp_set = set(largest_comp)

    # Classify pixels by degree (within largest component)
    degree: Dict[Tuple[int, int], int] = {}
    for p in largest_comp:
        degree[p] = sum(1 for nb in adj[p] if nb in comp_set)

    # Nodes = junctions (degree >= 3) + endpoints (degree == 1)
    nodes = {p for p in largest_comp if degree[p] != 2}
    if not nodes:
        # Pure loop with no junctions — just walk it
        return _simple_greedy_walk(largest_comp, adj, comp_set)

    # Trace edges between nodes
    edges: List[Tuple[Tuple[int, int], Tuple[int, int], List[Tuple[int, int]]]] = []
    visited_edges: set = set()

    for node in nodes:
        for nb in adj[node]:
            if nb not in comp_set:
                continue
            edge_key = (min(node, nb), max(node, nb))
            if edge_key in visited_edges:
                continue
            visited_edges.add(edge_key)

            # Trace from node through nb until we hit another node
            path_pixels = [node, nb]
            current = nb
            prev = node
            while current not in nodes:
                next_p = None
                for n2 in adj[current]:
                    if n2 != prev and n2 in comp_set:
                        next_p = n2
                        break
                if next_p is None:
                    break
                edge_key2 = (min(current, next_p), max(current, next_p))
                visited_edges.add(edge_key2)
                path_pixels.append(next_p)
                prev = current
                current = next_p

            end_node = current
            if end_node in nodes:
                edges.append((node, end_node, path_pixels))

    logger.info("Junction graph: %d nodes, %d edges", len(nodes), len(edges))

    # Build adjacency list for simplified graph
    node_adj: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], int, List[Tuple[int, int]]]]] = {
        n: [] for n in nodes
    }
    for i, (a, b, pixels) in enumerate(edges):
        node_adj[a].append((b, i, pixels))
        node_adj[b].append((a, i, list(reversed(pixels))))

    # Find longest cycle via DFS on junction graph
    best_cycle: Optional[List[int]] = None  # list of edge indices
    best_cycle_len = 0

    def _dfs_cycle(
        start: Tuple[int, int],
        current: Tuple[int, int],
        visited_nodes: set,
        used_edges: set,
        path_edges: List[int],
        path_len: int,
    ) -> None:
        nonlocal best_cycle, best_cycle_len
        for (neighbor, edge_idx, _pixels) in node_adj[current]:
            if edge_idx in used_edges:
                continue
            edge_len = len(edges[edge_idx][2])
            if neighbor == start and len(path_edges) >= 2:
                # Found a cycle
                total = path_len + edge_len
                if total > best_cycle_len:
                    best_cycle_len = total
                    best_cycle = path_edges + [edge_idx]
            elif neighbor not in visited_nodes:
                visited_nodes.add(neighbor)
                used_edges.add(edge_idx)
                path_edges.append(edge_idx)
                _dfs_cycle(start, neighbor, visited_nodes, used_edges,
                           path_edges, path_len + edge_len)
                path_edges.pop()
                used_edges.discard(edge_idx)
                visited_nodes.discard(neighbor)

    # Try starting from each node (limit to nodes with high degree for speed)
    sorted_nodes = sorted(nodes, key=lambda p: -degree[p])
    for start_node in sorted_nodes[:min(len(sorted_nodes), 10)]:
        _dfs_cycle(start_node, start_node, {start_node}, set(), [], 0)

    if best_cycle is None or best_cycle_len < min_cycle_length:
        logger.warning("No suitable cycle found in junction graph, falling back to greedy walk")
        return _simple_greedy_walk(largest_comp, adj, comp_set)

    logger.info("Best cycle: %d edges, %d pixels", len(best_cycle), best_cycle_len)

    # Reconstruct full pixel path from cycle edges
    # Need to orient edges correctly: each edge's end must match next edge's start
    cycle_edges = best_cycle
    ordered_pixels: List[Tuple[int, int]] = []

    # Determine edge orientations by chaining
    for ci, edge_idx in enumerate(cycle_edges):
        a, b, pixels = edges[edge_idx]
        if ci == 0:
            # First edge: try both orientations, pick one that chains with next
            if len(cycle_edges) > 1:
                next_a, next_b, _ = edges[cycle_edges[1]]
                if b == next_a or b == next_b:
                    ordered_pixels.extend(pixels[:-1])  # trim last (= next edge's start)
                else:
                    ordered_pixels.extend(reversed(pixels))
                    ordered_pixels = list(ordered_pixels)[:-1]
            else:
                ordered_pixels.extend(pixels[:-1])
        else:
            # Orient so that first pixel matches last added
            last_added = ordered_pixels[-1] if ordered_pixels else None
            if len(pixels) > 0 and pixels[0] == last_added:
                ordered_pixels.extend(pixels[1:-1] if ci < len(cycle_edges) - 1 else pixels[1:])
            elif len(pixels) > 0 and pixels[-1] == last_added:
                rev = list(reversed(pixels))
                ordered_pixels.extend(rev[1:-1] if ci < len(cycle_edges) - 1 else rev[1:])
            else:
                # Can't chain — just append
                ordered_pixels.extend(pixels[1:-1] if ci < len(cycle_edges) - 1 else pixels[1:])

    if len(ordered_pixels) < 10:
        logger.warning("Cycle pixel reconstruction too short (%d), falling back", len(ordered_pixels))
        return _simple_greedy_walk(largest_comp, adj, comp_set)

    logger.info("Centerline cycle: %d pixels", len(ordered_pixels))

    # Convert (row, col) → [x, y]
    result = np.array([[px, py] for (py, px) in ordered_pixels], dtype=np.float64)
    return result


def _simple_greedy_walk(
    comp: List[Tuple[int, int]],
    adj: Dict[Tuple[int, int], List[Tuple[int, int]]],
    comp_set: set,
) -> np.ndarray:
    """Fallback: angle-ordered greedy walk for simple skeletons without junctions."""
    start = comp[0]
    for p in comp:
        deg = sum(1 for nb in adj[p] if nb in comp_set)
        if deg == 1:
            start = p
            break

    path: List[Tuple[int, int]] = [start]
    visited: set = {start}
    heading_x, heading_y = 0.0, 0.0
    # Initialize heading
    for nb in adj[start]:
        if nb in comp_set:
            heading_x = float(nb[1] - start[1])
            heading_y = float(nb[0] - start[0])
            break

    current = start
    while True:
        candidates = [nb for nb in adj[current] if nb not in visited and nb in comp_set]
        if not candidates:
            break
        if len(candidates) == 1:
            chosen = candidates[0]
        else:
            h_len = math.hypot(heading_x, heading_y)
            if h_len < 1e-9:
                chosen = candidates[0]
            else:
                hx_n, hy_n = heading_x / h_len, heading_y / h_len
                best, best_cos = candidates[0], -2.0
                for nb in candidates:
                    dx = float(nb[1] - current[1])
                    dy = float(nb[0] - current[0])
                    d_len = math.hypot(dx, dy)
                    if d_len < 1e-9:
                        continue
                    cos_val = hx_n * dx / d_len + hy_n * dy / d_len
                    if cos_val > best_cos:
                        best_cos = cos_val
                        best = nb
                chosen = best

        heading_x = 0.7 * heading_x + 0.3 * float(chosen[1] - current[1])
        heading_y = 0.7 * heading_y + 0.3 * float(chosen[0] - current[0])
        visited.add(chosen)
        path.append(chosen)
        current = chosen

    logger.info("Greedy walk: %d / %d pixels", len(path), len(comp))
    result = np.array([[px, py] for (py, px) in path], dtype=np.float64)
    return result


def _resample_path(path: np.ndarray, spacing: float = 3.0) -> np.ndarray:
    """Resample path at uniform intervals."""
    # Compute cumulative arc length
    diffs = np.diff(path, axis=0)
    segment_lengths = np.sqrt((diffs ** 2).sum(axis=1))
    cum_length = np.concatenate([[0], np.cumsum(segment_lengths)])
    total_length = cum_length[-1]

    if total_length < spacing * 2:
        return path

    n_samples = max(int(total_length / spacing), 10)
    sample_dists = np.linspace(0, total_length, n_samples, endpoint=False)

    # Interpolate
    resampled = np.zeros((n_samples, 2), dtype=np.float64)
    for dim in range(2):
        resampled[:, dim] = np.interp(sample_dists, cum_length, path[:, dim])

    return resampled


# ---------------------------------------------------------------------------
# 2. Curvature computation
# ---------------------------------------------------------------------------

def compute_curvature(
    centerline: np.ndarray,
    window_m: float = 35.0,
    resample_spacing_m: float = 2.5,
) -> np.ndarray:
    """Compute curvature (unsigned angle change in radians) at each centerline point.

    Args:
        centerline: (N, 2) array of [x, y].
        window_m: Look-ahead/behind distance in meters for direction vectors.
        resample_spacing_m: Resampling interval used when building the centerline (meters).
            Must match the value passed to extract_centerline().

    Returns:
        (N,) array of curvature values (radians, always >= 0).
    """
    window = max(3, int(round(window_m / resample_spacing_m)))
    n = len(centerline)
    curvature = np.zeros(n, dtype=np.float64)

    for i in range(n):
        i_back = (i - window) % n
        i_fwd = (i + window) % n
        v_back = centerline[i] - centerline[i_back]
        v_fwd = centerline[i_fwd] - centerline[i]

        cross = v_back[0] * v_fwd[1] - v_back[1] * v_fwd[0]
        dot = v_back[0] * v_fwd[0] + v_back[1] * v_fwd[1]
        curvature[i] = abs(np.arctan2(cross, dot))

    return curvature


# ---------------------------------------------------------------------------
# 3. Composite bend detection
# ---------------------------------------------------------------------------

def detect_composite_bends(
    centerline: np.ndarray,
    curvature: np.ndarray,
    curvature_threshold: float = 0.50,
    min_bend_length_m: float = 12.0,
    merge_gap_m: float = 25.0,
    resample_spacing_m: float = 2.5,
) -> List[Dict[str, Any]]:
    """Identify composite bends (sustained high-curvature segments).

    Args:
        centerline: (N, 2) array.
        curvature: (N,) array from compute_curvature().
        curvature_threshold: Minimum curvature (radians) to count as "bending".
        min_bend_length_m: Minimum bend length in meters.
        merge_gap_m: Merge segments closer than this distance in meters.
        resample_spacing_m: Resampling interval used when building the centerline (meters).
            Must match the value passed to extract_centerline().

    Returns:
        List of dicts with keys: start_idx, end_idx, peak_idx, exit_idx, total_angle.
    """
    min_bend_length = max(2, int(round(min_bend_length_m / resample_spacing_m)))
    merge_gap = max(1, int(round(merge_gap_m / resample_spacing_m)))
    n = len(curvature)
    is_bend = curvature > curvature_threshold

    # Extract contiguous segments
    segments = []
    in_seg = False
    seg_start = 0

    for i in range(n):
        if is_bend[i] and not in_seg:
            seg_start = i
            in_seg = True
        elif not is_bend[i] and in_seg:
            segments.append((seg_start, i - 1))
            in_seg = False
    if in_seg:
        segments.append((seg_start, n - 1))

    if not segments:
        return []

    # Merge close segments (e.g. chicanes count as one bend)
    merged = [segments[0]]
    for seg in segments[1:]:
        prev = merged[-1]
        gap = seg[0] - prev[1]
        if gap <= merge_gap:
            merged[-1] = (prev[0], seg[1])
        else:
            merged.append(seg)

    # Filter by minimum length and compute bend info
    bends = []
    for start, end in merged:
        length = end - start + 1
        if length < min_bend_length:
            continue

        peak_idx = start + int(np.argmax(curvature[start:end + 1]))
        total_angle = float(np.sum(curvature[start:end + 1]))

        # Exit index: first point after bend where curvature drops below threshold
        exit_idx = (end + 1) % n

        bends.append({
            "start_idx": int(start),
            "end_idx": int(end),
            "peak_idx": int(peak_idx),
            "exit_idx": int(exit_idx),
            "total_angle": round(total_angle, 4),
        })

    logger.info("Detected %d composite bends", len(bends))
    return bends


# ---------------------------------------------------------------------------
# 4. Track width measurement
# ---------------------------------------------------------------------------

def snap_to_centerline(
    point: Union[List[float], Tuple[float, float]],
    centerline: np.ndarray,
) -> Tuple[int, List[float]]:
    """Find the nearest centerline index for a given (x, y) point.

    Args:
        point: [x, y] position to snap.
        centerline: (N, 2) array of centerline points.

    Returns:
        (nearest_index, snapped_point_as_list)
    """
    dists = np.linalg.norm(centerline - np.array(point), axis=1)
    idx = int(np.argmin(dists))
    return idx, centerline[idx].tolist()


def measure_track_width(
    road_mask: np.ndarray,
    centerline: np.ndarray,
    at_index: int,
    max_ray_m: float = 60.0,
    margin_px: float = 0.0,
    pixel_size_m: float = 0.3,
    off_road_tol_m: float = 1.5,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Measure track width at a centerline point by ray-marching perpendicular.

    Args:
        road_mask: HxW uint8 binary mask.
        centerline: (N, 2) array.
        at_index: Index into centerline.
        max_ray_m: Maximum ray length in meters.
        margin_px: Extra pixels to push L/R beyond road edge (for timing gates).
        pixel_size_m: Meters per pixel (from GeoTIFF metadata).
        off_road_tol_m: Tolerance in meters for centerline slightly off-road.

    Returns:
        (left_point [x,y], right_point [x,y], width_pixels).
    """
    max_ray = max(10, int(round(max_ray_m / pixel_size_m)))
    off_road_tol = max(1, int(round(off_road_tol_m / pixel_size_m)))
    n = len(centerline)
    pt = centerline[at_index]

    # Tangent direction (average of forward and backward)
    i_back = (at_index - 3) % n
    i_fwd = (at_index + 3) % n
    tangent = centerline[i_fwd] - centerline[i_back]
    tangent_len = np.linalg.norm(tangent)
    if tangent_len < 1e-6:
        return pt.copy(), pt.copy(), 0.0
    tangent = tangent / tangent_len

    # Normal (perpendicular): rotate tangent 90 degrees
    normal = np.array([-tangent[1], tangent[0]])

    h, w = road_mask.shape[:2]

    def _local_road_edge(direction: np.ndarray, off_road_tol: int = off_road_tol) -> int:
        """Scan outward from *pt* along *direction* and return the distance
        (in pixels) to the far edge of the **nearest** road section.

        - Tolerates up to *off_road_tol* initial off-road pixels (handles
          centerline slightly outside mask due to Savgol smoothing).
        - Stops at the first road→non-road transition, so it never crosses
          to a different track section in switchbacks.
        """
        # Check if centerline point itself is on road
        pt_ix, pt_iy = int(round(pt[0])), int(round(pt[1]))
        found_road = (0 <= pt_ix < w and 0 <= pt_iy < h
                      and road_mask[pt_iy, pt_ix] > 127)
        initial_gap = 0

        for d in range(1, max_ray + 1):
            px = int(round(pt[0] + direction[0] * d))
            py = int(round(pt[1] + direction[1] * d))

            if px < 0 or px >= w or py < 0 or py >= h:
                return d  # image boundary

            if road_mask[py, px] > 127:
                found_road = True
            else:
                if not found_road:
                    # Still looking for the first road pixel
                    initial_gap += 1
                    if initial_gap > off_road_tol:
                        return 0  # no road nearby on this side
                else:
                    # Was on road, now off → local road edge
                    return d

        return max_ray

    left_edge_d = _local_road_edge(normal)
    right_edge_d = _local_road_edge(-normal)

    def _safe_margin(edge_d: int, direction: np.ndarray) -> float:
        """Return the largest margin (up to margin_px) beyond edge_d that
        doesn't land on another road section."""
        for m in range(1, int(margin_px) + 1):
            total_d = edge_d + m
            px = int(round(pt[0] + direction[0] * total_d))
            py = int(round(pt[1] + direction[1] * total_d))
            if 0 <= px < w and 0 <= py < h and road_mask[py, px] > 127:
                # Would intrude into another road section — stop 1px before
                return float(m - 1)
        return margin_px

    left_margin = _safe_margin(left_edge_d, normal) if margin_px > 0 else 0.0
    right_margin = _safe_margin(right_edge_d, -normal) if margin_px > 0 else 0.0

    left_pt = pt + normal * (left_edge_d + left_margin)
    right_pt = pt - normal * (right_edge_d + right_margin)

    width = float(np.linalg.norm(left_pt - right_pt))

    return left_pt, right_pt, width


# ---------------------------------------------------------------------------
# 5. Timing point generation
# ---------------------------------------------------------------------------

def generate_timing_points(
    road_mask: np.ndarray,
    centerline: np.ndarray,
    curvature: np.ndarray,
    bends: List[Dict[str, Any]],
    track_direction: str = "clockwise",
    pixel_size_m: float = 0.3,
) -> List[Dict[str, Any]]:
    """Generate AC_TIME_N_L / AC_TIME_N_R pairs at each composite bend exit.

    Args:
        road_mask: HxW uint8 binary mask.
        centerline: (N, 2) ordered centerline.
        curvature: (N,) curvature array.
        bends: Output of detect_composite_bends().
        track_direction: "clockwise" or "counterclockwise".
        pixel_size_m: Meters per pixel (from GeoTIFF metadata).

    Returns:
        List of game object dicts ready for game_objects.json.
    """
    n = len(centerline)
    objects = []

    for i, bend in enumerate(bends):
        exit_idx = bend["exit_idx"]

        # Measure width at exit
        left_pt, right_pt, width = measure_track_width(
            road_mask, centerline, exit_idx, pixel_size_m=pixel_size_m)

        # Driving direction at exit
        i_back = (exit_idx - 3) % n
        i_fwd = (exit_idx + 3) % n
        tangent = centerline[i_fwd] - centerline[i_back]
        tangent_len = np.linalg.norm(tangent)
        if tangent_len < 1e-6:
            continue
        orientation = tangent / tangent_len

        # Determine left/right based on track direction
        # Normal pointing left of driving direction
        normal_left = np.array([-orientation[1], orientation[0]])
        # For clockwise tracks, "left" of driving direction is toward outside
        if track_direction == "counterclockwise":
            normal_left = -normal_left

        objects.append({
            "name": f"AC_TIME_{i}_L",
            "position": [round(float(left_pt[0]), 1), round(float(left_pt[1]), 1)],
            "orientation_z": [round(float(orientation[0]), 4), round(float(orientation[1]), 4)],
            "type": "timing_left",
        })
        objects.append({
            "name": f"AC_TIME_{i}_R",
            "position": [round(float(right_pt[0]), 1), round(float(right_pt[1]), 1)],
            "orientation_z": [round(float(orientation[0]), 4), round(float(orientation[1]), 4)],
            "type": "timing_right",
        })

    logger.info("Generated %d timing objects (%d pairs) from %d bends",
                len(objects), len(objects) // 2, len(bends))
    return objects


# ---------------------------------------------------------------------------
# 5b. Timing points from TIME_0 (VLM-generated start/finish line)
# ---------------------------------------------------------------------------

def generate_timing_points_from_time0(
    road_mask: np.ndarray,
    centerline: np.ndarray,
    curvature: np.ndarray,
    bends: List[Dict[str, Any]],
    time0_idx: int,
    track_direction: str = "clockwise",
    pixel_size_m: float = 0.3,
    timing_margin_m: float = 3.0,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Generate timing points starting from TIME_0, ordered along driving direction.

    1. Generate AC_TIME_0_L/R at time0_idx (start/finish line)
    2. Sort bends by their exit_idx distance from time0_idx in driving direction
    3. Generate AC_TIME_1_L/R, AC_TIME_2_L/R, ... at each bend exit (strict numeric)
    4. Add display-only "turn_label" (T1, T2, ...) to bends data for UI
    5. Push L/R points beyond road edge by margin

    Args:
        road_mask: HxW uint8 binary mask.
        centerline: (N, 2) ordered centerline.
        curvature: (N,) curvature array.
        bends: Output of detect_composite_bends().
        time0_idx: Centerline index of TIME_0 (from snap_to_centerline).
        track_direction: "clockwise" or "counterclockwise".
        pixel_size_m: Meters per pixel (from geo metadata).
        timing_margin_m: Push L/R beyond road edge by this many meters.

    Returns:
        (timing_objects, labeled_bends)
        - timing_objects: game object dicts with AC_TIME_N_L/R naming (N=0,1,2,...)
        - labeled_bends: bends with added "turn_label" field (display-only: T1, T2, ...)
    """
    n = len(centerline)
    margin_px = timing_margin_m / pixel_size_m if pixel_size_m > 0 else 5.0
    objects: List[Dict[str, Any]] = []

    # Minimum acceptable L/R distance in pixels (~ 6 meters)
    min_lr_dist_px = 6.0 / pixel_size_m if pixel_size_m > 0 else 20.0

    # Determine centerline winding direction using signed area (Shoelace).
    # In image coordinates (y-axis points down):
    #   positive signed area → CW winding
    #   negative signed area → CCW winding
    x, y = centerline[:, 0], centerline[:, 1]
    signed_area = 0.5 * (
        np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
        + x[-1] * y[0] - x[0] * y[-1]
    )
    centerline_is_cw = signed_area > 0
    track_is_cw = (track_direction == "clockwise")

    # If centerline winding matches driving direction, driving follows
    # increasing indices.  Otherwise driving follows decreasing indices.
    driving_follows_index = (centerline_is_cw == track_is_cw)
    logger.info("Centerline winding: %s, track driving: %s → indices %s",
                "CW" if centerline_is_cw else "CCW", track_direction,
                "forward" if driving_follows_index else "reverse")

    # Helper: generate L/R pair at a centerline index
    def _make_pair(timing_num: int, cl_idx: int) -> List[Dict[str, Any]]:
        left_pt, right_pt, width = measure_track_width(
            road_mask, centerline, cl_idx, margin_px=margin_px,
            pixel_size_m=pixel_size_m,
        )

        # If L/R are too close, the road may be very narrow here or the
        # centerline sits near the mask edge.  Fall back to a wider forced
        # spread using half the min distance on each side.
        lr_dist = float(np.linalg.norm(left_pt - right_pt))
        if lr_dist < min_lr_dist_px:
            logger.warning("TIME_%d L/R too close (%.1f px < %.1f px min) at idx %d, forcing spread",
                           timing_num, lr_dist, min_lr_dist_px, cl_idx)
            pt = centerline[cl_idx]
            i_back_t = (cl_idx - 3) % n
            i_fwd_t = (cl_idx + 3) % n
            tang = centerline[i_fwd_t] - centerline[i_back_t]
            tlen = np.linalg.norm(tang)
            if tlen > 1e-6:
                tang = tang / tlen
                norm = np.array([-tang[1], tang[0]])
                half_spread = min_lr_dist_px / 2.0
                left_pt = pt + norm * half_spread
                right_pt = pt - norm * half_spread

        # Driving direction at this point
        i_back = (cl_idx - 3) % n
        i_fwd = (cl_idx + 3) % n
        tangent = centerline[i_fwd] - centerline[i_back]
        tangent_len = np.linalg.norm(tangent)
        if tangent_len < 1e-6:
            return []
        orientation = tangent / tangent_len
        # tangent points in increasing-index direction; flip if driving is reversed
        if not driving_follows_index:
            orientation = -orientation

        return [
            {
                "name": f"AC_TIME_{timing_num}_L",
                "position": [round(float(left_pt[0]), 1), round(float(left_pt[1]), 1)],
                "orientation_z": [round(float(orientation[0]), 4), round(float(orientation[1]), 4)],
                "type": "timing_left",
            },
            {
                "name": f"AC_TIME_{timing_num}_R",
                "position": [round(float(right_pt[0]), 1), round(float(right_pt[1]), 1)],
                "orientation_z": [round(float(orientation[0]), 4), round(float(orientation[1]), 4)],
                "type": "timing_right",
            },
        ]

    # 1. TIME_0 at the start/finish line position
    objects.extend(_make_pair(0, time0_idx))

    # 2. Sort bends by distance from time0_idx in driving direction
    if not bends:
        return objects, []

    if driving_follows_index:
        key_fn = lambda b: (b["exit_idx"] - time0_idx) % n
    else:
        key_fn = lambda b: (time0_idx - b["exit_idx"]) % n

    sorted_bends = sorted(bends, key=key_fn)

    # 3. Generate AC_TIME_1, AC_TIME_2, ... at the straightest point between
    #    consecutive bends (minimum curvature in the gap).

    def _straightest_between(idx_from: int, idx_to: int) -> int:
        """Return centerline index with minimum curvature between idx_from
        and idx_to (in driving direction).  Falls back to midpoint if the
        segment is empty."""
        if driving_follows_index:
            dist = (idx_to - idx_from) % n
            if dist <= 1:
                return idx_from
            indices = [(idx_from + d) % n for d in range(1, dist)]
        else:
            dist = (idx_from - idx_to) % n
            if dist <= 1:
                return idx_from
            indices = [(idx_from - d) % n for d in range(1, dist)]

        if not indices:
            return idx_from
        curv_vals = curvature[indices]
        best_local = int(np.argmin(curv_vals))
        return indices[best_local]

    labeled_bends = []
    for turn_num, bend in enumerate(sorted_bends):
        labeled_bend = dict(bend)
        labeled_bend["turn_label"] = f"T{turn_num + 1}"
        labeled_bends.append(labeled_bend)

    # Generate timing gates between consecutive bend pairs
    for i in range(len(sorted_bends) - 1):
        exit_idx = sorted_bends[i]["exit_idx"]
        next_start_idx = sorted_bends[i + 1]["start_idx"]
        best_idx = _straightest_between(exit_idx, next_start_idx)
        pair = _make_pair(i + 1, best_idx)
        objects.extend(pair)

    logger.info("Generated %d timing objects (%d pairs): TIME_0 at idx %d + %d midpoints between %d bends",
                len(objects), len(objects) // 2, time0_idx,
                max(0, len(sorted_bends) - 1), len(sorted_bends))
    return objects, labeled_bends


def validate_and_fix_timing(
    timing_objects: List[Dict[str, Any]],
    centerline: np.ndarray,
    road_mask: np.ndarray,
    pixel_size_m: float,
    timing_margin_m: float = 3.0,
) -> List[Dict[str, Any]]:
    """Ensure timing L/R are outside road with margin. Fix in-place if needed.

    For each timing L/R point, if it's on the road surface, push it outward
    along the normal until it's off-road + margin.

    Args:
        timing_objects: List of timing object dicts.
        centerline: (N, 2) centerline array.
        road_mask: HxW uint8 binary mask.
        pixel_size_m: Meters per pixel.
        timing_margin_m: Desired margin beyond road edge in meters.

    Returns:
        Fixed timing objects (modified in-place and returned).
    """
    margin_px = timing_margin_m / pixel_size_m if pixel_size_m > 0 else 5.0
    h, w = road_mask.shape[:2]
    n = len(centerline)
    fixed_count = 0

    for obj in timing_objects:
        x, y = obj["position"]
        ix, iy = int(round(x)), int(round(y))

        # Check if on road
        if 0 <= ix < w and 0 <= iy < h and road_mask[iy, ix] > 127:
            # Find nearest centerline point
            _, snapped = snap_to_centerline([x, y], centerline)
            snap_idx, _ = snap_to_centerline([x, y], centerline)

            # Get normal direction at this point
            i_back = (snap_idx - 3) % n
            i_fwd = (snap_idx + 3) % n
            tangent = centerline[i_fwd] - centerline[i_back]
            tangent_len = np.linalg.norm(tangent)
            if tangent_len < 1e-6:
                continue
            tangent = tangent / tangent_len
            normal = np.array([-tangent[1], tangent[0]])

            # Determine push direction: which side of centerline is this point?
            to_point = np.array([x, y]) - centerline[snap_idx]
            dot = np.dot(to_point, normal)
            push_dir = normal if dot >= 0 else -normal

            # Ray-march outward until off-road, then add margin
            new_pt = np.array([x, y])
            for d in range(1, 200):
                test = centerline[snap_idx] + push_dir * d
                tx, ty = int(round(test[0])), int(round(test[1]))
                if tx < 0 or tx >= w or ty < 0 or ty >= h:
                    new_pt = test
                    break
                if road_mask[ty, tx] <= 127:
                    new_pt = test + push_dir * margin_px
                    break

            obj["position"] = [round(float(new_pt[0]), 1), round(float(new_pt[1]), 1)]
            fixed_count += 1

    if fixed_count:
        logger.info("Fixed %d timing points that were on road surface", fixed_count)
    return timing_objects


# ---------------------------------------------------------------------------
# Convenience: full pipeline from mask to timing points
# ---------------------------------------------------------------------------

def process_road_mask(
    road_mask_path: str,
    track_direction: str = "clockwise",
    time0_idx: Optional[int] = None,
    pixel_size_m: float = 0.3,
    timing_margin_m: float = 3.0,
) -> Dict[str, Any]:
    """End-to-end: road mask image -> centerline + bends + timing objects.

    Returns dict with keys: centerline, bends, timing_objects.
    """
    import cv2 as _cv2
    mask = _cv2.imread(road_mask_path, _cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read road mask: {road_mask_path}")

    return process_road_mask_from_array(
        mask, track_direction,
        time0_idx=time0_idx, pixel_size_m=pixel_size_m,
        timing_margin_m=timing_margin_m,
    )


def process_road_mask_from_array(
    mask: np.ndarray,
    track_direction: str = "clockwise",
    time0_idx: Optional[int] = None,
    pixel_size_m: float = 0.3,
    timing_margin_m: float = 3.0,
) -> Dict[str, Any]:
    """End-to-end: road mask array -> centerline + bends + timing objects.

    Args:
        mask: HxW uint8 grayscale mask (>127 = road).
        track_direction: 'clockwise' or 'counterclockwise'.
        time0_idx: Optional centerline index of TIME_0 (VLM-generated).
            If provided, uses generate_timing_points_from_time0().
            If None, falls back to legacy generate_timing_points().
        pixel_size_m: Meters per pixel (for margin computation).
        timing_margin_m: Push timing L/R beyond road edge by this (meters).

    Returns dict with keys: centerline, bends, timing_objects.
    """
    centerline = extract_centerline(mask, pixel_size_m=pixel_size_m)
    if len(centerline) < 10:
        logger.warning("Centerline too short, skipping bend detection")
        return {"centerline": centerline.tolist(), "bends": [], "timing_objects": []}

    curvature = compute_curvature(centerline)
    bends = detect_composite_bends(centerline, curvature)

    if time0_idx is not None:
        timing_objects, labeled_bends = generate_timing_points_from_time0(
            mask, centerline, curvature, bends, time0_idx,
            track_direction, pixel_size_m, timing_margin_m,
        )
        return {
            "centerline": centerline.tolist(),
            "bends": labeled_bends,
            "timing_objects": timing_objects,
        }
    else:
        timing_objects = generate_timing_points(
            mask, centerline, curvature, bends, track_direction,
            pixel_size_m=pixel_size_m,
        )
        return {
            "centerline": centerline.tolist(),
            "bends": bends,
            "timing_objects": timing_objects,
        }


def regenerate_from_centerline(
    centerline_points: List[List[float]],
    mask: np.ndarray,
    track_direction: str = "clockwise",
    time0_idx: Optional[int] = None,
    pixel_size_m: float = 0.3,
    timing_margin_m: float = 3.0,
) -> Dict[str, Any]:
    """Recompute bends + timing from an edited centerline.

    Called by the centerline editor after the user drags vertices.

    Args:
        centerline_points: List of [x, y] centerline coordinates.
        mask: HxW uint8 grayscale road mask (for track-width measurement).
        track_direction: 'clockwise' or 'counterclockwise'.
        time0_idx: Optional centerline index of TIME_0. When provided,
            uses generate_timing_points_from_time0() for ordered timing.
        pixel_size_m: Meters per pixel (for margin computation).
        timing_margin_m: Push timing L/R beyond road edge by this (meters).

    Returns:
        Dict with keys: bends, timing_objects.
    """
    centerline = np.array(centerline_points, dtype=np.float64)
    if len(centerline) < 10:
        logger.warning("Centerline too short for regeneration (%d points)", len(centerline))
        return {"bends": [], "timing_objects": []}

    curvature = compute_curvature(centerline)
    bends = detect_composite_bends(centerline, curvature)

    if time0_idx is not None:
        timing_objects, labeled_bends = generate_timing_points_from_time0(
            mask, centerline, curvature, bends, time0_idx,
            track_direction, pixel_size_m, timing_margin_m,
        )
        return {
            "bends": labeled_bends,
            "timing_objects": timing_objects,
        }
    else:
        timing_objects = generate_timing_points(
            mask, centerline, curvature, bends, track_direction,
            pixel_size_m=pixel_size_m,
        )
        return {
            "bends": bends,
            "timing_objects": timing_objects,
        }
