"""像素角点边界图算法 — 零间隙多边形提取

Pixel-Corner Boundary Graph algorithm for gap-free polygon extraction from
a composite label map.

Core idea: instead of extracting contours per-tag and trying to match shared
boundaries, we extract a unique boundary graph from the composite label map
in one pass, then build polygons from the graph.

Phase 1: Extract boundary edges (numpy vectorized, O(H×W))
Phase 2: Build chains between junction points
Phase 3: Simplify chains (Douglas-Peucker) + geo-coordinate conversion
Phase 4: Assemble closed rings per label (half-edge DCEL face tracing)
Phase 5: Triangulate (mapbox_earcut)
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Re-use earcut triangulation from mask_merger
from mask_merger import _triangulate_group_earcut, _canvas_to_geo

import json
from pathlib import Path

# Tag ID to name mapping
TAG_ID_TO_NAME = {1: "sand", 2: "grass", 3: "road2", 4: "road", 5: "kerb"}


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _serialize_graph(
    chains: List[dict],
    junctions: Set[Tuple[int, int]],
    labels: Set[int],
    canvas_w: int,
    canvas_h: int,
    bounds: dict,
) -> dict:
    """Serialize boundary graph to JSON-compatible dict."""
    # Sort junctions for stable IDs
    junction_list = sorted(junctions, key=lambda j: (j[1], j[0]))  # (y, x) order
    junction_to_id = {j: i for i, j in enumerate(junction_list)}

    # Build regions map
    label_to_chains: Dict[int, List[int]] = defaultdict(list)
    for ci, chain in enumerate(chains):
        for lbl in chain['labels']:
            if lbl != 0:
                label_to_chains[lbl].append(ci)

    regions = []
    for lbl in sorted(label_to_chains.keys()):
        tag = TAG_ID_TO_NAME.get(lbl, f"unknown_{lbl}")
        regions.append({
            "label": lbl,
            "tag": tag,
            "chain_ids": sorted(label_to_chains[lbl])
        })

    # Serialize chains
    chains_json = []
    for ci, chain in enumerate(chains):
        corners = chain['corners']
        start_j = junction_to_id.get(tuple(corners[0]))
        end_j = junction_to_id.get(tuple(corners[-1]))

        chains_json.append({
            "id": ci,
            "vertices_px": [[int(x), int(y)] for x, y in corners],
            "left_label": int(chain['left_label']),
            "right_label": int(chain['right_label']),
            "start_junction": start_j,
            "end_junction": end_j,
        })

    return {
        "meta": {
            "canvas_w": int(canvas_w),
            "canvas_h": int(canvas_h),
            "bounds_wgs84": {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                             for k, v in bounds.items()},
            "labels": [int(lbl) for lbl in sorted(labels)],
        },
        "junctions": [[int(x), int(y)] for x, y in junction_list],
        "chains": chains_json,
        "regions": regions,
    }


def _deserialize_graph(data: dict) -> Tuple[List[dict], Set[Tuple[int, int]], dict, int, int, Set[int]]:
    """Deserialize boundary graph from JSON dict."""
    meta = data["meta"]
    canvas_w = meta["canvas_w"]
    canvas_h = meta["canvas_h"]
    bounds = meta["bounds_wgs84"]
    labels = set(meta["labels"])

    # Rebuild junctions
    junction_list = [tuple(j) for j in data["junctions"]]
    junctions = set(junction_list)

    # Rebuild chains
    chains = []
    for c in data["chains"]:
        corners = [tuple(v) for v in c["vertices_px"]]
        labels_set = frozenset([c["left_label"], c["right_label"]])

        chain = {
            'corners': corners,
            'labels': labels_set,
            'left_label': c["left_label"],
            'right_label': c["right_label"],
        }
        chains.append(chain)

    return chains, junctions, bounds, canvas_w, canvas_h, labels


# ---------------------------------------------------------------------------
# Phase 1: Extract boundary edges
# ---------------------------------------------------------------------------

def extract_boundary_edges(
    composite: np.ndarray,
) -> np.ndarray:
    """Extract all boundary edges from a composite label map.

    A pixel (x, y) occupies the square from corner (x, y) to (x+1, y+1).
    Canonical directions: vertical edges go DOWN, horizontal edges go RIGHT.

    Returns:
        edges: int32 array of shape (N, 6) — each row is
               (x0, y0, x1, y1, label_a, label_b)
               where (x0,y0)→(x1,y1) is the canonical direction.

    Side convention (screen coords, Y-down):
        When traversing in canonical direction:
        - Vertical (DOWN):  right-of-traversal = label_a (image-left pixel)
        - Horizontal (RIGHT): right-of-traversal = label_b (image-below pixel)
    """
    H, W = composite.shape
    edges_list = []

    # --- Vertical edges (between horizontally adjacent pixels) ---
    left = composite[:, :-1]   # (H, W-1) — pixels at x=0..W-2
    right = composite[:, 1:]   # (H, W-1) — pixels at x=1..W-1
    diff_mask = left != right
    ys, xs = np.where(diff_mask)
    if len(xs) > 0:
        x0 = xs + 1; y0 = ys; x1 = xs + 1; y1 = ys + 1
        edges_list.append(np.stack([x0, y0, x1, y1, left[ys, xs], right[ys, xs]], axis=1))

    # Left canvas edge: pixel (0, y) vs implicit 0
    col0 = composite[:, 0]
    ys = np.where(col0 != 0)[0]
    if len(ys) > 0:
        z = np.zeros(len(ys), dtype=np.int32)
        edges_list.append(np.stack([z, ys, z, ys + 1,
                                    np.zeros(len(ys), dtype=composite.dtype), col0[ys]], axis=1))

    # Right canvas edge: pixel (W-1, y) vs implicit 0
    colW = composite[:, -1]
    ys = np.where(colW != 0)[0]
    if len(ys) > 0:
        w = np.full(len(ys), W, dtype=np.int32)
        edges_list.append(np.stack([w, ys, w, ys + 1,
                                    colW[ys], np.zeros(len(ys), dtype=composite.dtype)], axis=1))

    # --- Horizontal edges (between vertically adjacent pixels) ---
    top = composite[:-1, :]
    bot = composite[1:, :]
    diff_mask = top != bot
    ys, xs = np.where(diff_mask)
    if len(xs) > 0:
        x0 = xs; y0 = ys + 1; x1 = xs + 1; y1 = ys + 1
        edges_list.append(np.stack([x0, y0, x1, y1, top[ys, xs], bot[ys, xs]], axis=1))

    # Top canvas edge
    row0 = composite[0, :]
    xs = np.where(row0 != 0)[0]
    if len(xs) > 0:
        z = np.zeros(len(xs), dtype=np.int32)
        edges_list.append(np.stack([xs, z, xs + 1, z,
                                    np.zeros(len(xs), dtype=composite.dtype), row0[xs]], axis=1))

    # Bottom canvas edge
    rowH = composite[-1, :]
    xs = np.where(rowH != 0)[0]
    if len(xs) > 0:
        h = np.full(len(xs), H, dtype=np.int32)
        edges_list.append(np.stack([xs, h, xs + 1, h,
                                    rowH[xs], np.zeros(len(xs), dtype=composite.dtype)], axis=1))

    if not edges_list:
        return np.zeros((0, 6), dtype=np.int32)
    return np.concatenate(edges_list, axis=0).astype(np.int32)


# ---------------------------------------------------------------------------
# Phase 2: Build chains (with side-label tracking)
# ---------------------------------------------------------------------------

def _build_adjacency(edges: np.ndarray) -> Dict[Tuple[int, int], List[int]]:
    adj: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for i in range(len(edges)):
        e = edges[i]
        adj[(int(e[0]), int(e[1]))].append(i)
        adj[(int(e[2]), int(e[3]))].append(i)
    return adj


def _edge_label_pair(edges: np.ndarray, idx: int) -> frozenset:
    return frozenset((int(edges[idx, 4]), int(edges[idx, 5])))


def _right_label_canonical(edges: np.ndarray, idx: int) -> int:
    """Return the label on the RIGHT side when traversing edge in canonical direction.

    Screen coords (Y-down):
    - Vertical edge going DOWN: right-of-traversal = west pixel = label_a (e[4])
    - Horizontal edge going RIGHT: right-of-traversal = south pixel = label_b (e[5])
    """
    e = edges[idx]
    if int(e[0]) == int(e[2]):  # vertical (x0 == x1)
        return int(e[4])
    else:  # horizontal (y0 == y1)
        return int(e[5])


def build_chains(
    edges: np.ndarray,
) -> Tuple[List[dict], Set[Tuple[int, int]]]:
    """Build chains between junction points.

    Each chain stores:
        - 'corners': list of (x, y) corner coordinates
        - 'labels': frozenset of the two labels this chain separates
        - 'right_label': label on the RIGHT when traversing corners[0]→corners[-1]
                         (in screen coords, Y-down)
    """
    adj = _build_adjacency(edges)

    # Identify junctions
    junctions: Set[Tuple[int, int]] = set()
    for corner, edge_indices in adj.items():
        if len(edge_indices) != 2:
            junctions.add(corner)
        else:
            if _edge_label_pair(edges, edge_indices[0]) != _edge_label_pair(edges, edge_indices[1]):
                junctions.add(corner)

    used_edges: Set[int] = set()
    chains: List[dict] = []

    def _other_corner(edge_idx: int, corner: Tuple[int, int]) -> Tuple[int, int]:
        e = edges[edge_idx]
        c0 = (int(e[0]), int(e[1]))
        c1 = (int(e[2]), int(e[3]))
        return c1 if c0 == corner else c0

    def _compute_right_label(edge_idx: int, from_corner: Tuple[int, int]) -> int:
        """Compute right-of-traversal label when going from from_corner along edge."""
        e = edges[edge_idx]
        canonical_start = (int(e[0]), int(e[1]))
        rl_canonical = _right_label_canonical(edges, edge_idx)
        if from_corner == canonical_start:
            # Going in canonical direction → right label as computed
            return rl_canonical
        else:
            # Going in reverse → left and right swap
            la, lb = int(e[4]), int(e[5])
            return la if rl_canonical == lb else lb

    def _trace_chain(start_corner: Tuple[int, int], start_edge: int):
        chain_corners = [start_corner]
        used_edges.add(start_edge)
        label_pair = _edge_label_pair(edges, start_edge)
        right_label = _compute_right_label(start_edge, start_corner)

        current = _other_corner(start_edge, start_corner)
        chain_corners.append(current)

        while current not in junctions:
            next_edge = None
            for ei in adj[current]:
                if ei not in used_edges and _edge_label_pair(edges, ei) == label_pair:
                    next_edge = ei
                    break
            if next_edge is None:
                break
            used_edges.add(next_edge)
            current = _other_corner(next_edge, current)
            chain_corners.append(current)
            if current == start_corner:
                break

        return chain_corners, label_pair, right_label

    for junction in junctions:
        for ei in adj[junction]:
            if ei not in used_edges:
                corners, lp, rl = _trace_chain(junction, ei)
                labels_set = lp
                left_label = next(iter(labels_set - {rl})) if len(labels_set) > 1 else rl
                chains.append({
                    'corners': corners, 'labels': lp,
                    'right_label': rl, 'left_label': left_label,
                })

    # Closed loops (no junctions)
    for ei in range(len(edges)):
        if ei not in used_edges:
            e = edges[ei]
            start = (int(e[0]), int(e[1]))
            corners, lp, rl = _trace_chain(start, ei)
            labels_set = lp
            left_label = next(iter(labels_set - {rl})) if len(labels_set) > 1 else rl
            chains.append({
                'corners': corners, 'labels': lp,
                'right_label': rl, 'left_label': left_label,
            })

    return chains, junctions


# ---------------------------------------------------------------------------
# Phase 3: Simplify chains + geo conversion
# ---------------------------------------------------------------------------

def simplify_chains(
    chains: List[dict],
    junctions: Set[Tuple[int, int]],
    epsilon: float,
    bounds: dict,
    canvas_w: int,
    canvas_h: int,
) -> List[dict]:
    """Simplify chain geometry with Douglas-Peucker and convert to geo coords.

    Junction endpoints are always preserved (pinned).
    Mutates chains in-place, adding 'geo_coords' and 'simplified_corners'.
    """
    for chain in chains:
        corners = chain['corners']
        if len(corners) < 2:
            chain['geo_coords'] = []
            chain['simplified_corners'] = []
            continue

        is_closed = corners[0] == corners[-1] and len(corners) > 2

        if len(corners) <= 2 or epsilon <= 0:
            simplified = list(corners)
        else:
            pts = np.array(corners, dtype=np.float32).reshape(-1, 1, 2)
            approx = cv2.approxPolyDP(pts, epsilon, closed=is_closed)
            simplified = [tuple(p) for p in approx.reshape(-1, 2).astype(int)]

            if is_closed:
                if simplified[-1] != simplified[0]:
                    simplified.append(simplified[0])
            else:
                if simplified[0] != corners[0]:
                    simplified.insert(0, corners[0])
                if simplified[-1] != corners[-1]:
                    simplified.append(corners[-1])

        geo_coords = []
        for cx, cy in simplified:
            lon, lat = _canvas_to_geo(float(cx), float(cy), bounds, canvas_w, canvas_h)
            geo_coords.append((lon, lat))

        chain['geo_coords'] = geo_coords
        chain['simplified_corners'] = simplified

    return chains


# ---------------------------------------------------------------------------
# Phase 4: Assemble closed rings per label (DCEL half-edge face tracing)
# ---------------------------------------------------------------------------

def _angle_from(dx: float, dy: float) -> float:
    """Angle in [0, 2*pi) from direction vector."""
    a = math.atan2(dy, dx)
    return a if a >= 0 else a + 2 * math.pi


def assemble_rings(
    label: int,
    chains: List[dict],
    junctions: Set[Tuple[int, int]],
) -> List[Tuple[List[Tuple[float, float]], List[List[Tuple[float, float]]]]]:
    """Assemble closed rings for a given label using half-edge face tracing.

    Each chain has a known right_label/left_label for the forward direction.
    For target label L, we follow half-edges where L is on the LEFT side
    (standard DCEL convention: face is on the left of its boundary half-edges).

    At junctions, the "next" half-edge after arriving is determined by:
    twin of arriving half-edge → find it in the angle-sorted outgoing list →
    take the NEXT one in increasing-angle order (CW in screen coords).

    Returns:
        List of (outer_ring, holes).
    """
    # Collect half-edges for this label.
    # A half-edge (ci, True) means traverse chain ci forward.
    # A half-edge (ci, False) means traverse chain ci backward (left/right swap).
    # We want half-edges where `label` is on the LEFT.
    #   forward  + left_label == label → use (ci, True)
    #   backward + right_label == label (since reversing swaps sides) → use (ci, False)

    half_edges: List[Tuple[int, bool]] = []
    for ci, chain in enumerate(chains):
        if label not in chain['labels']:
            continue
        if chain.get('left_label') == label:
            half_edges.append((ci, True))
        if chain.get('right_label') == label:
            half_edges.append((ci, False))

    if not half_edges:
        return []

    # --- Helper functions ---
    def _get_corners(ci: int) -> list:
        return chains[ci].get('simplified_corners', chains[ci]['corners'])

    def _get_original_corners(ci: int) -> list:
        return chains[ci]['corners']

    def _end_junction(ci: int, fwd: bool) -> Tuple[int, int]:
        c = _get_original_corners(ci)
        return tuple(c[-1]) if fwd else tuple(c[0])

    def _outgoing_angle(ci: int, fwd: bool) -> float:
        """Compute outgoing angle using ORIGINAL (unsimplified) corners.

        After Douglas-Peucker simplification, the second point may shift,
        giving a wrong angle. The original pixel-grid corners always have
        axis-aligned directions (0/90/180/270°) at junctions.
        """
        c = _get_original_corners(ci)
        if fwd:
            dx, dy = c[1][0] - c[0][0], c[1][1] - c[0][1]
        else:
            dx, dy = c[-2][0] - c[-1][0], c[-2][1] - c[-1][1]
        return _angle_from(float(dx), float(dy))

    def _get_geo(ci: int, fwd: bool) -> List[Tuple[float, float]]:
        coords = chains[ci]['geo_coords']
        return list(coords) if fwd else list(reversed(coords))

    # --- Build sorted outgoing half-edge lists at each junction ---
    # Use ORIGINAL corners for start/end detection and angle computation.
    junction_all_out: Dict[Tuple[int, int], List[Tuple[float, int, bool]]] = defaultdict(list)
    for ci, chain in enumerate(chains):
        c = _get_original_corners(ci)
        if len(c) < 2:
            continue
        start = tuple(c[0])
        end = tuple(c[-1])
        is_closed = start == end and len(c) > 2
        if is_closed:
            continue
        if start in junctions:
            angle = _outgoing_angle(ci, True)
            junction_all_out[start].append((angle, ci, True))
        if end in junctions:
            angle = _outgoing_angle(ci, False)
            junction_all_out[end].append((angle, ci, False))

    for j in junction_all_out:
        junction_all_out[j].sort()

    # --- Build next-half-edge map ---
    # For half-edge h = (ci, fwd) arriving at end_junction:
    #   twin(h) = (ci, not fwd), which is outgoing at end_junction
    #   next(h) = outgoing half-edge just AFTER twin in angle-sorted order
    #
    # We only care about next(h) for half-edges in our label's set.
    def _find_next(ci: int, fwd: bool) -> Optional[Tuple[int, bool]]:
        """DCEL next-half-edge: immediately after twin in angle-sorted order.

        By the planar subdivision property, the half-edge immediately after
        twin(h) at the destination vertex bounds the SAME face as h.
        No filtering by label — the DCEL guarantees correctness.
        """
        end_j = _end_junction(ci, fwd)
        sorted_out = junction_all_out.get(end_j, [])
        if not sorted_out:
            return None

        twin = (ci, not fwd)
        twin_idx = None
        for i, (_, c, f) in enumerate(sorted_out):
            if c == twin[0] and f == twin[1]:
                twin_idx = i
                break
        if twin_idx is None:
            return None

        # Take IMMEDIATELY next — DCEL guarantees same face
        next_idx = (twin_idx + 1) % len(sorted_out)
        return (sorted_out[next_idx][1], sorted_out[next_idx][2])

    # --- Trace rings ---
    used: Set[Tuple[int, bool]] = set()
    rings_geo: List[List[Tuple[float, float]]] = []

    # 1. Closed-loop chains (no junctions involved)
    for ci, fwd in half_edges:
        c = _get_corners(ci)
        if len(c) < 2:
            continue
        if tuple(c[0]) == tuple(c[-1]) and len(c) > 2:
            ring = _get_geo(ci, fwd)
            if ring and ring[-1] != ring[0]:
                ring.append(ring[0])
            rings_geo.append(ring)
            used.add((ci, fwd))

    # 2. Open chains → trace through junctions
    for start_he in half_edges:
        if start_he in used:
            continue

        ci0, fwd0 = start_he
        c = _get_corners(ci0)
        if len(c) < 2:
            continue
        if tuple(c[0]) == tuple(c[-1]):
            continue  # closed loop handled above

        ring_geo: List[Tuple[float, float]] = []
        current = start_he
        steps = 0

        while current not in used:
            used.add(current)
            seg = _get_geo(current[0], current[1])
            if ring_geo:
                ring_geo.extend(seg[1:])
            else:
                ring_geo.extend(seg)

            nxt = _find_next(current[0], current[1])
            if nxt is None:
                break
            current = nxt
            steps += 1
            if current == start_he:
                break
            if steps > 10000:
                logger.warning("  Ring trace exceeded 10000 steps, aborting")
                break

        if ring_geo and len(ring_geo) >= 3:
            if ring_geo[-1] != ring_geo[0]:
                ring_geo.append(ring_geo[0])
            rings_geo.append(ring_geo)

    if not rings_geo:
        return []

    # --- Classify outer vs hole by signed area ---
    outers: List[List[Tuple[float, float]]] = []
    holes: List[List[Tuple[float, float]]] = []

    for ring in rings_geo:
        area = _signed_area_geo(ring)
        if abs(area) < 1e-15:
            continue
        # In our DCEL, "label on left" tracing in screen coords (Y-down)
        # produces CW outer rings. In geo coords (Y-up), CW → negative area.
        # So: negative area = outer ring, positive = hole.
        if area < 0:
            outers.append(ring)
        else:
            holes.append(ring)

    # Match holes to outers
    result = []
    for outer in outers:
        matched = [h for h in holes if _point_in_ring(h[0], outer)]
        result.append((outer, matched))

    if not outers and holes:
        for hole in holes:
            result.append((list(reversed(hole)), []))

    return result


def _signed_area_geo(ring: List[Tuple[float, float]]) -> float:
    """Signed area via shoelace. Positive = CCW in math coords."""
    area = 0.0
    for i in range(len(ring) - 1):
        x0, y0 = ring[i]
        x1, y1 = ring[i + 1]
        area += (x1 - x0) * (y1 + y0)
    return area / 2.0


def _point_in_ring(point: Tuple[float, float], ring: List[Tuple[float, float]]) -> bool:
    px, py = point
    inside = False
    j = len(ring) - 1
    for i in range(len(ring)):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


# ---------------------------------------------------------------------------
# Three-phase API
# ---------------------------------------------------------------------------

def phase1_extract(
    composite: np.ndarray,
    bounds: dict,
    canvas_w: int,
    canvas_h: int,
    output_path: Optional[str] = None,
) -> dict:
    """Phase 1: Extract boundary graph from composite label map.

    Returns JSON-serializable dict with junctions, chains, and regions.
    Optionally writes to output_path.
    """
    labels = set(np.unique(composite)) - {0}
    edges = extract_boundary_edges(composite)
    chains, junctions = build_chains(edges)

    graph = _serialize_graph(chains, junctions, labels, canvas_w, canvas_h, bounds)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(graph, f, indent=2)

    return graph


def phase2_simplify(
    graph: dict | str,
    epsilon: float,
    output_path: Optional[str] = None,
) -> dict:
    """Phase 2: Simplify chain geometry and convert to geo coordinates.

    Args:
        graph: Phase 1 output dict or path to boundary_graph.json
        epsilon: Douglas-Peucker tolerance in pixels
        output_path: Optional path to write simplified graph

    Returns:
        Simplified graph with vertices_simplified_px and vertices_geo added to each chain.
    """
    if isinstance(graph, str):
        with open(graph, 'r') as f:
            graph = json.load(f)

    chains, junctions, bounds, canvas_w, canvas_h, labels = _deserialize_graph(graph)
    chains = simplify_chains(chains, junctions, epsilon, bounds, canvas_w, canvas_h)

    # Add simplified data to JSON chains
    for ci, chain in enumerate(chains):
        graph["chains"][ci]["vertices_simplified_px"] = [
            [int(x), int(y)] for x, y in chain['simplified_corners']
        ]
        graph["chains"][ci]["vertices_geo"] = [
            [float(lon), float(lat)] for lon, lat in chain['geo_coords']
        ]

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(graph, f, indent=2)

    return graph


def phase3_mesh(simplified: dict | str) -> Dict[int, List[dict]]:
    """Phase 3: Assemble rings and triangulate meshes from simplified graph.

    Args:
        simplified: Phase 2 output dict or path to boundary_graph_simplified.json

    Returns:
        {label: [{'vertices': [(lon, lat), ...], 'faces': [[i0, i1, i2], ...]}]}
    """
    if isinstance(simplified, str):
        with open(simplified, 'r') as f:
            simplified = json.load(f)

    chains, junctions, bounds, canvas_w, canvas_h, labels = _deserialize_graph(simplified)

    # Restore geo_coords and simplified_corners from JSON
    for ci, chain in enumerate(chains):
        json_chain = simplified["chains"][ci]
        chain['geo_coords'] = [tuple(v) for v in json_chain["vertices_geo"]]
        chain['simplified_corners'] = [tuple(v) for v in json_chain["vertices_simplified_px"]]

    result: Dict[int, List[dict]] = {}

    for label in sorted(labels):
        ring_groups = assemble_rings(label, chains, junctions)

        groups = []
        for outer, holes in ring_groups:
            if len(outer) < 3:
                continue

            outer_geo = [[x, y] for x, y in outer]
            holes_geo = [[[x, y] for x, y in h] for h in holes]

            faces = _triangulate_group_earcut(outer_geo, holes_geo)
            if faces is None:
                continue

            all_verts = list(outer)
            for h in holes:
                all_verts.extend(h)

            groups.append({'vertices': all_verts, 'faces': faces})

        if groups:
            result[label] = groups

    return result


# ---------------------------------------------------------------------------
# Phase 5: Triangulation + main entry point
# ---------------------------------------------------------------------------

def extract_all_tag_polygons(
    composite: np.ndarray,
    bounds: dict,
    canvas_w: int,
    canvas_h: int,
    epsilon: float = 1.0,
    debug_dir: Optional[str] = None,
) -> Dict[int, List[dict]]:
    """Main entry point: extract zero-gap polygons for all labels.

    Args:
        composite: (H, W) uint8 label map, 0 = background
        bounds: WGS84 bounds dict
        canvas_w, canvas_h: dimensions of composite
        epsilon: Douglas-Peucker simplification tolerance (pixels)
        debug_dir: Optional directory to save intermediate JSON files

    Returns:
        {tag_id: [{'vertices': [(lon, lat), ...], 'faces': [[i0, i1, i2], ...]}]}
    """
    logger.info("=== Pixel-Corner Boundary Graph Extraction ===")
    logger.info("Canvas: %dx%d, epsilon: %.1f", canvas_w, canvas_h, epsilon)

    # Phase 1: Extract boundary graph
    graph_path = str(Path(debug_dir) / "boundary_graph.json") if debug_dir else None
    graph = phase1_extract(composite, bounds, canvas_w, canvas_h, output_path=graph_path)
    logger.info("Phase 1: %d chains, %d junctions", len(graph["chains"]), len(graph["junctions"]))

    # Phase 2: Simplify
    simplified_path = str(Path(debug_dir) / "boundary_graph_simplified.json") if debug_dir else None
    simplified = phase2_simplify(graph, epsilon, output_path=simplified_path)
    logger.info("Phase 2: Simplified with epsilon=%.1f", epsilon)

    # Phase 3: Mesh
    result = phase3_mesh(simplified)
    logger.info("Phase 3: Generated meshes for %d labels", len(result))

    return result
