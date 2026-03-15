"""Topology-aware contour extraction for zero-gap polygon generation.

Instead of simplifying contours per-tag independently (which causes gaps
at shared boundaries), this module:
1. Extracts shared boundary chains from the composite label map
2. Simplifies each chain ONCE
3. For each tag's raw contour, classifies pixels as shared/non-shared
4. Replaces shared segments with pre-simplified chain coords
5. Both adjacent tags reference the SAME chain coords = zero gap
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("sam3_pipeline.topology_contour")

# Debug output directory
_DEBUG_DIR = None

# Debug info collector: {(tag_id, contour_idx): [{"run_idx": ..., "chain_id": ..., "reversed": ...}, ...]}
_DEBUG_CHAIN_USAGE = {}

def set_debug_dir(path: str):
    """Set directory for debug visualizations."""
    global _DEBUG_DIR, _DEBUG_CHAIN_USAGE
    _DEBUG_DIR = path
    _DEBUG_CHAIN_USAGE = {}  # Clear previous debug info
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SharedChain:
    tag_pair: Tuple[int, int]       # (lower_id, higher_id)
    raw_pixels: np.ndarray          # (N, 2) pixel coords [(x,y), ...]
    simplified_geo: List[List[float]]  # [[lon,lat], ...]
    simplified_px: np.ndarray       # (M, 2) simplified pixel coords


@dataclass
class SharedChainLibrary:
    chains: List[SharedChain] = field(default_factory=list)
    pair_index: Dict[Tuple[int, int], List[int]] = field(default_factory=dict)
    pixel_index: Dict[Tuple[int, int], Tuple[int, int]] = field(default_factory=dict)

    def add_chain(self, chain: SharedChain) -> int:
        idx = len(self.chains)
        self.chains.append(chain)
        pair = chain.tag_pair
        if pair not in self.pair_index:
            self.pair_index[pair] = []
        self.pair_index[pair].append(idx)
        # Index raw pixels for O(1) lookup
        for pos, (px, py) in enumerate(chain.raw_pixels):
            self.pixel_index[(int(px), int(py))] = (idx, pos)
        return idx


# ---------------------------------------------------------------------------
# Build shared chain library
# ---------------------------------------------------------------------------

def build_shared_chain_library(
    composite: np.ndarray,
    bounds: dict,
    canvas_w: int,
    canvas_h: int,
    epsilon: float,
) -> SharedChainLibrary:
    """Extract all shared boundary chains from the composite label map.

    Uses vectorized numpy operations for boundary detection, then
    cv2.findContours to get ordered chains per tag pair.
    """
    from mask_merger import _canvas_to_geo

    lib = SharedChainLibrary()
    h, w = composite.shape

    # Vectorized boundary detection
    # Horizontal boundaries: pixel (x,y) differs from (x+1,y)
    h_left = composite[:, :-1]
    h_right = composite[:, 1:]
    h_diff = (h_left != h_right) & (h_left > 0) & (h_right > 0)

    # Vertical boundaries: pixel (x,y) differs from (x,y+1)
    v_top = composite[:-1, :]
    v_bottom = composite[1:, :]
    v_diff = (v_top != v_bottom) & (v_top > 0) & (v_bottom > 0)

    # Collect all tag pairs
    tag_pairs = set()

    h_ys, h_xs = np.where(h_diff)
    for i in range(len(h_ys)):
        t1, t2 = int(h_left[h_ys[i], h_xs[i]]), int(h_right[h_ys[i], h_xs[i]])
        tag_pairs.add((min(t1, t2), max(t1, t2)))

    v_ys, v_xs = np.where(v_diff)
    for i in range(len(v_ys)):
        t1, t2 = int(v_top[v_ys[i], v_xs[i]]), int(v_bottom[v_ys[i], v_xs[i]])
        tag_pairs.add((min(t1, t2), max(t1, t2)))

    logger.info("  Found %d tag pairs with shared boundaries", len(tag_pairs))

    # For each tag pair, build boundary mask and extract chains
    for pair in sorted(tag_pairs):
        lo, hi = pair
        # Boundary pixels: on the lower-tag side of horizontal/vertical edges
        boundary_mask = np.zeros((h, w), dtype=np.uint8)

        # Horizontal edges: lower-tag side pixel
        h_match = h_diff & (
            ((h_left == lo) & (h_right == hi)) |
            ((h_left == hi) & (h_right == lo))
        )
        h_match_ys, h_match_xs = np.where(h_match)
        for i in range(len(h_match_ys)):
            y, x = h_match_ys[i], h_match_xs[i]
            # Mark both pixels at the boundary
            boundary_mask[y, x] = 255
            boundary_mask[y, x + 1] = 255

        # Vertical edges: mark both pixels
        v_match = v_diff & (
            ((v_top == lo) & (v_bottom == hi)) |
            ((v_top == hi) & (v_bottom == lo))
        )
        v_match_ys, v_match_xs = np.where(v_match)
        for i in range(len(v_match_ys)):
            y, x = v_match_ys[i], v_match_xs[i]
            boundary_mask[y, x] = 255
            boundary_mask[y + 1, x] = 255

        n_boundary = int(np.count_nonzero(boundary_mask))
        if n_boundary < 3:
            continue

        # Use findContours to get ordered chains
        contours, _ = cv2.findContours(
            boundary_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
        )

        for contour in contours:
            if len(contour) < 3:
                continue
            raw_px = contour.reshape(-1, 2)  # (N, 2) as (x, y)

            # Simplify
            approx = cv2.approxPolyDP(contour, epsilon, closed=False)
            simplified_px = approx.reshape(-1, 2)

            # Convert simplified to geo
            simplified_geo = []
            for px, py in simplified_px:
                lon, lat = _canvas_to_geo(float(px), float(py), bounds, canvas_w, canvas_h)
                simplified_geo.append([lon, lat])

            chain = SharedChain(
                tag_pair=pair,
                raw_pixels=raw_px,
                simplified_geo=simplified_geo,
                simplified_px=simplified_px,
            )
            lib.add_chain(chain)

        logger.info(
            "  Pair (%d,%d): %d boundary px, %d chains",
            lo, hi, n_boundary, len(lib.pair_index.get(pair, [])),
        )

    logger.info("  Total chains: %d", len(lib.chains))
    return lib


# ---------------------------------------------------------------------------
# Contour assembly with shared boundary replacement
# ---------------------------------------------------------------------------

def _classify_contour_pixels(
    raw_pixels: np.ndarray,
    tag_id: int,
    composite: np.ndarray,
) -> np.ndarray:
    """Classify each contour pixel as shared or non-shared.

    Returns array of neighbor labels (0 = non-shared, N = shared with tag N).
    For each pixel, check 4-neighbors for a different non-zero tag.
    """
    h, w = composite.shape
    n = len(raw_pixels)
    neighbor_labels = np.zeros(n, dtype=np.int32)

    for i in range(n):
        x, y = int(raw_pixels[i, 0]), int(raw_pixels[i, 1])
        best_neighbor = 0
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h:
                nval = int(composite[ny, nx])
                if nval != 0 and nval != tag_id:
                    # Prefer highest priority (highest id) neighbor
                    if nval > best_neighbor:
                        best_neighbor = nval
        neighbor_labels[i] = best_neighbor

    return neighbor_labels


def _partition_into_runs(
    neighbor_labels: np.ndarray,
) -> List[Tuple[int, int, int]]:
    """Partition contour pixels into runs of consecutive same-label pixels.

    Returns list of (start_idx, end_idx, label) tuples.
    end_idx is inclusive.
    """
    n = len(neighbor_labels)
    if n == 0:
        return []

    runs = []
    start = 0
    current_label = neighbor_labels[0]

    for i in range(1, n):
        if neighbor_labels[i] != current_label:
            runs.append((start, i - 1, int(current_label)))
            start = i
            current_label = neighbor_labels[i]
    runs.append((start, n - 1, int(current_label)))

    return runs


def _merge_short_runs(
    runs: List[Tuple[int, int, int]],
    min_length: int = 3,
) -> List[Tuple[int, int, int]]:
    """Merge very short runs into adjacent longer runs."""
    if len(runs) <= 1:
        return runs

    merged = list(runs)
    changed = True
    while changed:
        changed = False
        new_merged = []
        i = 0
        while i < len(merged):
            start, end, label = merged[i]
            length = end - start + 1

            if length < min_length and len(new_merged) > 0:
                # Merge into previous run
                prev_start, prev_end, prev_label = new_merged[-1]
                new_merged[-1] = (prev_start, end, prev_label)
                changed = True
            elif length < min_length and i + 1 < len(merged):
                # Merge into next run
                next_start, next_end, next_label = merged[i + 1]
                merged[i + 1] = (start, next_end, next_label)
                changed = True
            else:
                new_merged.append((start, end, label))
            i += 1
        merged = new_merged

    return merged


def _find_matching_chain(
    run_pixels: np.ndarray,
    tag_id: int,
    neighbor_label: int,
    chain_lib: SharedChainLibrary,
) -> Optional[Tuple[SharedChain, bool]]:
    """Find the matching shared chain for a run of boundary pixels.

    Returns (chain, reversed) or None.
    """
    pair = (min(tag_id, neighbor_label), max(tag_id, neighbor_label))
    chain_indices = chain_lib.pair_index.get(pair, [])

    if not chain_indices:
        return None

    run_start = (int(run_pixels[0, 0]), int(run_pixels[0, 1]))
    run_end = (int(run_pixels[-1, 0]), int(run_pixels[-1, 1]))

    best_chain = None
    best_score = float('inf')
    best_reversed = False

    for cidx in chain_indices:
        chain = chain_lib.chains[cidx]
        chain_start = (int(chain.raw_pixels[0, 0]), int(chain.raw_pixels[0, 1]))
        chain_end = (int(chain.raw_pixels[-1, 0]), int(chain.raw_pixels[-1, 1]))

        # Dense overlap check: sample every 5 pixels or all if short
        chain_pixel_set = set(
            (int(chain.raw_pixels[j, 0]), int(chain.raw_pixels[j, 1]))
            for j in range(len(chain.raw_pixels))
        )

        step = max(1, len(run_pixels) // 50)  # Denser sampling
        overlap = 0
        for k in range(0, len(run_pixels), step):
            px = (int(run_pixels[k, 0]), int(run_pixels[k, 1]))
            if px in chain_pixel_set:
                overlap += 1
            else:
                # Check 3px radius
                found = False
                for ddx in range(-3, 4):
                    for ddy in range(-3, 4):
                        if (px[0] + ddx, px[1] + ddy) in chain_pixel_set:
                            overlap += 1
                            found = True
                            break
                    if found:
                        break

        sample_count = len(range(0, len(run_pixels), step))
        if sample_count == 0:
            continue
        overlap_ratio = overlap / sample_count

        if overlap_ratio < 0.6:  # Stricter threshold
            continue

        # Determine direction
        dist_fwd_start = abs(run_start[0] - chain_start[0]) + abs(run_start[1] - chain_start[1])
        dist_fwd_end = abs(run_end[0] - chain_end[0]) + abs(run_end[1] - chain_end[1])
        dist_rev_start = abs(run_start[0] - chain_end[0]) + abs(run_start[1] - chain_end[1])
        dist_rev_end = abs(run_end[0] - chain_start[0]) + abs(run_end[1] - chain_start[1])

        is_reversed = dist_rev_start + dist_rev_end < dist_fwd_start + dist_fwd_end

        # Endpoint distance check
        max_endpoint_dist = max(dist_fwd_start, dist_fwd_end) if not is_reversed else max(dist_rev_start, dist_rev_end)

        chain_len = len(chain.raw_pixels)
        run_len = len(run_pixels)
        length_ratio = min(chain_len, run_len) / max(chain_len, run_len, 1)

        # Normalized endpoint distance
        normalized_dist = max_endpoint_dist / max(chain_len, 1)

        # Reject if endpoint too far (relative to chain length)
        if normalized_dist > 0.2:  # 20% of chain length
            continue

        # Quality score: weighted combination (lower is better)
        quality_score = (1 - overlap_ratio) * 3.0 + normalized_dist * 2.0 + (1 - length_ratio) * 1.0

        if quality_score < best_score:
            best_score = quality_score
            best_chain = chain
            best_reversed = is_reversed

    if best_chain is not None:
        logger.debug(f"    Matched chain for tag {tag_id}<->{neighbor_label}: score={best_score:.3f}, reversed={best_reversed}, run_len={len(run_pixels)}")
        return (best_chain, best_reversed)
    logger.debug(f"    No chain match for tag {tag_id}<->{neighbor_label}, run_len={len(run_pixels)}")
    return None


def _assemble_contour(
    raw_pixels: np.ndarray,
    tag_id: int,
    composite: np.ndarray,
    chain_lib: SharedChainLibrary,
    bounds: dict,
    canvas_w: int,
    canvas_h: int,
    epsilon: float,
    contour_idx: int = -1,
) -> List[List[float]]:
    """Assemble a contour by replacing shared segments with chain coords.

    1. Classify each pixel as shared/non-shared
    2. Partition into runs
    3. For shared runs, use pre-simplified chain geo coords
    4. For non-shared runs, simplify independently
    5. Concatenate all segments
    """
    from mask_merger import _canvas_to_geo

    n = len(raw_pixels)
    if n < 3:
        return []

    # Classify pixels
    neighbor_labels = _classify_contour_pixels(raw_pixels, tag_id, composite)

    # Partition into runs
    runs = _partition_into_runs(neighbor_labels)

    # Merge short runs
    runs = _merge_short_runs(runs, min_length=3)

    # Assemble geo coordinates
    result_geo = []

    # Debug: record chain usage
    debug_key = (tag_id, contour_idx)
    if _DEBUG_DIR and debug_key not in _DEBUG_CHAIN_USAGE:
        _DEBUG_CHAIN_USAGE[debug_key] = []

    for run_idx, (start, end, label) in enumerate(runs):
        run_pixels = raw_pixels[start:end + 1]

        if label != 0:
            # Shared run: find matching chain
            match = _find_matching_chain(run_pixels, tag_id, label, chain_lib)
            if match is not None:
                chain, is_reversed = match
                geo_pts = list(chain.simplified_geo)
                if is_reversed:
                    geo_pts = list(reversed(geo_pts))
                result_geo.extend(geo_pts)

                # Record chain usage
                if _DEBUG_DIR:
                    # Find chain_id by object identity (not value equality)
                    chain_id = next((idx for idx, c in enumerate(chain_lib.chains) if c is chain), -1)
                    _DEBUG_CHAIN_USAGE[debug_key].append({
                        "run_idx": run_idx,
                        "run_start": start,
                        "run_end": end,
                        "neighbor_label": int(label),
                        "chain_id": chain_id,
                        "reversed": is_reversed,
                    })
                continue
            else:
                logger.debug(
                    "  No matching chain for tag %d, neighbor %d, run len=%d. Falling back.",
                    tag_id, label, len(run_pixels),
                )

        # Non-shared run (or fallback): simplify independently
        if len(run_pixels) < 2:
            for px, py in run_pixels:
                lon, lat = _canvas_to_geo(float(px), float(py), bounds, canvas_w, canvas_h)
                result_geo.append([lon, lat])
        else:
            contour_arr = run_pixels.reshape(-1, 1, 2).astype(np.float32)
            approx = cv2.approxPolyDP(contour_arr, epsilon, closed=False)
            for pt in approx:
                px, py = float(pt[0][0]), float(pt[0][1])
                lon, lat = _canvas_to_geo(px, py, bounds, canvas_w, canvas_h)
                result_geo.append([lon, lat])

    return result_geo


def _save_self_intersecting_contour_debug(
    tag_id: int, contour_idx: int, geo_pts: List[List[float]],
    bounds: dict, canvas_w: int, canvas_h: int,
    raw_pixels: np.ndarray = None
):
    """Save debug visualization of self-intersecting contour."""
    if _DEBUG_DIR is None:
        return

    try:
        # Convert geo to canvas pixels for visualization
        from mask_merger import _geo_to_canvas
        canvas_pts = []
        for lon, lat in geo_pts:
            x, y = _geo_to_canvas(lon, lat, bounds, canvas_w, canvas_h)
            canvas_pts.append((int(x), int(y)))

        # Create debug image
        img = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        # Draw raw contour if available (in blue)
        if raw_pixels is not None and len(raw_pixels) > 0:
            raw_pts = np.array(raw_pixels, dtype=np.int32)
            cv2.polylines(img, [raw_pts], True, (255, 0, 0), 1)  # Blue: raw

        # Draw simplified contour (in green)
        pts = np.array(canvas_pts, dtype=np.int32)
        cv2.polylines(img, [pts], True, (0, 255, 0), 2)  # Green: simplified

        # Mark vertices with small circles
        for i, pt in enumerate(canvas_pts):
            color = (0, 0, 255) if i == 0 else (255, 0, 0) if i == len(canvas_pts)-1 else (0, 255, 255)
            cv2.circle(img, pt, 3, color, -1)

        # Draw line segments with numbers to show order
        for i in range(len(canvas_pts)):
            p1 = canvas_pts[i]
            p2 = canvas_pts[(i+1) % len(canvas_pts)]
            # Draw segment number at midpoint
            mid = ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2)
            if i % 5 == 0:  # Only label every 5th segment to avoid clutter
                cv2.putText(img, str(i), mid, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)

        # Save
        filename = f"self_intersect_tag{tag_id}_contour{contour_idx}.png"
        cv2.imwrite(os.path.join(_DEBUG_DIR, filename), img)
        logger.debug(f"    Saved debug image: {filename} (raw={raw_pixels is not None}, simplified={len(canvas_pts)} pts)")
    except Exception as e:
        logger.debug(f"    Failed to save debug image: {e}")


def _export_topology_debug(
    chain_lib: SharedChainLibrary,
    result: Dict[int, List[dict]],
    tag_name_to_id: dict,
) -> None:
    """Export topology debug information to JSON."""
    if _DEBUG_DIR is None:
        logger.warning("  Topology debug: _DEBUG_DIR is None, skipping export")
        return

    logger.info(f"  Exporting topology debug to {_DEBUG_DIR}...")

    try:
        id_to_name = {v: k for k, v in tag_name_to_id.items()}
        logger.info(f"  Processing {len(chain_lib.chains)} chains and {len(result)} tags")

        # Export shared chains
        chains_data = []
        for idx, chain in enumerate(chain_lib.chains):
            tag1, tag2 = chain.tag_pair
            chains_data.append({
                "chain_id": idx,
                "tag_pair": [id_to_name.get(tag1, f"tag{tag1}"), id_to_name.get(tag2, f"tag{tag2}")],
                "raw_pixel_count": len(chain.raw_pixels),
                "simplified_vertex_count": len(chain.simplified_geo),
                "simplified_vertices": chain.simplified_geo,
            })

        # Export tag contours
        tags_data = []
        for tag_id, groups in result.items():
            tag_name = id_to_name.get(tag_id, f"tag{tag_id}")
            groups_data = []
            for gidx, group in enumerate(groups):
                groups_data.append({
                    "group_id": gidx,
                    "vertex_count": len(group.get("vertices", [])),
                    "face_count": len(group.get("faces", [])),
                    "vertices": group.get("vertices", []),
                })
            tags_data.append({
                "tag_id": tag_id,
                "tag_name": tag_name,
                "group_count": len(groups),
                "groups": groups_data,
            })

        # Export chain usage info
        chain_usage_data = []
        for (tag_id, contour_idx), usage_list in _DEBUG_CHAIN_USAGE.items():
            tag_name = id_to_name.get(tag_id, f"tag{tag_id}")
            chain_usage_data.append({
                "tag_id": tag_id,
                "tag_name": tag_name,
                "contour_idx": contour_idx,
                "chain_usage": usage_list,
            })

        output = {
            "shared_chains": chains_data,
            "tags": tags_data,
            "chain_usage": chain_usage_data,
        }

        output_path = os.path.join(_DEBUG_DIR, "topology_debug.json")
        logger.info(f"  Writing to {output_path}...")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        file_size = os.path.getsize(output_path)
        logger.info(f"  Topology debug info saved: {output_path} ({file_size} bytes)")
    except Exception as e:
        import traceback
        logger.error(f"  Failed to export topology debug: {e}")
        logger.error(traceback.format_exc())



# ---------------------------------------------------------------------------
# Single tag extraction
# ---------------------------------------------------------------------------

def _extract_single_tag_contours(
    binary: np.ndarray,
    tag_id: int,
    composite: np.ndarray,
    chain_lib: SharedChainLibrary,
    bounds: dict,
    canvas_w: int,
    canvas_h: int,
    epsilon: float,
    min_contour_area: int,
) -> List[dict]:
    """Extract contours for a single tag, using shared chains for boundaries.

    Returns list of groups with 'vertices' and 'faces'.
    """
    from mask_merger import _triangulate_group_earcut, _check_self_intersection

    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )

    if not contours or hierarchy is None:
        return []

    hier = hierarchy[0]
    groups = []
    outer_to_group: Dict[int, int] = {}

    # Assemble geo contours for each valid contour
    geo_contours: Dict[int, List[List[float]]] = {}

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < min_contour_area:
            continue

        raw_px = contour.reshape(-1, 2)
        geo_pts = _assemble_contour(
            raw_px, tag_id, composite, chain_lib,
            bounds, canvas_w, canvas_h, epsilon, contour_idx=i,
        )

        if len(geo_pts) < 3:
            continue

        # Check self-intersection; if found, try to fix
        if _check_self_intersection(geo_pts):
            logger.debug("  Self-intersection detected for tag %d contour %d (len=%d), retrying with eps/2", tag_id, i, len(geo_pts))

            # Debug: save problematic contour info
            logger.debug("    Contour bounds: x=[%.6f, %.6f], y=[%.6f, %.6f]",
                        min(p[0] for p in geo_pts), max(p[0] for p in geo_pts),
                        min(p[1] for p in geo_pts), max(p[1] for p in geo_pts))

            geo_pts = _assemble_contour(
                raw_px, tag_id, composite, chain_lib,
                bounds, canvas_w, canvas_h, epsilon / 2, contour_idx=i,
            )
            if len(geo_pts) < 3 or _check_self_intersection(geo_pts):
                logger.debug("    Still self-intersecting after eps/2, trying Shapely fix...")

                # Save debug visualization with raw contour
                _save_self_intersecting_contour_debug(tag_id, i, geo_pts, bounds, canvas_w, canvas_h, raw_px)

                # Try aggressive Shapely fix
                try:
                    from shapely.geometry import Polygon
                    from shapely import make_valid
                    poly = Polygon([(p[0], p[1]) for p in geo_pts])

                    # Try make_valid first
                    fixed = make_valid(poly)
                    if fixed.geom_type == 'Polygon' and fixed.is_valid:
                        geo_pts = [[x, y] for x, y in fixed.exterior.coords[:-1]]
                        if not _check_self_intersection(geo_pts):
                            logger.debug("    ✓ Fixed contour %d with make_valid", i)
                        else:
                            # Try aggressive buffer
                            fixed = poly.buffer(-0.00001).buffer(0.00001)
                            if fixed.geom_type == 'Polygon' and fixed.is_valid:
                                geo_pts = [[x, y] for x, y in fixed.exterior.coords[:-1]]
                                logger.debug("    ✓ Fixed contour %d with buffer smoothing", i)
                            else:
                                logger.warning("    ✗ Contour %d still self-intersecting after all fixes (len=%d)", i, len(geo_pts))
                    else:
                        logger.warning("    ✗ Contour %d still self-intersecting after Shapely fix", i)
                except Exception as e:
                    logger.warning("    ✗ Contour %d Shapely fix failed: %s", i, e)

        geo_contours[i] = geo_pts

    # Build groups from hierarchy
    for i in geo_contours:
        if hier[i][3] == -1:  # outer contour (no parent)
            gidx = len(groups)
            outer_to_group[i] = gidx
            groups.append({
                "include": [geo_contours[i]],
                "exclude": [],
                "vertices": None,
                "faces": None,
            })

    # Attach holes
    for i in geo_contours:
        parent = hier[i][3]
        if parent >= 0 and parent in outer_to_group:
            gidx = outer_to_group[parent]
            groups[gidx]["exclude"].append(geo_contours[i])

    # Triangulate
    total_faces = 0
    for group in groups:
        outer = group["include"][0]
        holes = group["exclude"]
        faces = _triangulate_group_earcut(outer, holes)
        if faces is not None:
            all_verts = list(outer)
            for hole in holes:
                all_verts.extend(hole)
            group["vertices"] = all_verts
            group["faces"] = faces
            total_faces += len(faces)

    logger.info(
        "  Tag %d: %d groups, %d total faces",
        tag_id, len(groups), total_faces,
    )
    return groups


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def extract_all_tag_contours(
    composite: np.ndarray,
    bounds: dict,
    canvas_w: int,
    canvas_h: int,
    simplify_epsilon: float = 1.0,
    min_contour_area: int = 100,
) -> Dict[int, List[dict]]:
    """Extract contours for all tags with topology-aware shared boundaries.

    Returns Dict[tag_id, List[group_dict]] where each group has
    'vertices' (List[[lon,lat]]) and 'faces' (List[[i,j,k]]).
    """
    from mask_gap_filler import SURFACE_TAGS, TAG_NAME_TO_ID

    logger.info("Building shared chain library...")
    chain_lib = build_shared_chain_library(
        composite, bounds, canvas_w, canvas_h, simplify_epsilon,
    )

    result: Dict[int, List[dict]] = {}

    for tag_name in SURFACE_TAGS:
        tag_id = TAG_NAME_TO_ID[tag_name]
        binary = (composite == tag_id).astype(np.uint8) * 255
        n_pixels = int(np.count_nonzero(binary))
        if n_pixels == 0:
            continue

        logger.info("Extracting tag '%s' (id=%d, %d px)...", tag_name, tag_id, n_pixels)
        groups = _extract_single_tag_contours(
            binary, tag_id, composite, chain_lib,
            bounds, canvas_w, canvas_h,
            simplify_epsilon, min_contour_area,
        )

        # Filter out groups without valid triangulation
        valid_groups = [g for g in groups if g.get("vertices") and g.get("faces")]
        if valid_groups:
            result[tag_id] = valid_groups
            logger.info(
                "  Tag '%s': %d valid groups (%d total verts, %d total faces)",
                tag_name, len(valid_groups),
                sum(len(g["vertices"]) for g in valid_groups),
                sum(len(g["faces"]) for g in valid_groups),
            )

    # Export topology debug info
    logger.info(f"Checking topology debug export: _DEBUG_DIR={_DEBUG_DIR}")
    if _DEBUG_DIR:
        _export_topology_debug(chain_lib, result, TAG_NAME_TO_ID)
    else:
        logger.warning("Skipping topology debug export: _DEBUG_DIR is None")

    return result
