"""
Shared Boundary Extractor for Stage 8 Mask Gap Elimination.

Extracts and vectorizes shared boundaries between adjacent mask regions,
ensuring that neighboring polygons share identical boundary coordinates.
"""

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict
import numpy as np
import cv2


@dataclass
class SharedBoundary:
    """Represents a shared boundary between two adjacent tags."""
    tag_pair: Tuple[int, int]  # (tag1, tag2), tag1 < tag2
    pixel_chain: List[Tuple[int, int]]  # Original pixel coordinates
    simplified_pixels: List[Tuple[int, int]]  # Simplified pixel coordinates
    geo_coords: List[Tuple[float, float]]  # Geographic coordinates (from simplified)
    length_pixels: float  # Boundary length in pixels

    def get_for_tag(self, tag: int) -> List[Tuple[float, float]]:
        """Get boundary coordinates for a specific tag (may reverse)."""
        if tag == self.tag_pair[0]:
            return self.geo_coords
        else:
            return list(reversed(self.geo_coords))

    def get_pixels_for_tag(self, tag: int) -> List[Tuple[int, int]]:
        """Get simplified pixel coordinates for a specific tag (may reverse)."""
        if tag == self.tag_pair[0]:
            return self.simplified_pixels
        else:
            return list(reversed(self.simplified_pixels))


class SharedBoundaryLibrary:
    """Library of shared boundaries with fast tag-based lookup."""

    def __init__(self):
        self.boundaries: List[SharedBoundary] = []
        self._tag_index: Dict[int, List[int]] = defaultdict(list)

    def add(self, boundary: SharedBoundary):
        """Add a boundary to the library."""
        idx = len(self.boundaries)
        self.boundaries.append(boundary)
        self._tag_index[boundary.tag_pair[0]].append(idx)
        self._tag_index[boundary.tag_pair[1]].append(idx)

    def get_boundaries_for_tag(self, tag: int) -> List[SharedBoundary]:
        """Get all boundaries involving the specified tag."""
        return [self.boundaries[i] for i in self._tag_index[tag]]


def detect_boundary_pixels(composite: np.ndarray) -> Dict[Tuple[int, int], Set[Tuple[int, int]]]:
    """
    Detect boundary pixels between adjacent tags using 4-connectivity.

    Args:
        composite: Label map where each pixel contains a tag ID

    Returns:
        Dict mapping (tag1, tag2) pairs to sets of boundary pixel coordinates
    """
    h, w = composite.shape
    boundaries = defaultdict(set)

    for y in range(h):
        for x in range(w):
            tag1 = int(composite[y, x])
            if tag1 == 0:
                continue

            # Check right neighbor
            if x + 1 < w:
                tag2 = int(composite[y, x + 1])
                if tag2 != 0 and tag2 != tag1:
                    pair = tuple(sorted([tag1, tag2]))
                    boundaries[pair].add((x, y))
                    boundaries[pair].add((x + 1, y))

            # Check bottom neighbor
            if y + 1 < h:
                tag2 = int(composite[y + 1, x])
                if tag2 != 0 and tag2 != tag1:
                    pair = tuple(sorted([tag1, tag2]))
                    boundaries[pair].add((x, y))
                    boundaries[pair].add((x, y + 1))

    return boundaries


def trace_boundary_chains(
    boundary_pixels: Set[Tuple[int, int]],
    min_chain_length: int = 3
) -> List[List[Tuple[int, int]]]:
    """
    Trace boundary pixels into continuous chains using 8-connectivity.

    Args:
        boundary_pixels: Set of boundary pixel coordinates
        min_chain_length: Minimum chain length to keep

    Returns:
        List of chains, each chain is a list of (x, y) coordinates
    """
    visited = set()
    chains = []

    def get_neighbors(x: int, y: int) -> List[Tuple[int, int]]:
        """Get 8-connected neighbors that are boundary pixels."""
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if (nx, ny) in boundary_pixels and (nx, ny) not in visited:
                    neighbors.append((nx, ny))
        return neighbors

    for start_pixel in boundary_pixels:
        if start_pixel in visited:
            continue

        # DFS to trace the chain
        chain = [start_pixel]
        visited.add(start_pixel)
        current = start_pixel

        while True:
            neighbors = get_neighbors(*current)
            if not neighbors:
                break
            # Pick the first available neighbor
            next_pixel = neighbors[0]
            chain.append(next_pixel)
            visited.add(next_pixel)
            current = next_pixel

        if len(chain) >= min_chain_length:
            chains.append(chain)

    return chains


def _canvas_to_geo(
    x: float, y: float,
    bounds: Dict[str, float],
    canvas_w: int,
    canvas_h: int
) -> Tuple[float, float]:
    """Convert canvas pixel coordinates to geographic coordinates."""
    left = bounds["left"]
    right = bounds["right"]
    top = bounds["top"]
    bottom = bounds["bottom"]

    lon = left + (x / canvas_w) * (right - left)
    lat = top - (y / canvas_h) * (top - bottom)
    return (lon, lat)


def simplify_and_vectorize_boundary(
    chain: List[Tuple[int, int]],
    bounds: Dict[str, float],
    canvas_w: int,
    canvas_h: int,
    epsilon: float = 1.0
) -> Tuple[List[Tuple[int, int]], List[Tuple[float, float]]]:
    """
    Simplify boundary chain using Douglas-Peucker and convert to geo coordinates.

    Args:
        chain: Pixel coordinate chain
        bounds: Geographic bounds
        canvas_w, canvas_h: Canvas dimensions
        epsilon: Simplification threshold in pixels

    Returns:
        Tuple of (simplified_pixels, geo_coords)
    """
    chain_array = np.array(chain, dtype=np.float32).reshape(-1, 1, 2)
    simplified = cv2.approxPolyDP(chain_array, epsilon, closed=False)

    simplified_pixels = []
    geo_coords = []
    for pt in simplified:
        x, y = float(pt[0][0]), float(pt[0][1])
        simplified_pixels.append((int(round(x)), int(round(y))))
        lon, lat = _canvas_to_geo(x, y, bounds, canvas_w, canvas_h)
        geo_coords.append((lon, lat))

    return simplified_pixels, geo_coords


def extract_all_shared_boundaries(
    composite: np.ndarray,
    bounds: Dict[str, float],
    canvas_w: int,
    canvas_h: int,
    simplify_epsilon: float = 1.0,
    min_chain_length: int = 3,
    logger=None
) -> SharedBoundaryLibrary:
    """
    Extract all shared boundaries from a composite label map.

    Args:
        composite: Label map with tag IDs
        bounds: Geographic bounds
        canvas_w, canvas_h: Canvas dimensions
        simplify_epsilon: Boundary simplification threshold (pixels)
        min_chain_length: Minimum chain length to keep
        logger: Optional logger for detailed output

    Returns:
        SharedBoundaryLibrary containing all extracted boundaries
    """
    import logging
    if logger is None:
        logger = logging.getLogger(__name__)

    library = SharedBoundaryLibrary()

    # Step 1: Detect boundary pixels
    logger.info("  Step 1: Detecting boundary pixels...")
    boundary_pixel_map = detect_boundary_pixels(composite)
    logger.info("    Found %d tag pairs with boundaries", len(boundary_pixel_map))

    for tag_pair, pixels in boundary_pixel_map.items():
        logger.info("    Tag pair %s: %d boundary pixels", tag_pair, len(pixels))

    # Step 2: Trace and vectorize each tag pair's boundaries
    logger.info("  Step 2: Tracing boundary chains...")
    total_chains = 0
    for tag_pair, pixels in boundary_pixel_map.items():
        chains = trace_boundary_chains(pixels, min_chain_length)
        logger.info("    Tag pair %s: %d chains", tag_pair, len(chains))
        total_chains += len(chains)

        for chain_idx, chain in enumerate(chains):
            # Simplify and convert to geo coordinates
            simplified_pixels, geo_coords = simplify_and_vectorize_boundary(
                chain, bounds, canvas_w, canvas_h, simplify_epsilon
            )

            if len(geo_coords) >= 2:
                length = len(chain)
                boundary = SharedBoundary(
                    tag_pair=tag_pair,
                    pixel_chain=chain,
                    simplified_pixels=simplified_pixels,
                    geo_coords=geo_coords,
                    length_pixels=float(length)
                )
                library.add(boundary)
                logger.debug("      Chain %d: %d pixels -> %d geo points",
                           chain_idx, len(chain), len(geo_coords))

    logger.info("  Total: %d boundaries from %d chains", len(library.boundaries), total_chains)
    return library


def visualize_shared_boundaries(
    composite: np.ndarray,
    library: SharedBoundaryLibrary,
    output_path: str
) -> None:
    """
    Visualize shared boundaries on the composite image.

    Args:
        composite: Label map with tag IDs
        library: Extracted boundaries
        output_path: Path to save visualization image
    """
    import cv2

    # Create RGB visualization
    h, w = composite.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)

    # Color map for tags
    tag_colors = {
        1: (100, 100, 255),  # road - blue
        2: (255, 200, 100),  # kerb - orange
        3: (100, 255, 100),  # grass - green
        4: (255, 255, 150),  # sand - yellow
        5: (150, 150, 255),  # road2 - light blue
    }

    # Draw composite with colors
    for tag_id, color in tag_colors.items():
        mask = (composite == tag_id)
        vis[mask] = color

    # Draw boundaries in bright red
    for boundary in library.boundaries:
        for x, y in boundary.pixel_chain:
            if 0 <= x < w and 0 <= y < h:
                vis[y, x] = (0, 0, 255)  # Red

    cv2.imwrite(output_path, vis)


def visualize_boundary_matches(
    composite: np.ndarray,
    tag_id: int,
    contour_geo: List[Tuple[float, float]],
    matches: List[Tuple[int, int, SharedBoundary]],
    bounds: Dict[str, float],
    canvas_w: int,
    canvas_h: int,
    output_path: str
) -> None:
    """
    Visualize which contour segments matched shared boundaries.

    Args:
        composite: Label map
        tag_id: Current tag ID
        contour_geo: Contour in geographic coordinates
        matches: Matched segments
        bounds: Geographic bounds
        canvas_w, canvas_h: Canvas dimensions
        output_path: Path to save visualization
    """
    import cv2

    # Create visualization
    h, w = composite.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)

    # Draw current tag in gray
    mask = (composite == tag_id)
    vis[mask] = (100, 100, 100)

    # Convert contour back to canvas coordinates and draw
    def geo_to_canvas(lon: float, lat: float) -> Tuple[int, int]:
        left, right = bounds["left"], bounds["right"]
        top, bottom = bounds["top"], bounds["bottom"]
        x = int((lon - left) / (right - left) * canvas_w)
        y = int((top - lat) / (top - bottom) * canvas_h)
        return (x, y)

    # Draw original contour in blue
    contour_canvas = [geo_to_canvas(lon, lat) for lon, lat in contour_geo]
    for i in range(len(contour_canvas)):
        p1 = contour_canvas[i]
        p2 = contour_canvas[(i + 1) % len(contour_canvas)]
        cv2.line(vis, p1, p2, (255, 0, 0), 1)  # Blue

    # Draw matched segments in green
    for start_idx, end_idx, boundary in matches:
        for i in range(start_idx, end_idx + 1):
            idx = i % len(contour_canvas)
            next_idx = (i + 1) % len(contour_canvas)
            p1 = contour_canvas[idx]
            p2 = contour_canvas[next_idx]
            cv2.line(vis, p1, p2, (0, 255, 0), 2)  # Green (thicker)

    cv2.imwrite(output_path, vis)


def generate_boundary_report(
    library: SharedBoundaryLibrary,
    output_path: str
) -> None:
    """
    Generate a text report of all extracted boundaries.

    Args:
        library: Boundary library
        output_path: Path to save report
    """
    tag_names = {1: "road", 2: "kerb", 3: "grass", 4: "sand", 5: "road2"}

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Shared Boundary Extraction Report\n\n")
        f.write(f"Total boundaries: {len(library.boundaries)}\n\n")

        # Group by tag pair
        pair_groups = {}
        for boundary in library.boundaries:
            pair = boundary.tag_pair
            if pair not in pair_groups:
                pair_groups[pair] = []
            pair_groups[pair].append(boundary)

        for pair, boundaries in sorted(pair_groups.items()):
            tag1_name = tag_names.get(pair[0], f"tag{pair[0]}")
            tag2_name = tag_names.get(pair[1], f"tag{pair[1]}")
            f.write(f"## {tag1_name} <-> {tag2_name} (tags {pair[0]}, {pair[1]})\n\n")
            f.write(f"Boundaries: {len(boundaries)}\n\n")

            for idx, boundary in enumerate(boundaries):
                f.write(f"### Boundary {idx + 1}\n")
                f.write(f"- Pixel chain length: {len(boundary.pixel_chain)}\n")
                f.write(f"- Simplified geo points: {len(boundary.geo_coords)}\n")
                f.write(f"- Length (pixels): {boundary.length_pixels:.1f}\n\n")

        f.write("\n## Per-Tag Summary\n\n")
        for tag_id in range(1, 6):
            tag_name = tag_names.get(tag_id, f"tag{tag_id}")
            boundaries = library.get_boundaries_for_tag(tag_id)
            f.write(f"- {tag_name} (tag {tag_id}): {len(boundaries)} boundaries\n")


def _geo_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate approximate distance between two geographic points in meters."""
    lat1, lon1 = p1[1], p1[0]
    lat2, lon2 = p2[1], p2[0]

    # Simple Euclidean approximation (good enough for small distances)
    dlat = (lat2 - lat1) * 111320  # 1 degree latitude ≈ 111.32 km
    dlon = (lon2 - lon1) * 111320 * np.cos(np.radians((lat1 + lat2) / 2))

    return np.sqrt(dlat**2 + dlon**2)


def _point_to_segment_distance(
    point: Tuple[float, float],
    seg_start: Tuple[float, float],
    seg_end: Tuple[float, float]
) -> float:
    """Calculate minimum distance from point to line segment in meters."""
    px, py = point[0], point[1]
    ax, ay = seg_start[0], seg_start[1]
    bx, by = seg_end[0], seg_end[1]

    # Vector from A to B
    abx, aby = bx - ax, by - ay
    # Vector from A to P
    apx, apy = px - ax, py - ay

    # Project P onto AB
    ab_len_sq = abx * abx + aby * aby
    if ab_len_sq < 1e-12:
        # Degenerate segment, return distance to point A
        return _geo_distance(point, seg_start)

    t = max(0, min(1, (apx * abx + apy * aby) / ab_len_sq))

    # Closest point on segment
    closest = (ax + t * abx, ay + t * aby)
    return _geo_distance(point, closest)


def _point_to_polyline_distance(
    point: Tuple[float, float],
    polyline: List[Tuple[float, float]]
) -> float:
    """Calculate minimum distance from point to polyline in meters."""
    if len(polyline) < 2:
        return _geo_distance(point, polyline[0]) if polyline else float('inf')

    min_dist = float('inf')
    for i in range(len(polyline) - 1):
        dist = _point_to_segment_distance(point, polyline[i], polyline[i+1])
        min_dist = min(min_dist, dist)
    return min_dist


def _get_bbox(coords: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    """Get bounding box (min_lon, min_lat, max_lon, max_lat) for coordinates."""
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return (min(lons), min(lats), max(lons), max(lats))


def _bbox_overlaps(bbox1: Tuple[float, float, float, float],
                   bbox2: Tuple[float, float, float, float],
                   margin_m: float = 10.0) -> bool:
    """Check if two bounding boxes overlap with margin in meters."""
    # Convert margin to approximate degrees (rough approximation)
    margin_deg = margin_m / 111320.0

    min_lon1, min_lat1, max_lon1, max_lat1 = bbox1
    min_lon2, min_lat2, max_lon2, max_lat2 = bbox2

    # Expand bbox1 by margin
    min_lon1 -= margin_deg
    min_lat1 -= margin_deg
    max_lon1 += margin_deg
    max_lat1 += margin_deg

    # Check overlap
    return not (max_lon1 < min_lon2 or max_lon2 < min_lon1 or
                max_lat1 < min_lat2 or max_lat2 < min_lat1)


def match_contour_segments(
    contour_geo: List[Tuple[float, float]],
    shared_boundaries: List[SharedBoundary],
    tolerance_m: float = 0.05,
    logger=None
) -> List[Tuple[int, int, SharedBoundary]]:
    """
    Match contour segments to shared boundaries.

    Args:
        contour_geo: Contour in geographic coordinates
        shared_boundaries: Candidate boundaries for this tag
        tolerance_m: Matching tolerance in meters
        logger: Optional logger

    Returns:
        List of (start_idx, end_idx, boundary) for matched segments
    """
    import logging
    if logger is None:
        logger = logging.getLogger(__name__)

    matches = []
    n_contour = len(contour_geo)

    # Spatial filtering: compute contour bbox and filter boundaries
    contour_bbox = _get_bbox(contour_geo)
    candidate_boundaries = []
    for boundary in shared_boundaries:
        boundary_bbox = _get_bbox(boundary.geo_coords)
        if _bbox_overlaps(contour_bbox, boundary_bbox, margin_m=tolerance_m * 5):
            candidate_boundaries.append(boundary)

    if len(candidate_boundaries) < len(shared_boundaries):
        logger.debug("    Spatial filter: %d/%d boundaries (saved %d checks)",
                    len(candidate_boundaries), len(shared_boundaries),
                    len(shared_boundaries) - len(candidate_boundaries))

    for boundary_idx, boundary in enumerate(candidate_boundaries):
        if boundary_idx > 0 and boundary_idx % 50 == 0:
            logger.info("      Processing boundary %d/%d...", boundary_idx, len(candidate_boundaries))

        boundary_coords = boundary.geo_coords
        n_boundary = len(boundary_coords)

        if n_boundary < 2:
            continue

        best_match_score = 0
        best_start_idx = -1
        window_size = min(n_boundary, n_contour)

        # Sliding window search using point-to-polyline distance
        for i in range(n_contour):
            # Early termination: if remaining positions can't beat best score
            if n_contour - i + best_match_score < n_boundary * 0.4:
                break

            match_count = 0
            for j in range(window_size):
                contour_idx = (i + j) % n_contour
                dist = _point_to_polyline_distance(contour_geo[contour_idx], boundary_coords)
                if dist < tolerance_m:
                    match_count += 1

                # Early exit if this window can't beat best score
                remaining = window_size - j - 1
                if match_count + remaining <= best_match_score:
                    break

            if match_count > best_match_score:
                best_match_score = match_count
                best_start_idx = i

        # Accept if >30% of boundary points match (balanced threshold)
        match_ratio = best_match_score / n_boundary if n_boundary > 0 else 0
        if best_match_score >= n_boundary * 0.30:
            end_idx = (best_start_idx + n_boundary - 1) % n_contour
            matches.append((best_start_idx, end_idx, boundary))
            logger.debug("      Boundary %d: matched at contour[%d:%d], ratio=%.1f%%",
                        boundary_idx, best_start_idx, end_idx, match_ratio * 100)
        else:
            logger.debug("      Boundary %d: no match (best ratio=%.1f%%)",
                        boundary_idx, match_ratio * 100)

    return matches


def rebuild_contour_with_shared_boundaries(
    contour_geo: List[Tuple[float, float]],
    matches: List[Tuple[int, int, SharedBoundary]],
    tag: int
) -> List[Tuple[float, float]]:
    """
    Rebuild contour by replacing matched segments with shared boundaries.
    Correctly handles closed ring contours where indices wrap around.
    """
    if not matches:
        return contour_geo

    n = len(contour_geo)
    if n == 0:
        return contour_geo

    # Sort matches by start index
    sorted_matches = sorted(matches, key=lambda m: m[0])

    rebuilt = []

    for idx, (start, end, boundary) in enumerate(sorted_matches):
        boundary_coords = boundary.get_for_tag(tag)

        if idx == 0:
            # First match: add gap from last match's end to this start
            last_end = sorted_matches[-1][1]

            if last_end < start - 1:
                # Normal gap: last_end+1 to start-1
                rebuilt.extend(contour_geo[last_end + 1:start])
            elif last_end >= start:
                # Overlapping matches, skip gap
                pass
            else:
                # Wrapped: last_end+1 to n-1, then 0 to start-1
                if last_end + 1 < n:
                    rebuilt.extend(contour_geo[last_end + 1:n])
                if start > 0:
                    rebuilt.extend(contour_geo[0:start])
        else:
            # Subsequent matches: add gap from previous end to this start
            prev_end = sorted_matches[idx - 1][1]
            if prev_end < start - 1:
                rebuilt.extend(contour_geo[prev_end + 1:start])

        # Add boundary
        rebuilt.extend(boundary_coords)

    return rebuilt


def simplify_with_shared_boundaries(
    raw_pixel_contours: Dict[int, np.ndarray],
    boundary_lib,
    tag_id: int,
    bounds: Dict[str, float],
    canvas_w: int,
    canvas_h: int,
    epsilon: float,
    logger
) -> Dict[int, List[List[float]]]:
    """
    Simplify contours with shared boundary awareness (fundamental solution).

    Shared edges use pre-simplified boundary coordinates.
    Non-shared edges are simplified independently.

    Returns: {contour_idx: [[lon, lat], ...]}
    """
    import logging
    if logger is None:
        logger = logging.getLogger(__name__)

    boundaries_for_tag = boundary_lib.get_boundaries_for_tag(tag_id)
    logger.info("  Tag (id=%d): %d available shared boundaries", tag_id, len(boundaries_for_tag))

    if not boundaries_for_tag:
        return {}

    # Build pixel lookup: pixel -> list of boundary indices
    pixel_to_boundaries = {}
    for b_idx, boundary in enumerate(boundaries_for_tag):
        for px, py in boundary.simplified_pixels:
            key = (px, py)
            if key not in pixel_to_boundaries:
                pixel_to_boundaries[key] = []
            pixel_to_boundaries[key].append(b_idx)

    geo_contours = {}
    total_matched = 0

    from pipeline_config import PipelineConfig
    tolerance = getattr(PipelineConfig(), 's8_boundary_match_tolerance_m', 0.15)

    for contour_idx, raw_contour in raw_pixel_contours.items():
        # Convert to geo for matching
        raw_geo = []
        for pt in raw_contour:
            if len(pt.shape) == 1:
                px, py = float(pt[0]), float(pt[1])
            else:
                px, py = float(pt[0][0]), float(pt[0][1])
            lon, lat = _canvas_to_geo(px, py, bounds, canvas_w, canvas_h)
            raw_geo.append((lon, lat))

        # Match segments
        matches = match_contour_segments(raw_geo, boundaries_for_tag, tolerance_m=tolerance, logger=logger)

        if matches:
            total_matched += 1
            logger.info("    Contour %d: matched %d boundary segments", contour_idx, len(matches))
            rebuilt = rebuild_contour_with_shared_boundaries(raw_geo, matches, tag_id)
            geo_contours[contour_idx] = [[pt[0], pt[1]] for pt in rebuilt]
        else:
            # No match: standard simplification
            contour_array = raw_contour.reshape(-1, 1, 2) if len(raw_contour.shape) == 2 else raw_contour
            approx = cv2.approxPolyDP(contour_array, epsilon, closed=True)
            geo_pts = []
            for pt in approx:
                px, py = float(pt[0][0]), float(pt[0][1])
                lon, lat = _canvas_to_geo(px, py, bounds, canvas_w, canvas_h)
                geo_pts.append([lon, lat])
            geo_contours[contour_idx] = geo_pts

    logger.info("  Tag (id=%d): matched %d/%d contours", tag_id, total_matched, len(raw_pixel_contours))
    return geo_contours
