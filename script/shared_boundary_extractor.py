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
    geo_coords: List[Tuple[float, float]]  # Geographic coordinates
    length_pixels: float  # Boundary length in pixels

    def get_for_tag(self, tag: int) -> List[Tuple[float, float]]:
        """Get boundary coordinates for a specific tag (may reverse)."""
        if tag == self.tag_pair[0]:
            return self.geo_coords
        else:
            return list(reversed(self.geo_coords))


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
) -> List[Tuple[float, float]]:
    """
    Simplify boundary chain using Douglas-Peucker and convert to geo coordinates.

    Args:
        chain: Pixel coordinate chain
        bounds: Geographic bounds
        canvas_w, canvas_h: Canvas dimensions
        epsilon: Simplification threshold in pixels

    Returns:
        List of geographic coordinates
    """
    chain_array = np.array(chain, dtype=np.float32).reshape(-1, 1, 2)
    simplified = cv2.approxPolyDP(chain_array, epsilon, closed=False)

    geo_coords = []
    for pt in simplified:
        x, y = float(pt[0][0]), float(pt[0][1])
        lon, lat = _canvas_to_geo(x, y, bounds, canvas_w, canvas_h)
        geo_coords.append((lon, lat))

    return geo_coords


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
            geo_coords = simplify_and_vectorize_boundary(
                chain, bounds, canvas_w, canvas_h, simplify_epsilon
            )

            if len(geo_coords) >= 2:
                length = len(chain)
                boundary = SharedBoundary(
                    tag_pair=tag_pair,
                    pixel_chain=chain,
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

    for boundary_idx, boundary in enumerate(shared_boundaries):
        boundary_coords = boundary.geo_coords
        n_boundary = len(boundary_coords)

        if n_boundary < 2:
            continue

        best_match_score = 0
        best_start_idx = -1

        # Sliding window search
        for i in range(n_contour):
            match_count = 0
            for j in range(min(n_boundary, n_contour)):
                contour_idx = (i + j) % n_contour
                dist = _geo_distance(contour_geo[contour_idx], boundary_coords[j])
                if dist < tolerance_m:
                    match_count += 1

            if match_count > best_match_score:
                best_match_score = match_count
                best_start_idx = i

        # Accept if >60% of boundary points match
        match_ratio = best_match_score / n_boundary if n_boundary > 0 else 0
        if best_match_score >= n_boundary * 0.6:
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

    Args:
        contour_geo: Original contour
        matches: Matched segments [(start, end, boundary), ...]
        tag: Current tag ID

    Returns:
        Rebuilt contour with shared boundaries
    """
    if not matches:
        return contour_geo

    # Sort by start index
    matches = sorted(matches, key=lambda m: m[0])

    rebuilt = []
    last_end = -1

    for start, end, boundary in matches:
        # Add original points between last segment and current
        if last_end + 1 < start:
            rebuilt.extend(contour_geo[last_end + 1:start])

        # Add shared boundary (possibly reversed)
        boundary_coords = boundary.get_for_tag(tag)
        rebuilt.extend(boundary_coords)

        last_end = end

    # Add remaining points
    if last_end + 1 < len(contour_geo):
        rebuilt.extend(contour_geo[last_end + 1:])

    return rebuilt
