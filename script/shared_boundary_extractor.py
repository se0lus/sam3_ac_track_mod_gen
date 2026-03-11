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
    lon = bounds["min_lon"] + (x / canvas_w) * (bounds["max_lon"] - bounds["min_lon"])
    lat = bounds["max_lat"] - (y / canvas_h) * (bounds["max_lat"] - bounds["min_lat"])
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
    min_chain_length: int = 3
) -> SharedBoundaryLibrary:
    """
    Extract all shared boundaries from a composite label map.

    Args:
        composite: Label map with tag IDs
        bounds: Geographic bounds
        canvas_w, canvas_h: Canvas dimensions
        simplify_epsilon: Boundary simplification threshold (pixels)
        min_chain_length: Minimum chain length to keep

    Returns:
        SharedBoundaryLibrary containing all extracted boundaries
    """
    library = SharedBoundaryLibrary()

    # Step 1: Detect boundary pixels
    boundary_pixel_map = detect_boundary_pixels(composite)

    # Step 2: Trace and vectorize each tag pair's boundaries
    for tag_pair, pixels in boundary_pixel_map.items():
        chains = trace_boundary_chains(pixels, min_chain_length)

        for chain in chains:
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

    return library


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
    tolerance_m: float = 0.05
) -> List[Tuple[int, int, SharedBoundary]]:
    """
    Match contour segments to shared boundaries.

    Args:
        contour_geo: Contour in geographic coordinates
        shared_boundaries: Candidate boundaries for this tag
        tolerance_m: Matching tolerance in meters

    Returns:
        List of (start_idx, end_idx, boundary) for matched segments
    """
    matches = []
    n_contour = len(contour_geo)

    for boundary in shared_boundaries:
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
        if best_match_score >= n_boundary * 0.6:
            end_idx = (best_start_idx + n_boundary - 1) % n_contour
            matches.append((best_start_idx, end_idx, boundary))

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
