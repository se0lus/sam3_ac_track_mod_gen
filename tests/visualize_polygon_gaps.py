"""Gap diagnostic tool for Blender polygon JSONs.

Loads all *_merged_blender.json files from a gap_filled directory,
rasterizes them back to a composite label map, and finds gap pixels
(pixels that are 0 but have 2+ different non-zero neighbors).

Usage:
    python tests/visualize_polygon_gaps.py <gap_filled_dir> [--output gaps.png] [--size 2000]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

# Add script dir to path
_script_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "script")
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)


TAG_COLORS_BGR = {
    "sand":  (100, 200, 200),
    "grass": (0, 200, 0),
    "road2": (180, 180, 180),
    "road":  (100, 100, 100),
    "kerb":  (0, 0, 255),
}

TAG_NAME_TO_ID = {"sand": 1, "grass": 2, "road2": 3, "road": 4, "kerb": 5}


def load_all_tag_polygons(gap_filled_dir: str) -> dict:
    """Load all *_merged_blender.json files.

    Returns dict: tag_name -> list of mesh_groups (each with geo_xy + faces).
    """
    result = {}
    for entry in os.listdir(gap_filled_dir):
        entry_path = os.path.join(gap_filled_dir, entry)
        if not os.path.isdir(entry_path):
            continue
        json_path = os.path.join(entry_path, f"{entry}_merged_blender.json")
        if not os.path.isfile(json_path):
            continue
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        result[entry] = data.get("mesh_groups", [])
    return result


def compute_bounds(polygons: dict) -> dict:
    """Compute geographic bounds from all polygon vertices."""
    all_lons = []
    all_lats = []
    for tag, groups in polygons.items():
        for mg in groups:
            for pt in mg.get("geo_xy", []):
                if len(pt) >= 2:
                    all_lons.append(pt[0])
                    all_lats.append(pt[1])

    if not all_lons:
        return {"left": 0, "right": 1, "top": 1, "bottom": 0}

    margin = 0.0001  # small margin in degrees
    return {
        "left": min(all_lons) - margin,
        "right": max(all_lons) + margin,
        "top": max(all_lats) + margin,
        "bottom": min(all_lats) - margin,
    }


def rasterize_to_composite(
    polygons: dict, bounds: dict, width: int, height: int,
) -> np.ndarray:
    """Rasterize polygon mesh groups back to a label map."""
    composite = np.zeros((height, width), dtype=np.uint8)

    def geo_to_px(lon, lat):
        x = (lon - bounds["left"]) / (bounds["right"] - bounds["left"]) * width
        y = (bounds["top"] - lat) / (bounds["top"] - bounds["bottom"]) * height
        return int(round(x)), int(round(y))

    for tag_name, groups in polygons.items():
        tag_id = TAG_NAME_TO_ID.get(tag_name, 0)
        if tag_id == 0:
            continue
        for mg in groups:
            geo_xy = mg.get("geo_xy", [])
            faces = mg.get("faces", [])
            if not geo_xy or not faces:
                continue

            canvas_pts = []
            for pt in geo_xy:
                if len(pt) >= 2:
                    canvas_pts.append(geo_to_px(pt[0], pt[1]))
                else:
                    canvas_pts.append((0, 0))

            for face in faces:
                if len(face) < 3:
                    continue
                tri = np.array([
                    [canvas_pts[face[0]][0], canvas_pts[face[0]][1]],
                    [canvas_pts[face[1]][0], canvas_pts[face[1]][1]],
                    [canvas_pts[face[2]][0], canvas_pts[face[2]][1]],
                ], dtype=np.int32)
                cv2.fillPoly(composite, [tri], tag_id)

    return composite


def find_gap_pixels(composite: np.ndarray) -> np.ndarray:
    """Find pixels that are 0 but have 2+ different non-zero neighbors.

    Returns binary mask of gap pixels.
    """
    h, w = composite.shape
    gaps = np.zeros((h, w), dtype=np.uint8)

    zero_mask = (composite == 0)
    zero_ys, zero_xs = np.where(zero_mask)

    for i in range(len(zero_ys)):
        y, x = zero_ys[i], zero_xs[i]
        neighbors = set()
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                val = int(composite[ny, nx])
                if val > 0:
                    neighbors.add(val)
        if len(neighbors) >= 2:
            gaps[y, x] = 255

    return gaps


def measure_edge_gaps(
    polygons: dict, bounds: dict, width: int, height: int,
) -> dict:
    """Measure minimum edge-to-edge distances between adjacent tags.

    Returns dict of (tag1, tag2) -> min_distance_pixels.
    """
    # Rasterize each tag separately and find edges
    tag_edges = {}
    for tag_name, groups in polygons.items():
        tag_id = TAG_NAME_TO_ID.get(tag_name, 0)
        if tag_id == 0:
            continue
        mask = np.zeros((height, width), dtype=np.uint8)

        def geo_to_px(lon, lat):
            x = (lon - bounds["left"]) / (bounds["right"] - bounds["left"]) * width
            y = (bounds["top"] - lat) / (bounds["top"] - bounds["bottom"]) * height
            return int(round(x)), int(round(y))

        for mg in groups:
            geo_xy = mg.get("geo_xy", [])
            faces = mg.get("faces", [])
            if not geo_xy or not faces:
                continue
            canvas_pts = [geo_to_px(pt[0], pt[1]) if len(pt) >= 2 else (0, 0) for pt in geo_xy]
            for face in faces:
                if len(face) < 3:
                    continue
                tri = np.array([
                    [canvas_pts[face[0]][0], canvas_pts[face[0]][1]],
                    [canvas_pts[face[1]][0], canvas_pts[face[1]][1]],
                    [canvas_pts[face[2]][0], canvas_pts[face[2]][1]],
                ], dtype=np.int32)
                cv2.fillPoly(mask, [tri], 255)

        # Edge = eroded XOR original
        kernel = np.ones((3, 3), dtype=np.uint8)
        eroded = cv2.erode(mask, kernel)
        edge = mask & ~eroded
        tag_edges[tag_name] = edge

    # Measure distances between edge pixels of different tags
    distances = {}
    tag_names = list(tag_edges.keys())
    for i in range(len(tag_names)):
        for j in range(i + 1, len(tag_names)):
            t1, t2 = tag_names[i], tag_names[j]
            e1 = tag_edges[t1]
            e2 = tag_edges[t2]

            if np.count_nonzero(e1) == 0 or np.count_nonzero(e2) == 0:
                continue

            # Use distance transform for efficiency
            dist_from_e1 = cv2.distanceTransform(255 - e1, cv2.DIST_L2, 5)
            min_dist = float(np.min(dist_from_e1[e2 > 0])) if np.any(e2 > 0) else float('inf')
            distances[(t1, t2)] = min_dist

    return distances


def visualize(
    composite: np.ndarray,
    gaps: np.ndarray,
    output_path: str,
) -> None:
    """Save colorized visualization with gap pixels highlighted."""
    h, w = composite.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)

    id_to_name = {v: k for k, v in TAG_NAME_TO_ID.items()}
    for tag_id in range(1, 6):
        tag_name = id_to_name.get(tag_id)
        if tag_name and tag_name in TAG_COLORS_BGR:
            vis[composite == tag_id] = TAG_COLORS_BGR[tag_name]

    # Highlight gaps in bright magenta
    vis[gaps > 0] = (255, 0, 255)

    # Add text
    n_gaps = int(np.count_nonzero(gaps))
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"Gap pixels: {n_gaps}"
    color = (0, 255, 0) if n_gaps == 0 else (0, 0, 255)
    cv2.putText(vis, text, (10, 30), font, 0.8, color, 2)

    cv2.imwrite(output_path, vis)
    print(f"Visualization saved: {output_path}")
    print(f"Gap pixels: {n_gaps}")


def main():
    parser = argparse.ArgumentParser(description="Visualize polygon gaps")
    parser.add_argument("gap_filled_dir", help="Path to gap_filled directory")
    parser.add_argument("--output", "-o", default="gaps.png", help="Output image path")
    parser.add_argument("--size", type=int, default=2000, help="Rasterization size (max dimension)")
    args = parser.parse_args()

    if not os.path.isdir(args.gap_filled_dir):
        print(f"Error: directory not found: {args.gap_filled_dir}")
        sys.exit(1)

    print(f"Loading polygons from {args.gap_filled_dir}...")
    polygons = load_all_tag_polygons(args.gap_filled_dir)
    if not polygons:
        print("No polygon data found.")
        sys.exit(1)

    print(f"Loaded tags: {list(polygons.keys())}")
    for tag, groups in polygons.items():
        n_verts = sum(len(mg.get("geo_xy", [])) for mg in groups)
        n_faces = sum(len(mg.get("faces", [])) for mg in groups)
        print(f"  {tag}: {len(groups)} groups, {n_verts} verts, {n_faces} faces")

    bounds = compute_bounds(polygons)
    aspect = (bounds["right"] - bounds["left"]) / max(bounds["top"] - bounds["bottom"], 1e-10)
    if aspect > 1:
        width = args.size
        height = max(1, int(args.size / aspect))
    else:
        height = args.size
        width = max(1, int(args.size * aspect))

    print(f"Rasterizing at {width}x{height}...")
    composite = rasterize_to_composite(polygons, bounds, width, height)

    print("Finding gap pixels...")
    gaps = find_gap_pixels(composite)

    print("Measuring edge gaps...")
    distances = measure_edge_gaps(polygons, bounds, width, height)
    for (t1, t2), dist in sorted(distances.items()):
        status = "OK (touching)" if dist <= 1.5 else f"GAP ({dist:.1f}px)"
        print(f"  {t1} <-> {t2}: min edge distance = {dist:.2f}px - {status}")

    visualize(composite, gaps, args.output)


if __name__ == "__main__":
    main()
