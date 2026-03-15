"""Debug pixel-corner extraction for kerb — visualize every phase.

Outputs PNG images to output_shajing/08_blender_polygons/pcorner_debug/
"""
from __future__ import annotations

import json
import logging
import os
import sys
from collections import defaultdict

import cv2
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPT_DIR = os.path.join(_PROJECT_ROOT, "script")
sys.path.insert(0, _SCRIPT_DIR)

from pipeline_config import PipelineConfig

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("debug_kerb")

KERB_ID = 5
# Colors for different labels (BGR)
LABEL_COLORS = {
    0: (30, 30, 30),      # background
    1: (100, 200, 200),   # sand
    2: (0, 180, 0),       # grass
    3: (180, 180, 180),   # road2
    4: (100, 100, 100),   # road
    5: (0, 0, 255),       # kerb
}


def _load_composite(config):
    """Reproduce the composite from s08 pipeline."""
    from mask_merger import _read_geotiff_bounds, _compute_canvas_size
    from mask_gap_filler import (
        fill_mask_gaps, build_driveable_zone,
        SURFACE_TAGS, TAG_NAME_TO_ID, INDEPENDENT_TAGS,
        _compute_meters_per_pixel,
    )
    from stages.s08_blender_polygons import _load_walls, _rasterize_blender_json

    geotiff_meta = _read_geotiff_bounds(config.geotiff_path)
    bounds = geotiff_meta["bounds"]
    bounds_wgs84 = geotiff_meta["bounds_wgs84"]
    canvas_w, canvas_h, _ = _compute_canvas_size(geotiff_meta, config.mask_on_clips_dir)

    input_dir = config.merge_segments_result
    if not os.path.isdir(input_dir):
        input_dir = config.merge_segments_dir

    walls, wall_resolution = _load_walls(config)
    driveable = build_driveable_zone(canvas_h, canvas_w, walls, wall_resolution=wall_resolution)

    surface_masks = {}
    independent_masks = {}
    for tag_name in SURFACE_TAGS + INDEPENDENT_TAGS:
        json_path = os.path.join(input_dir, tag_name, f"{tag_name}_merged_blender.json")
        if not os.path.isfile(json_path):
            continue
        mask = _rasterize_blender_json(json_path, bounds_wgs84, canvas_w, canvas_h)
        if tag_name in SURFACE_TAGS:
            surface_masks[tag_name] = mask
        else:
            independent_masks[tag_name] = mask

    composite = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    for tag_name in SURFACE_TAGS:
        if tag_name not in surface_masks:
            continue
        composite[surface_masks[tag_name] > 127] = TAG_NAME_TO_ID[tag_name]
    composite[~driveable] = 0

    fill_zone = driveable.copy()
    for mask in independent_masks.values():
        fill_zone[mask > 127] = False

    filled = fill_mask_gaps(composite, fill_zone, bounds,
                            gap_threshold_m=config.s8_gap_fill_threshold_m,
                            default_tag=config.s8_gap_fill_default_tag)
    composite = filled

    if config.s8_filter_narrow_strips and config.s8_min_width_m > 0:
        mpp = _compute_meters_per_pixel(bounds, canvas_w, canvas_h)
        radius = max(1, int(round(config.s8_min_width_m / 2 / mpp)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
        for tag_name in reversed(SURFACE_TAGS):
            tag_id = TAG_NAME_TO_ID[tag_name]
            mask = (composite == tag_id).astype(np.uint8) * 255
            if np.count_nonzero(mask) == 0:
                continue
            opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            removed = (mask > 0) & (opened == 0)
            if np.count_nonzero(removed) > 0:
                composite[removed] = 0

    return composite, bounds_wgs84, canvas_w, canvas_h


def _crop_to_label(composite, label, pad=20):
    """Crop composite to bounding box of label + padding. Returns crop, (x0, y0)."""
    ys, xs = np.where(composite == label)
    if len(xs) == 0:
        return composite, (0, 0)
    x0 = max(0, xs.min() - pad)
    y0 = max(0, ys.min() - pad)
    x1 = min(composite.shape[1], xs.max() + pad + 1)
    y1 = min(composite.shape[0], ys.max() + pad + 1)
    return composite[y0:y1, x0:x1].copy(), (x0, y0)


def _colorize(composite):
    """Colorize composite label map."""
    h, w = composite.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for label, color in LABEL_COLORS.items():
        img[composite == label] = color
    return img


def main():
    config_json = os.path.join(_PROJECT_ROOT, "webtools_config.json")
    config = PipelineConfig()
    with open(config_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    for key, value in data.items():
        if not key.startswith("_") and key != "manual_stages" and hasattr(config, key):
            setattr(config, key, value)
    config.resolve()

    debug_dir = os.path.join(config.output_dir, "08_blender_polygons", "pcorner_debug")
    os.makedirs(debug_dir, exist_ok=True)
    logger.info("Debug output: %s", debug_dir)

    # --- Load composite ---
    logger.info("Loading composite...")
    composite, bounds_wgs84, canvas_w, canvas_h = _load_composite(config)

    # Crop to kerb region
    crop, (ox, oy) = _crop_to_label(composite, KERB_ID, pad=30)
    cH, cW = crop.shape
    logger.info("Cropped to kerb: %dx%d, offset (%d, %d)", cW, cH, ox, oy)

    # --- 0. Save cropped composite ---
    img0 = _colorize(crop)
    cv2.imwrite(os.path.join(debug_dir, "00_composite_crop.png"), img0)
    logger.info("Saved 00_composite_crop.png")

    # --- Phase 1: Boundary edges (on cropped composite) ---
    from pixel_corner_contour_extractor import (
        extract_boundary_edges, build_chains, simplify_chains,
        assemble_rings, _edge_label_pair, _build_adjacency,
        _triangulate_group_earcut, _canvas_to_geo,
    )

    edges = extract_boundary_edges(crop)
    logger.info("Phase 1: %d edges", len(edges))

    # Draw all edges on a larger canvas (corner-space is +1 in each dim)
    scale = 4  # draw at 4x for visibility
    eW, eH = (cW + 1) * scale, (cH + 1) * scale

    img1 = np.zeros((eH, eW, 3), dtype=np.uint8)
    # Draw pixels as filled squares
    for y in range(cH):
        for x in range(cW):
            label = crop[y, x]
            if label > 0:
                color = LABEL_COLORS.get(int(label), (128, 128, 128))
                cv2.rectangle(img1,
                              (x * scale + 1, y * scale + 1),
                              ((x + 1) * scale - 1, (y + 1) * scale - 1),
                              color, -1)

    # Draw edges
    for i in range(len(edges)):
        e = edges[i]
        x0, y0, x1, y1 = int(e[0]), int(e[1]), int(e[2]), int(e[3])
        lbl_l, lbl_r = int(e[4]), int(e[5])
        # Color: kerb boundary = bright red, other = dim
        if KERB_ID in (lbl_l, lbl_r):
            color = (0, 0, 255)
            thickness = 2
        else:
            color = (80, 80, 80)
            thickness = 1
        cv2.line(img1, (x0 * scale, y0 * scale), (x1 * scale, y1 * scale), color, thickness)

    cv2.imwrite(os.path.join(debug_dir, "01_edges_all.png"), img1)
    logger.info("Saved 01_edges_all.png")

    # --- Phase 2: Chains ---
    chains, junctions = build_chains(edges)
    logger.info("Phase 2: %d chains, %d junctions", len(chains), len(junctions))

    # Identify kerb chains
    kerb_chain_indices = []
    for ci, chain in enumerate(chains):
        if KERB_ID in chain['labels']:
            kerb_chain_indices.append(ci)
    logger.info("  Kerb chains: %d", len(kerb_chain_indices))

    # Draw chains with unique colors
    img2 = np.zeros((eH, eW, 3), dtype=np.uint8)
    # Background: dim composite
    for y in range(cH):
        for x in range(cW):
            label = crop[y, x]
            if label > 0:
                color = LABEL_COLORS.get(int(label), (128, 128, 128))
                cv2.rectangle(img2,
                              (x * scale + 1, y * scale + 1),
                              ((x + 1) * scale - 1, (y + 1) * scale - 1),
                              tuple(c // 3 for c in color), -1)

    np.random.seed(42)
    chain_colors = [(int(np.random.randint(60, 255)), int(np.random.randint(60, 255)),
                      int(np.random.randint(60, 255))) for _ in range(len(chains))]

    for ci in kerb_chain_indices:
        chain = chains[ci]
        corners = chain['corners']
        color = chain_colors[ci]
        labels_str = str(set(chain['labels']))
        for j in range(len(corners) - 1):
            p0 = (corners[j][0] * scale, corners[j][1] * scale)
            p1 = (corners[j+1][0] * scale, corners[j+1][1] * scale)
            cv2.line(img2, p0, p1, color, 2)
        # Draw chain endpoints
        start = corners[0]
        end = corners[-1]
        cv2.circle(img2, (start[0] * scale, start[1] * scale), 4, (255, 255, 0), -1)
        cv2.circle(img2, (end[0] * scale, end[1] * scale), 4, (0, 255, 255), -1)

    # Draw junctions
    for junc in junctions:
        cv2.circle(img2, (junc[0] * scale, junc[1] * scale), 6, (255, 255, 255), 2)

    cv2.imwrite(os.path.join(debug_dir, "02_chains_kerb.png"), img2)
    logger.info("Saved 02_chains_kerb.png")

    # --- Log chain details ---
    for ci in kerb_chain_indices:
        chain = chains[ci]
        corners = chain['corners']
        start = tuple(corners[0])
        end = tuple(corners[-1])
        is_closed = start == end and len(corners) > 2
        logger.info("  Chain %d: labels=%s, %d pts, start=%s, end=%s, closed=%s",
                     ci, set(chain['labels']), len(corners), start, end, is_closed)

    # --- Phase 3: Simplify ---
    # We need to convert crop coordinates → geo using the crop offset
    # Actually, let's just work in pixel coordinates for visualization
    # and run simplification without geo conversion for now
    chains = simplify_chains(chains, junctions, epsilon=config.s8_shared_boundary_epsilon,
                             bounds=bounds_wgs84, canvas_w=canvas_w, canvas_h=canvas_h)

    # Draw simplified chains
    img3 = img2.copy()
    for ci in kerb_chain_indices:
        chain = chains[ci]
        simp = chain.get('simplified_corners', chain['corners'])
        color = chain_colors[ci]
        for j in range(len(simp) - 1):
            p0 = (simp[j][0] * scale, simp[j][1] * scale)
            p1 = (simp[j+1][0] * scale, simp[j+1][1] * scale)
            cv2.line(img3, p0, p1, (255, 255, 255), 3)  # white = simplified
            cv2.line(img3, p0, p1, color, 1)             # chain color inside

    cv2.imwrite(os.path.join(debug_dir, "03_chains_simplified.png"), img3)
    logger.info("Saved 03_chains_simplified.png")

    # --- Phase 4: Assemble rings ---
    ring_groups = assemble_rings(KERB_ID, chains, junctions)
    logger.info("Phase 4: %d ring groups for kerb", len(ring_groups))

    # Draw each ring group in pixel space (convert back from geo)
    from mask_merger import _geo_to_canvas

    img4 = np.zeros((eH, eW, 3), dtype=np.uint8)
    # Background
    for y in range(cH):
        for x in range(cW):
            label = crop[y, x]
            if label > 0:
                color = LABEL_COLORS.get(int(label), (128, 128, 128))
                cv2.rectangle(img4,
                              (x * scale + 1, y * scale + 1),
                              ((x + 1) * scale - 1, (y + 1) * scale - 1),
                              tuple(c // 4 for c in color), -1)

    ring_colors = [
        (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 0),
        (255, 0, 255), (0, 128, 255), (255, 128, 0), (128, 0, 255),
        (0, 255, 128), (128, 255, 0), (255, 128, 128), (128, 128, 255),
    ]

    for gi, (outer, holes) in enumerate(ring_groups):
        color = ring_colors[gi % len(ring_colors)]
        # Convert geo → pixel for drawing
        pts = []
        for lon, lat in outer:
            cx, cy = _geo_to_canvas(lon, lat, bounds_wgs84, canvas_w, canvas_h)
            # Offset to crop coordinates
            px = int(round((cx - ox) * scale))
            py = int(round((cy - oy) * scale))
            pts.append((px, py))

        # Draw outer ring
        for j in range(len(pts) - 1):
            cv2.line(img4, pts[j], pts[j+1], color, 2)

        # Mark vertices
        for pt in pts:
            cv2.circle(img4, pt, 3, color, -1)

        # Label
        if pts:
            from pixel_corner_contour_extractor import _signed_area_geo
            area = _signed_area_geo(outer)
            label_text = f"R{gi} ({len(outer)}v, area={area:.2e})"
            cv2.putText(img4, label_text, (pts[0][0] + 5, pts[0][1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        # Draw holes
        for hi, hole in enumerate(holes):
            pts_h = []
            for lon, lat in hole:
                cx, cy = _geo_to_canvas(lon, lat, bounds_wgs84, canvas_w, canvas_h)
                px = int(round((cx - ox) * scale))
                py = int(round((cy - oy) * scale))
                pts_h.append((px, py))
            for j in range(len(pts_h) - 1):
                cv2.line(img4, pts_h[j], pts_h[j+1], (0, 0, 200), 1)

    cv2.imwrite(os.path.join(debug_dir, "04_rings_kerb.png"), img4)
    logger.info("Saved 04_rings_kerb.png")

    # --- Log ring details ---
    from pixel_corner_contour_extractor import _signed_area_geo
    for gi, (outer, holes) in enumerate(ring_groups):
        area = _signed_area_geo(outer)
        logger.info("  Ring %d: %d outer verts, %d holes, signed_area=%.6e",
                     gi, len(outer), len(holes), area)
        # Check closure
        if outer[0] != outer[-1]:
            logger.warning("    NOT CLOSED! start=%s end=%s", outer[0], outer[-1])
        # Check self-intersection (simple heuristic: count vertex duplicates)
        vset = set(outer)
        if len(vset) < len(outer) - 1:  # -1 for closing point
            logger.warning("    Duplicate vertices: %d unique / %d total",
                           len(vset), len(outer))

    # --- Phase 5: Triangulate and verify ---
    logger.info("Phase 5: Triangulating...")
    good_groups = 0
    bad_groups = 0
    for gi, (outer, holes) in enumerate(ring_groups):
        if len(outer) < 3:
            logger.warning("  Ring %d: too few verts (%d)", gi, len(outer))
            bad_groups += 1
            continue
        outer_geo = [[x, y] for x, y in outer]
        holes_geo = [[[x, y] for x, y in h] for h in holes]
        faces = _triangulate_group_earcut(outer_geo, holes_geo)
        if faces is None:
            logger.warning("  Ring %d: TRIANGULATION FAILED (%d verts, %d holes)",
                           gi, len(outer), len(holes))
            bad_groups += 1
        else:
            good_groups += 1
            logger.info("  Ring %d: OK, %d faces", gi, len(faces))

    logger.info("")
    logger.info("=== SUMMARY ===")
    logger.info("Total ring groups: %d", len(ring_groups))
    logger.info("  Good: %d", good_groups)
    logger.info("  Bad:  %d", bad_groups)
    logger.info("Debug images saved to: %s", debug_dir)

    # --- Also save per-ring individual images ---
    for gi, (outer, holes) in enumerate(ring_groups):
        img_r = np.zeros((eH, eW, 3), dtype=np.uint8)
        # Background composite
        for y in range(cH):
            for x in range(cW):
                label = crop[y, x]
                if label > 0:
                    c = LABEL_COLORS.get(int(label), (128, 128, 128))
                    cv2.rectangle(img_r,
                                  (x * scale + 1, y * scale + 1),
                                  ((x + 1) * scale - 1, (y + 1) * scale - 1),
                                  tuple(v // 4 for v in c), -1)

        pts = []
        for lon, lat in outer:
            cx, cy = _geo_to_canvas(lon, lat, bounds_wgs84, canvas_w, canvas_h)
            px = int(round((cx - ox) * scale))
            py = int(round((cy - oy) * scale))
            pts.append((px, py))

        # Draw with vertex indices
        for j in range(len(pts) - 1):
            cv2.line(img_r, pts[j], pts[j+1], (0, 255, 0), 2)
        for j, pt in enumerate(pts):
            cv2.circle(img_r, pt, 3, (0, 255, 255), -1)
            if len(pts) < 60:  # only label if not too many
                cv2.putText(img_r, str(j), (pt[0]+4, pt[1]-4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)

        area = _signed_area_geo(outer)
        cv2.putText(img_r, f"Ring {gi}: {len(outer)}v, area={area:.2e}",
                    (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.imwrite(os.path.join(debug_dir, f"05_ring_{gi:02d}.png"), img_r)

    logger.info("Saved per-ring images 05_ring_XX.png")


if __name__ == "__main__":
    main()
