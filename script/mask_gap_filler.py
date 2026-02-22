"""Mask gap filler: eliminate gaps within the driveable zone.

Takes a composite label map and a pre-built fill_zone mask, then fills
small gaps via morphological closing, priority dilation, and default fill.

Algorithm:
  1. Identify gap_pixels = (composite == 0) AND fill_zone
  2. Morphological closing per surface tag (high→low priority)
  3. Priority dilation per tag (high→low)
  4. Default-fill remaining voids with configurable tag (default: road2)
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger("sam3_pipeline.gap_filler")

# Surface tags ordered by priority (low → high).
# Must match the order used in Stage 5 compositing.
SURFACE_TAGS = ["sand", "grass", "road2", "road", "kerb"]
TAG_NAME_TO_ID = {name: idx + 1 for idx, name in enumerate(SURFACE_TAGS)}

# Independent tags (not gap-filled, used as fill-zone exclusions)
INDEPENDENT_TAGS = ["trees", "building", "water", "concrete"]

# Colors for debug images (BGR)
_TAG_COLORS_BGR = {
    "sand":  (100, 200, 200),  # yellow-ish
    "grass": (0, 200, 0),      # green
    "road2": (180, 180, 180),  # light grey
    "road":  (100, 100, 100),  # dark grey
    "kerb":  (0, 0, 255),      # red
    "trees": (0, 128, 0),      # dark green
    "building": (128, 0, 128), # purple
    "water": (255, 128, 0),    # cyan-ish
    "concrete": (200, 200, 200),
}


def fill_mask_gaps(
    composite: np.ndarray,
    fill_zone: np.ndarray,
    bounds: Dict[str, float],
    gap_threshold_m: float = 0.20,
    default_tag: str = "road2",
    debug_dir: str | None = None,
) -> np.ndarray:
    """Fill gaps in composite mask within the pre-built fill zone.

    Args:
        composite: (H, W) uint8 label map, 0=none, 1-5=surface tags.
        fill_zone: (H, W) bool mask — True where filling is allowed.
        bounds: Geo bounds dict with left/right/top/bottom.
        gap_threshold_m: Physical threshold in metres for closing/dilation.
        default_tag: Tag name for remaining void fill (default: road2).
        debug_dir: If set, save intermediate mask images for inspection.

    Returns:
        Filled composite (H, W) uint8, same encoding as input.
    """
    h, w = composite.shape[:2]
    filled = composite.copy()
    fill_zone = fill_zone.astype(bool)

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        _save_debug_colorized(composite, "00_input_composite", debug_dir)
        fz_img = fill_zone.astype(np.uint8) * 255
        cv2.imwrite(os.path.join(debug_dir, "01_fill_zone.png"), fz_img)
        logger.info("  DEBUG: saved 00_input_composite.png, 01_fill_zone.png")

    # Step 1: Identify gap pixels
    gap_pixels = (filled == 0) & fill_zone
    total_gaps = int(np.count_nonzero(gap_pixels))
    total_fill = int(np.count_nonzero(fill_zone))
    logger.info(
        "Gap pixels before fill: %d (%.1f%% of fill zone, %.1f%% of image)",
        total_gaps,
        total_gaps / max(total_fill, 1) * 100,
        total_gaps / max(h * w, 1) * 100,
    )

    if debug_dir:
        gap_img = gap_pixels.astype(np.uint8) * 255
        cv2.imwrite(os.path.join(debug_dir, "02_gap_pixels_initial.png"), gap_img)

    if total_gaps == 0:
        logger.info("No gaps to fill")
        return filled

    # Compute kernel size from physical threshold
    mpp = _compute_meters_per_pixel(bounds, w, h)
    kernel_radius = max(1, int(round(gap_threshold_m / mpp)))
    kernel_size = 2 * kernel_radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    logger.info(
        "Kernel: radius=%d px (%.4f m/px, threshold=%.2f m)",
        kernel_radius, mpp, gap_threshold_m,
    )

    # Step 2: Morphological closing — fill same-tag holes (high→low priority)
    for tag_name in reversed(SURFACE_TAGS):
        tag_id = TAG_NAME_TO_ID[tag_name]
        tag_mask = (filled == tag_id).astype(np.uint8) * 255
        if np.count_nonzero(tag_mask) == 0:
            continue

        closed = cv2.morphologyEx(tag_mask, cv2.MORPH_CLOSE, kernel)
        new_pixels = (closed > 0) & gap_pixels
        n_new = int(np.count_nonzero(new_pixels))
        if n_new > 0:
            filled[new_pixels] = tag_id
            gap_pixels = (filled == 0) & fill_zone
            logger.info("  Closing '%s': filled %d px", tag_name, n_new)

    remaining_after_close = int(np.count_nonzero(gap_pixels))
    logger.info("After closing: %d remaining", remaining_after_close)

    if debug_dir:
        _save_debug_colorized(filled, "03_after_closing", debug_dir)
        gap_img = gap_pixels.astype(np.uint8) * 255
        cv2.imwrite(os.path.join(debug_dir, "03_gaps_after_closing.png"), gap_img)

    # Step 3: Priority dilation — fill cross-tag gaps (high→low priority)
    if remaining_after_close > 0:
        for tag_name in reversed(SURFACE_TAGS):
            tag_id = TAG_NAME_TO_ID[tag_name]
            tag_mask = (filled == tag_id).astype(np.uint8) * 255
            if np.count_nonzero(tag_mask) == 0:
                continue

            dilated = cv2.dilate(tag_mask, kernel, iterations=1)
            new_pixels = (dilated > 0) & gap_pixels
            n_new = int(np.count_nonzero(new_pixels))
            if n_new > 0:
                filled[new_pixels] = tag_id
                gap_pixels = (filled == 0) & fill_zone
                logger.info("  Dilation '%s': filled %d px", tag_name, n_new)

    remaining_after_dilate = int(np.count_nonzero(gap_pixels))
    logger.info("After dilation: %d remaining", remaining_after_dilate)

    if debug_dir:
        _save_debug_colorized(filled, "04_after_dilation", debug_dir)
        gap_img = gap_pixels.astype(np.uint8) * 255
        cv2.imwrite(os.path.join(debug_dir, "04_gaps_after_dilation.png"), gap_img)

    # Step 4: Default fill remaining gaps
    if remaining_after_dilate > 0:
        default_id = TAG_NAME_TO_ID.get(default_tag, TAG_NAME_TO_ID["road2"])
        filled[gap_pixels] = default_id
        logger.info("Default fill (%s): %d pixels", default_tag, remaining_after_dilate)

    final_gaps = int(np.count_nonzero((filled == 0) & fill_zone))
    logger.info("Gap pixels after fill: %d", final_gaps)

    if debug_dir:
        _save_debug_colorized(filled, "05_final_filled", debug_dir)

    return filled


def build_driveable_zone(
    h: int,
    w: int,
    walls: List[Dict[str, Any]],
    wall_resolution: tuple[int, int] | None = None,
    debug_dir: str | None = None,
) -> np.ndarray:
    """Build binary driveable zone from wall polygons.

    driveable = inside(outer_wall) AND NOT inside(obstacle walls)

    Args:
        h, w: Canvas dimensions.
        walls: Wall dicts from walls.json with type, points, closed.
        wall_resolution: (wall_w, wall_h) if walls were generated at a
            different resolution (Stage 2 modelscale). Points are scaled
            to canvas resolution.
        debug_dir: Save debug images if set.

    Returns:
        (H, W) bool mask — True for driveable pixels.
    """
    zone = np.zeros((h, w), dtype=np.uint8)

    # Compute scale factors if wall resolution differs from canvas
    sx, sy = 1.0, 1.0
    if wall_resolution is not None:
        wall_w, wall_h = wall_resolution
        if wall_w > 0 and wall_h > 0 and (wall_w != w or wall_h != h):
            sx = w / wall_w
            sy = h / wall_h
            logger.info(
                "  Wall→canvas scale: wall %dx%d → canvas %dx%d, sx=%.3f, sy=%.3f",
                wall_w, wall_h, w, h, sx, sy,
            )

    def _scale_pts(pts: np.ndarray) -> np.ndarray:
        if sx == 1.0 and sy == 1.0:
            return np.round(pts).astype(np.int32)
        scaled = pts.astype(np.float64)
        scaled[:, 0] *= sx
        scaled[:, 1] *= sy
        return np.round(scaled).astype(np.int32)

    # Classify walls
    outer_pts = None
    obstacle_walls = []
    for wall in walls:
        pts = np.array(wall["points"], dtype=np.float64)
        if len(pts) < 3:
            continue
        wtype = wall.get("type", "")
        if wtype == "outer":
            outer_pts = _scale_pts(pts)
        else:
            obstacle_walls.append(_scale_pts(pts))

    if outer_pts is None:
        logger.warning("No outer wall found — driveable zone is entire image")
        zone[:] = 1
    else:
        cv2.fillPoly(zone, [outer_pts], 1)
        logger.info(
            "  Outer wall: %d points, coverage=%.1f%%",
            len(outer_pts),
            np.count_nonzero(zone) / max(h * w, 1) * 100,
        )

    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, "01a_outer_wall.png"), zone * 255)

    # Subtract obstacle wall interiors
    for pts in obstacle_walls:
        mask_tmp = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask_tmp, [pts], 1)
        zone[mask_tmp > 0] = 0

    if obstacle_walls:
        logger.info(
            "  After %d obstacle walls: coverage=%.1f%%",
            len(obstacle_walls),
            np.count_nonzero(zone) / max(h * w, 1) * 100,
        )

    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, "01b_after_obstacles.png"), zone * 255)

    return zone.astype(bool)


def _compute_meters_per_pixel(
    bounds: Dict[str, float],
    w: int,
    h: int,
) -> float:
    """Compute approximate meters per pixel from geo bounds.

    Handles both geographic (WGS84, degrees) and projected (UTM etc., metres).
    """
    left, right = bounds["left"], bounds["right"]
    bottom, top = bounds["bottom"], bounds["top"]
    geo_w = right - left
    geo_h = top - bottom

    if -180 <= left <= 180 and -180 <= right <= 180 and -90 <= bottom <= 90 and -90 <= top <= 90:
        # Geographic CRS (degrees)
        lat_mid = (top + bottom) / 2.0
        m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat_mid))
        m_per_deg_lat = 111_320.0
        px = geo_w * m_per_deg_lon / w
        py = geo_h * m_per_deg_lat / h
    else:
        # Projected CRS (already in metres)
        px = geo_w / w
        py = geo_h / h

    return (px + py) / 2.0


def _save_debug_colorized(
    labels: np.ndarray,
    name: str,
    debug_dir: str,
) -> None:
    """Save a colorized label map as a debug PNG."""
    h, w = labels.shape[:2]
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for tag_name, tid in TAG_NAME_TO_ID.items():
        color = _TAG_COLORS_BGR.get(tag_name, (128, 128, 128))
        img[labels == tid] = color
    path = os.path.join(debug_dir, f"{name}.png")
    cv2.imwrite(path, img)
    logger.info("  DEBUG: saved %s.png", name)
