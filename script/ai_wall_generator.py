"""
Virtual wall generation for Assetto Corsa tracks.

Fully programmatic approach using SAM3 masks:
  1. Outer wall: flood-fill from road through connected driveable areas.
  2. Obstacle walls: individual typed walls for trees, buildings, water.

Pure Python -- no Blender dependency, no LLM dependency.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mask loading helpers
# ---------------------------------------------------------------------------

def _load_mask(mask_path: Optional[Union[str, Path]], target_shape: Tuple[int, int]) -> Optional[np.ndarray]:
    """Load a mask image and resize to target (H, W) if needed.

    Returns a binary uint8 array (0 or 255), or None if path is invalid.
    """
    if mask_path is None or not Path(str(mask_path)).is_file():
        return None
    mask = np.array(Image.open(str(mask_path)).convert("L"))
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    h, w = target_shape
    if binary.shape[:2] != (h, w):
        binary = cv2.resize(binary, (w, h), interpolation=cv2.INTER_NEAREST)
    return binary


def _subtract_road(mask: np.ndarray, road_binary: np.ndarray) -> np.ndarray:
    """Boolean subtract: mask AND NOT road. Ensures mask doesn't overlap road."""
    return cv2.bitwise_and(mask, cv2.bitwise_not(road_binary))


# ---------------------------------------------------------------------------
# SAM3 flood-fill outer wall
# ---------------------------------------------------------------------------

def _extract_outer_wall_flood(
    aerial_image: np.ndarray,
    road_mask: np.ndarray,
    trees_mask: Optional[np.ndarray] = None,
    grass_mask: Optional[np.ndarray] = None,
    kerb_mask: Optional[np.ndarray] = None,
    sand_mask: Optional[np.ndarray] = None,
    building_mask: Optional[np.ndarray] = None,
    water_mask: Optional[np.ndarray] = None,
    concrete_mask: Optional[np.ndarray] = None,
    simplify_epsilon: float = 3.0,
    buffer_pixels: int = 8,
) -> List[List[int]]:
    """Extract the outer wall by flood-filling from road through connected driveable areas.

    Algorithm:
    1. Visible area = non-black pixels in aerial image
    2. Driveable mask = road | kerb | sand | grass | concrete
    3. Obstacle mask = trees | building | water (minus road — road always wins)
    4. Connectivity = driveable AND NOT obstacle AND visible_area
    5. Flood-fill from road through connectivity
    6. Morphological close + erode for buffer
    7. Extract largest external contour

    No track_proximity_pixels tuning needed — flood-fill naturally stops at
    obstacles and image borders.
    """
    h, w = aerial_image.shape[:2]

    # Step 1: Visible area (non-black in aerial photo)
    gray = cv2.cvtColor(aerial_image, cv2.COLOR_RGB2GRAY)
    _, visible = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    kernel_small = np.ones((3, 3), np.uint8)
    visible = cv2.erode(visible, kernel_small, iterations=3)
    visible = cv2.morphologyEx(visible, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))

    # Step 2: Road binary (required seed)
    _, road_binary = cv2.threshold(road_mask, 127, 255, cv2.THRESH_BINARY)

    # Step 3: Driveable mask = road | kerb | sand | grass | concrete
    driveable = road_binary.copy()
    for mask in (kerb_mask, sand_mask, grass_mask, concrete_mask):
        if mask is not None:
            driveable = cv2.bitwise_or(driveable, mask)

    # Step 4: Obstacle mask = trees | building | water (minus road)
    obstacle = np.zeros((h, w), dtype=np.uint8)
    for mask in (trees_mask, building_mask, water_mask):
        if mask is not None:
            obstacle = cv2.bitwise_or(obstacle, mask)
    # Road always wins over obstacles
    obstacle = cv2.bitwise_and(obstacle, cv2.bitwise_not(road_binary))

    # Step 5: Connectivity = (driveable OR unclassified) AND NOT obstacle AND visible
    # Unclassified = visible pixels not claimed by any mask (gap between road and grass)
    # Allowing flood through unclassified areas bridges SAM3 mask gaps
    any_classified = driveable.copy()
    for mask in (trees_mask, building_mask, water_mask):
        if mask is not None:
            any_classified = cv2.bitwise_or(any_classified, mask)
    unclassified = cv2.bitwise_and(visible, cv2.bitwise_not(any_classified))
    flood_passable = cv2.bitwise_or(driveable, unclassified)
    connectivity = cv2.bitwise_and(flood_passable, cv2.bitwise_not(obstacle))
    connectivity = cv2.bitwise_and(connectivity, visible)

    # Step 6: Flood-fill from road through connectivity
    kernel_dilate = np.ones((3, 3), np.uint8)
    seed = road_binary.copy()
    while True:
        expanded = cv2.dilate(seed, kernel_dilate, iterations=1)
        expanded = cv2.bitwise_and(expanded, connectivity)
        # Also keep original seed pixels
        expanded = cv2.bitwise_or(expanded, seed)
        if np.array_equal(expanded, seed):
            break
        seed = expanded

    # Step 7: Morphological close to bridge small gaps
    kernel_close = np.ones((15, 15), np.uint8)
    playable = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    # Step 8: Erode for safety margin
    if buffer_pixels > 0:
        kernel_buf = np.ones((3, 3), np.uint8)
        playable = cv2.erode(playable, kernel_buf, iterations=buffer_pixels // 3)

    # Step 9: Extract largest external contour
    contours, _ = cv2.findContours(playable, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    if not contours:
        raise RuntimeError("No playable area found via flood-fill")

    outer = max(contours, key=cv2.contourArea)
    simplified = cv2.approxPolyDP(outer, simplify_epsilon, True)
    points = simplified.reshape(-1, 2).tolist()
    logger.info("Outer wall (flood-fill): %d points", len(points))
    return points


# ---------------------------------------------------------------------------
# Typed obstacle walls
# ---------------------------------------------------------------------------

def _extract_obstacle_walls(
    road_mask: np.ndarray,
    outer_wall_points: List[List[int]],
    trees_mask: Optional[np.ndarray] = None,
    building_mask: Optional[np.ndarray] = None,
    water_mask: Optional[np.ndarray] = None,
    min_area: int = 1500,
    wall_buffer_pixels: int = 30,
    simplify_epsilon: float = 4.0,
) -> List[Dict[str, Any]]:
    """Extract individual typed obstacle walls near the outer wall boundary.

    Obstacles sit at and just outside the outer wall (trees block the flood-fill,
    so they end up on the wall boundary).  We capture obstacles within a buffer
    zone around the outer wall perimeter.

    For each obstacle type (tree/building/water):
    1. Constrain to outer wall + buffer zone (catches obstacles just outside)
    2. Subtract road surface
    3. Morphological open to remove noise
    4. Find contours, keep those with area >= min_area
    5. Each contour -> wall entry with specific type
    """
    h, w = road_mask.shape[:2]
    _, road_binary = cv2.threshold(road_mask, 127, 255, cv2.THRESH_BINARY)

    # Fill outer wall polygon
    boundary_mask = np.zeros((h, w), dtype=np.uint8)
    outer_arr = np.array(outer_wall_points, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(boundary_mask, [outer_arr], 255)

    # Expand boundary outward to capture obstacles just outside the wall
    kernel_buf = np.ones((3, 3), np.uint8)
    expanded_boundary = cv2.dilate(boundary_mask, kernel_buf,
                                   iterations=wall_buffer_pixels // 3)

    obstacle_types = [
        ("tree", trees_mask),
        ("building", building_mask),
        ("water", water_mask),
    ]

    walls: List[Dict[str, Any]] = []
    kernel_open = np.ones((7, 7), np.uint8)

    for obstacle_type, mask in obstacle_types:
        if mask is None:
            continue

        # 1. Constrain to outer wall + buffer zone
        constrained = cv2.bitwise_and(mask, expanded_boundary)

        # 2. Subtract road surface
        constrained = cv2.bitwise_and(constrained, cv2.bitwise_not(road_binary))

        # 3. Morphological open to remove noise
        constrained = cv2.morphologyEx(constrained, cv2.MORPH_OPEN, kernel_open)

        # 4. Find contours
        contours, _ = cv2.findContours(constrained, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_TC89_L1)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            simplified = cv2.approxPolyDP(contour, simplify_epsilon, True)
            points = simplified.reshape(-1, 2).tolist()
            walls.append({
                "type": obstacle_type,
                "points": points,
                "closed": True,
            })
            logger.info("Obstacle wall [%s]: %d points, area=%.0f",
                        obstacle_type, len(points), area)

    logger.info("Found %d obstacle walls total", len(walls))
    return walls


# ---------------------------------------------------------------------------
# Wall rendering (for preview/context images)
# ---------------------------------------------------------------------------

def _render_walls_on_image(
    aerial_np: np.ndarray,
    walls: List[Dict[str, Any]],
) -> np.ndarray:
    """Draw walls on the aerial image. Returns a copy with walls in distinct colors."""
    _COLOR_MAP = {
        "outer":        (0, 255, 0),
        "tree":         (0, 100, 0),
        "building":     (128, 0, 128),
        "water":        (0, 191, 255),
        "separation":   (255, 0, 0),
    }
    canvas = aerial_np.copy()
    for wall in walls:
        wtype = wall.get("type", "outer")
        color = _COLOR_MAP.get(wtype, (128, 128, 128))
        pts = np.array(wall["points"], dtype=np.int32).reshape(-1, 1, 2)
        closed = wall.get("closed", True)
        cv2.polylines(canvas, [pts], isClosed=closed, color=color, thickness=3)
    return canvas


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------

def generate_walls_from_masks(
    aerial_image_path: Union[str, Path],
    road_mask_path: Union[str, Path],
    trees_mask_path: Optional[Union[str, Path]] = None,
    grass_mask_path: Optional[Union[str, Path]] = None,
    kerb_mask_path: Optional[Union[str, Path]] = None,
    sand_mask_path: Optional[Union[str, Path]] = None,
    building_mask_path: Optional[Union[str, Path]] = None,
    water_mask_path: Optional[Union[str, Path]] = None,
    concrete_mask_path: Optional[Union[str, Path]] = None,
    min_inner_area: int = 2000,
    output_dir: Optional[Union[str, Path]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Generate wall JSON using SAM3 masks (fully programmatic).

    Steps:
      1. Outer wall (flood-fill from road through connected driveable areas)
      2. Obstacle walls (individual typed walls for trees, buildings, water)

    Returns:
        Wall JSON dict with ``walls`` list.
    """
    aerial = np.array(Image.open(str(aerial_image_path)).convert("RGB"))
    road_mask_raw = np.array(Image.open(str(road_mask_path)).convert("L"))
    h, w = aerial.shape[:2]
    _, road_binary = cv2.threshold(road_mask_raw, 127, 255, cv2.THRESH_BINARY)
    if road_binary.shape[:2] != (h, w):
        road_binary = cv2.resize(road_binary, (w, h), interpolation=cv2.INTER_NEAREST)

    # === Load SAM3 masks ===
    mask_specs = {
        "trees": trees_mask_path,
        "grass": grass_mask_path,
        "kerb": kerb_mask_path,
        "sand": sand_mask_path,
        "building": building_mask_path,
        "water": water_mask_path,
        "concrete": concrete_mask_path,
    }
    masks: Dict[str, Optional[np.ndarray]] = {}
    for name, path in mask_specs.items():
        mask = _load_mask(path, (h, w))
        if mask is not None:
            mask = _subtract_road(mask, road_binary)
            coverage = np.sum(mask > 0) / mask.size
            logger.info("SAM3 %s mask: %.1f%% coverage (road subtracted)", name, coverage * 100)
        masks[name] = mask

    if masks["trees"] is None:
        logger.warning("No trees mask available")

    walls: List[Dict[str, Any]] = []

    # === Step 1: Outer wall (flood-fill) ===
    logger.info("--- Step 1/2: Outer wall (SAM3 flood-fill) ---")
    outer_points = _extract_outer_wall_flood(
        aerial, road_binary,
        trees_mask=masks["trees"], grass_mask=masks["grass"],
        kerb_mask=masks["kerb"], sand_mask=masks["sand"],
        building_mask=masks["building"], water_mask=masks["water"],
        concrete_mask=masks["concrete"],
    )
    walls.append({"type": "outer", "points": outer_points, "closed": True})

    # === Step 2: Obstacle walls (typed) ===
    logger.info("--- Step 2/2: Obstacle walls ---")
    obstacle_walls = _extract_obstacle_walls(
        road_binary, outer_points,
        trees_mask=masks["trees"], building_mask=masks["building"],
        water_mask=masks["water"], min_area=min_inner_area,
    )
    walls.extend(obstacle_walls)

    # Save context image
    if output_dir:
        ctx_img = _render_walls_on_image(aerial, walls)
        ctx_path = Path(str(output_dir)) / "walls_context_programmatic.png"
        Image.fromarray(ctx_img).save(str(ctx_path))
        logger.info("Saved walls context: %s", ctx_path)

    return {"walls": walls}


# Keep backwards-compatible API
def generate_walls_from_mask(
    aerial_image_path: Union[str, Path],
    mask_image_path: Union[str, Path],
    min_inner_area: int = 2000,
) -> Dict[str, Any]:
    """Legacy API: Generate walls from a single road mask."""
    return generate_walls_from_masks(
        aerial_image_path, mask_image_path, min_inner_area=min_inner_area,
    )


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

_VALID_WALL_TYPES = (
    "outer", "tree", "building", "water",
    "inner", "barrier", "separation",  # legacy compat
    "tire_barrier", "guardrail",       # legacy compat
)


def validate_walls_json(data: Any) -> List[str]:
    """Return a list of validation error strings (empty = valid)."""
    errors: List[str] = []
    if not isinstance(data, dict):
        errors.append("Root must be a dict")
        return errors
    walls = data.get("walls")
    if not isinstance(walls, list) or len(walls) == 0:
        errors.append("'walls' must be a non-empty list")
        return errors

    outer_count = 0
    for i, wall in enumerate(walls):
        if not isinstance(wall, dict):
            errors.append(f"walls[{i}] is not a dict")
            continue
        wtype = wall.get("type")
        if wtype not in _VALID_WALL_TYPES:
            errors.append(f"walls[{i}].type must be one of {_VALID_WALL_TYPES}, got {wtype!r}")
        if wtype == "outer":
            outer_count += 1
        pts = wall.get("points")
        if not isinstance(pts, list) or len(pts) < 2:
            errors.append(f"walls[{i}].points must be a list with >= 2 points")
            continue
        for j, pt in enumerate(pts):
            if not isinstance(pt, (list, tuple)) or len(pt) != 2:
                errors.append(f"walls[{i}].points[{j}] must be [x, y]")
            else:
                if not all(isinstance(v, (int, float)) for v in pt):
                    errors.append(f"walls[{i}].points[{j}] contains non-numeric value")

    if outer_count != 1:
        errors.append(f"Exactly 1 outer wall required, found {outer_count}")
    return errors


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_walls(
    image_path: Union[str, Path],
    mask_image_path: Optional[Union[str, Path]] = None,
    *,
    trees_mask_path: Optional[Union[str, Path]] = None,
    grass_mask_path: Optional[Union[str, Path]] = None,
    kerb_mask_path: Optional[Union[str, Path]] = None,
    sand_mask_path: Optional[Union[str, Path]] = None,
    building_mask_path: Optional[Union[str, Path]] = None,
    water_mask_path: Optional[Union[str, Path]] = None,
    concrete_mask_path: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Generate wall polygons (fully programmatic, no LLM).

    Steps:
      1. Outer wall (SAM3 flood-fill from road through driveable areas)
      2. Obstacle walls (typed walls for trees, buildings, water)

    Returns:
        Validated wall JSON dict.
    """
    if mask_image_path is None or not Path(str(mask_image_path)).is_file():
        raise ValueError(
            f"Road mask is required for wall generation, got: {mask_image_path}"
        )

    logger.info("Generating walls via SAM3 flood-fill + obstacle walls (no LLM)")
    result = generate_walls_from_masks(
        image_path, mask_image_path,
        trees_mask_path=trees_mask_path,
        grass_mask_path=grass_mask_path,
        kerb_mask_path=kerb_mask_path,
        sand_mask_path=sand_mask_path,
        building_mask_path=building_mask_path,
        water_mask_path=water_mask_path,
        concrete_mask_path=concrete_mask_path,
        output_dir=output_dir,
    )

    errors = validate_walls_json(result)
    if errors:
        raise ValueError(
            f"Wall JSON validation failed: {errors}\n"
            f"Raw: {json.dumps(result, indent=2)[:500]}"
        )

    return result
