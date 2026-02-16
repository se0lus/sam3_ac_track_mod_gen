"""
Virtual wall generation for Assetto Corsa tracks.

Multi-step approach:
  1. Outer wall (programmatic): boundary of playable area via contour extraction.
  2. Separation walls (programmatic): grass islands between parallel track sections.
  3. Tire barriers (focused LLM): colored tire rows at corners/runoffs.
  4. Guardrails (focused LLM): concrete barriers, guardrails, fences.

Each LLM step receives the aerial image with previous walls drawn on it,
giving visual context to prevent duplication.

Pure Python -- no Blender dependency.
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
# Image preparation
# ---------------------------------------------------------------------------

def prepare_image_for_llm(image_path: Union[str, Path], max_size: int = 2048) -> Image.Image:
    """Scale a track map image so its longest side is at most *max_size* pixels."""
    img = Image.open(str(image_path)).convert("RGB")
    w, h = img.size
    if max(w, h) <= max_size:
        return img
    scale = max_size / max(w, h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return img.resize((new_w, new_h), Image.LANCZOS)


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
# Mask-based contour extraction (primary method)
# ---------------------------------------------------------------------------

def _extract_outer_wall(
    aerial_image: np.ndarray,
    trees_mask: Optional[np.ndarray] = None,
    road_mask: Optional[np.ndarray] = None,
    grass_mask: Optional[np.ndarray] = None,
    simplify_epsilon: float = 3.0,
    buffer_pixels: int = 8,
    track_proximity_pixels: int = 80,
) -> List[List[int]]:
    """Extract the outer wall from the playable area boundary.

    The outer wall defines the outermost boundary of the playable area.
    Strategy:
    1. Start with the road mask, dilate it to create a "track neighborhood"
    2. Remove trees and buildings from the neighborhood
    3. Include grass as driveable buffer
    4. The boundary = outer wall

    The track_proximity_pixels parameter limits how far the wall can be from
    any track surface, preventing it from including distant buildings or
    irrelevant areas.
    """
    h, w = aerial_image.shape[:2]

    # Step 1: Determine visible area (non-black in aerial photo)
    gray = cv2.cvtColor(aerial_image, cv2.COLOR_RGB2GRAY)
    _, visible = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    kernel_small = np.ones((3, 3), np.uint8)
    visible = cv2.erode(visible, kernel_small, iterations=3)
    visible = cv2.morphologyEx(visible, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))

    # Step 2: Get trees mask (SAM3-based, generated in stage 2)
    if trees_mask is not None:
        if trees_mask.shape[:2] != (h, w):
            trees_mask = cv2.resize(trees_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        _, trees_binary = cv2.threshold(trees_mask, 127, 255, cv2.THRESH_BINARY)
    else:
        # No trees mask: use empty (all zeros = no trees to exclude)
        trees_binary = np.zeros((h, w), dtype=np.uint8)

    # Step 3: Create track neighborhood - dilate road mask to define max extent
    if road_mask is not None:
        if road_mask.shape[:2] != (h, w):
            road_mask = cv2.resize(road_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        _, road_binary = cv2.threshold(road_mask, 127, 255, cv2.THRESH_BINARY)

        # Create track neighborhood: everything within track_proximity_pixels of the road
        kernel_prox = np.ones((5, 5), np.uint8)
        track_neighborhood = cv2.dilate(
            road_binary, kernel_prox, iterations=track_proximity_pixels // 5
        )
    else:
        track_neighborhood = visible

    # Step 4: Build playable area
    # Start with track neighborhood, constrain to visible area, exclude trees
    playable = cv2.bitwise_and(track_neighborhood, visible)
    playable = cv2.bitwise_and(playable, cv2.bitwise_not(trees_binary))

    # Ensure road surface is always included
    if road_mask is not None:
        playable = cv2.bitwise_or(playable, road_binary)

    # Include grass if available (pre-processed SAM3 mask, road already subtracted)
    if grass_mask is not None:
        if grass_mask.shape[:2] != (h, w):
            grass_mask = cv2.resize(grass_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        # Only include grass within the track neighborhood
        grass_in_range = cv2.bitwise_and(grass_mask, track_neighborhood)
        playable = cv2.bitwise_or(playable, grass_in_range)

    # Step 5: Close gaps and smooth
    kernel_close = np.ones((15, 15), np.uint8)
    playable = cv2.morphologyEx(playable, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    # Step 6: Erode slightly to create buffer from boundary
    if buffer_pixels > 0:
        kernel_buf = np.ones((3, 3), np.uint8)
        playable = cv2.erode(playable, kernel_buf, iterations=buffer_pixels // 3)

    # Step 7: Find outer contour
    contours, _ = cv2.findContours(playable, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    if not contours:
        raise RuntimeError("No playable area found")

    outer = max(contours, key=cv2.contourArea)
    simplified = cv2.approxPolyDP(outer, simplify_epsilon, True)
    points = simplified.reshape(-1, 2).tolist()
    logger.info("Outer wall: %d points (track neighborhood boundary, excluding trees)", len(points))
    return points


def _extract_separation_walls(
    road_mask: np.ndarray,
    outer_wall_points: List[List[int]],
    trees_mask: Optional[np.ndarray] = None,
    min_area: int = 1500,
    erode_pixels: int = 6,
    road_proximity_pixels: int = 30,
    simplify_epsilon: float = 4.0,
) -> List[List[List[int]]]:
    """Extract separation walls between parallel track sections.

    These are grass/non-road zones *inside* the outer wall that separate
    different track sections.  Unlike the old ``_extract_inner_walls`` (which
    used hole detection and was broken because grass fills all gaps), this:

    1. Fills the outer wall polygon → boundary_mask
    2. separation_zones = boundary AND NOT road AND NOT trees
    3. Erodes inward so walls sit inside grass, not on road edge
    4. Morphological open to remove noise
    5. Keeps only contours with road nearby on >=2 quadrants
       (distinguishes inter-track grass from perimeter grass)
    """
    h, w = road_mask.shape[:2]
    _, road_binary = cv2.threshold(road_mask, 127, 255, cv2.THRESH_BINARY)

    # Step 1: Fill outer wall polygon as boundary
    boundary_mask = np.zeros((h, w), dtype=np.uint8)
    outer_arr = np.array(outer_wall_points, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(boundary_mask, [outer_arr], 255)

    # Step 2: separation_zones = boundary AND NOT road AND NOT trees
    separation = cv2.bitwise_and(boundary_mask, cv2.bitwise_not(road_binary))
    if trees_mask is not None:
        if trees_mask.shape[:2] != (h, w):
            trees_mask = cv2.resize(trees_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        _, trees_binary = cv2.threshold(trees_mask, 127, 255, cv2.THRESH_BINARY)
        separation = cv2.bitwise_and(separation, cv2.bitwise_not(trees_binary))

    # Step 3: Erode inward so walls don't sit on road edge
    if erode_pixels > 0:
        kernel_erode = np.ones((3, 3), np.uint8)
        separation = cv2.erode(separation, kernel_erode, iterations=max(1, erode_pixels // 3))

    # Step 4: Morphological open to remove noise
    kernel_open = np.ones((7, 7), np.uint8)
    separation = cv2.morphologyEx(separation, cv2.MORPH_OPEN, kernel_open)

    # Step 5: Find contours
    contours, _ = cv2.findContours(separation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

    # Step 6: Filter — keep only contours with road on >=2 quadrants
    kernel_dilate = np.ones((5, 5), np.uint8)
    walls: List[List[List[int]]] = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        # Dilate contour region, check road overlap in 4 quadrants
        contour_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
        dilated = cv2.dilate(contour_mask, kernel_dilate,
                             iterations=max(1, road_proximity_pixels // 5))
        check_zone = cv2.bitwise_and(dilated, cv2.bitwise_not(contour_mask))

        # Get centroid for quadrant split
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Check each quadrant for road presence
        quadrants_with_road = 0
        for qy_slice, qx_slice in [
            (slice(0, cy), slice(0, cx)),       # top-left
            (slice(0, cy), slice(cx, w)),        # top-right
            (slice(cy, h), slice(0, cx)),        # bottom-left
            (slice(cy, h), slice(cx, w)),         # bottom-right
        ]:
            quad_check = check_zone[qy_slice, qx_slice]
            quad_road = road_binary[qy_slice, qx_slice]
            overlap = cv2.bitwise_and(quad_check, quad_road)
            if np.any(overlap > 0):
                quadrants_with_road += 1

        if quadrants_with_road >= 2:
            simplified = cv2.approxPolyDP(contour, simplify_epsilon, True)
            points = simplified.reshape(-1, 2).tolist()
            walls.append(points)
            logger.info("Separation wall: %d points, area=%.0f, road in %d quadrants",
                        len(points), area, quadrants_with_road)

    logger.info("Found %d separation walls", len(walls))
    return walls


def _render_walls_on_image(
    aerial_np: np.ndarray,
    walls: List[Dict[str, Any]],
) -> np.ndarray:
    """Draw existing walls on the aerial image for LLM visual context.

    Returns a copy with walls drawn in distinct colors:
      outer → green (0,255,0)
      separation → red (255,0,0)
      tire_barrier → orange (255,165,0)
      guardrail → yellow (255,255,0)
    """
    _COLOR_MAP = {
        "outer":        (0, 255, 0),
        "separation":   (255, 0, 0),
        "tire_barrier": (255, 165, 0),
        "guardrail":    (255, 255, 0),
    }
    canvas = aerial_np.copy()
    for wall in walls:
        wtype = wall.get("type", "outer")
        color = _COLOR_MAP.get(wtype, (128, 128, 128))
        pts = np.array(wall["points"], dtype=np.int32).reshape(-1, 1, 2)
        closed = wall.get("closed", True)
        cv2.polylines(canvas, [pts], isClosed=closed, color=color, thickness=3)
    return canvas


def _scale_wall_coords(
    walls: List[Dict[str, Any]], sx: float, sy: float,
) -> List[Dict[str, Any]]:
    """Return a copy of walls with point coordinates scaled by (sx, sy)."""
    scaled = []
    for w in walls:
        sw = dict(w)
        sw["points"] = [[x * sx, y * sy] for x, y in w["points"]]
        scaled.append(sw)
    return scaled


def generate_walls_from_masks(
    aerial_image_path: Union[str, Path],
    road_mask_path: Union[str, Path],
    trees_mask_path: Optional[Union[str, Path]] = None,
    grass_mask_path: Optional[Union[str, Path]] = None,
    min_inner_area: int = 2000,
    api_key: Optional[str] = None,
    model_name: str = "gemini-2.5-pro",
    temperature: float = 0.2,
    use_llm_barriers: bool = True,
    output_dir: Optional[Union[str, Path]] = None,
    hires_aerial_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Generate wall JSON using 4-step multi-type wall generation.

    Steps:
      1. Outer wall (programmatic contour)
      2. Separation walls (programmatic — grass islands between track sections)
      3. Tire barriers (LLM, with outer+separation context drawn on image)
      4. Guardrails (LLM, with all previous walls drawn on image)

    Args:
        aerial_image_path: Path to the aerial/modelscale image.
        road_mask_path: Path to the merged road binary mask.
        trees_mask_path: Path to the trees mask (optional).
        grass_mask_path: Path to the grass mask (optional).
        min_inner_area: Minimum pixel area for a separation wall island.
        api_key: Gemini API key (for LLM steps 3-4).
        model_name: Gemini model name.
        temperature: LLM temperature.
        use_llm_barriers: Whether to run LLM steps 3-4.
        output_dir: Directory for intermediate context images.
        hires_aerial_path: Optional high-res downscale of GeoTIFF (up to 2048px)
            for sharper VLM input. Coordinates are scaled back to modelscale.

    Returns:
        Wall JSON dict with ``walls`` list.
    """
    aerial = np.array(Image.open(str(aerial_image_path)).convert("RGB"))
    road_mask_raw = np.array(Image.open(str(road_mask_path)).convert("L"))
    h, w = aerial.shape[:2]
    _, road_binary = cv2.threshold(road_mask_raw, 127, 255, cv2.THRESH_BINARY)
    if road_binary.shape[:2] != (h, w):
        road_binary = cv2.resize(road_binary, (w, h), interpolation=cv2.INTER_NEAREST)

    # Load high-res aerial for VLM if available
    hires_aerial: Optional[np.ndarray] = None
    if hires_aerial_path and Path(str(hires_aerial_path)).is_file():
        hires_aerial = np.array(Image.open(str(hires_aerial_path)).convert("RGB"))
        logger.info("High-res aerial loaded: %dx%d (modelscale: %dx%d, %.1fx)",
                     hires_aerial.shape[1], hires_aerial.shape[0], w, h,
                     hires_aerial.shape[1] / w)

    # === Load SAM3 masks ===
    trees_mask = _load_mask(trees_mask_path, (h, w))
    grass_mask = _load_mask(grass_mask_path, (h, w))

    if trees_mask is not None:
        trees_mask = _subtract_road(trees_mask, road_binary)
        coverage = np.sum(trees_mask > 0) / trees_mask.size
        logger.info("SAM3 trees mask: %.1f%% coverage (road subtracted)", coverage * 100)
    else:
        logger.warning("No trees mask available")

    if grass_mask is not None:
        grass_mask = _subtract_road(grass_mask, road_binary)
        coverage = np.sum(grass_mask > 0) / grass_mask.size
        logger.info("SAM3 grass mask: %.1f%% coverage (road subtracted)", coverage * 100)

    walls: List[Dict[str, Any]] = []

    # === Step 1: Outer wall (programmatic) ===
    logger.info("--- Step 1/4: Outer wall ---")
    outer_points = _extract_outer_wall(
        aerial, trees_mask=trees_mask, road_mask=road_binary, grass_mask=grass_mask,
    )
    walls.append({"type": "outer", "points": outer_points, "closed": True})

    # === Step 2: Separation walls (programmatic) ===
    logger.info("--- Step 2/4: Separation walls ---")
    sep_walls = _extract_separation_walls(
        road_binary, outer_points, trees_mask=trees_mask,
        min_area=min_inner_area,
    )
    for pts in sep_walls:
        walls.append({"type": "separation", "points": pts, "closed": True})

    # Save intermediate context image after programmatic steps
    if output_dir:
        ctx_img = _render_walls_on_image(aerial, walls)
        ctx_path = Path(str(output_dir)) / "walls_context_programmatic.png"
        Image.fromarray(ctx_img).save(str(ctx_path))
        logger.info("Saved programmatic walls context: %s", ctx_path)

    # === Step 3: Tire barriers (LLM) ===
    if use_llm_barriers and api_key:
        logger.info("--- Step 3/4: Tire barriers (LLM) ---")
        tire_barriers = _detect_tire_barriers_llm(
            aerial, road_mask_path, walls,
            api_key=api_key, model_name=model_name,
            temperature=temperature, output_dir=output_dir,
            hires_aerial=hires_aerial,
        )
        for b in tire_barriers:
            walls.append({
                "type": "tire_barrier",
                "points": b["points"],
                "closed": b.get("closed", False),
                "description": b.get("description", ""),
            })

        # === Step 4: Guardrails (LLM) ===
        logger.info("--- Step 4/4: Guardrails (LLM) ---")
        guardrails = _detect_guardrails_llm(
            aerial, road_mask_path, walls,
            api_key=api_key, model_name=model_name,
            temperature=temperature, output_dir=output_dir,
            hires_aerial=hires_aerial,
        )
        for b in guardrails:
            walls.append({
                "type": "guardrail",
                "points": b["points"],
                "closed": b.get("closed", False),
                "description": b.get("description", ""),
            })
    else:
        logger.info("--- Steps 3-4 skipped (no API key or LLM disabled) ---")

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
# LLM-based barrier detection (supplement for physical barriers in grass)
# ---------------------------------------------------------------------------

_TIRE_BARRIER_PROMPT = """\
You are a racing game level designer analyzing a {width}x{height} px aerial photo
of a karting/racing circuit.

The image has the following overlays already drawn:
- GREEN lines: outer boundary wall (programmatic, already placed)
- RED lines: separation walls between parallel track sections (already placed)

Your task: identify **tire barriers** visible in the aerial photo.
Tire barriers are rows of colored tires typically placed at:
- Corner apexes (inside of sharp turns)
- Runoff areas at the end of straights
- Chicane entry/exit points

Rules:
- ONLY identify clearly visible tire barrier rows in the photo.
- Do NOT duplicate the green outer wall or red separation walls.
- Barriers must NOT cross the track surface (the dark asphalt).
- Coordinates are pixel [x, y] on the image (origin = top-left).
- Use at least 5 points per barrier to trace the curve.
- If no tire barriers are visible, return an empty list.

Output JSON:
{{
  "additional_barriers": [
    {{
      "description": "tire barrier at turn N apex",
      "points": [[x1,y1], [x2,y2], ...],
      "closed": false
    }}
  ]
}}
"""

_GUARDRAIL_PROMPT = """\
You are a racing game level designer analyzing a {width}x{height} px aerial photo
of a karting/racing circuit.

The image has the following overlays already drawn:
- GREEN lines: outer boundary wall (programmatic, already placed)
- RED lines: separation walls between parallel track sections (already placed)
- ORANGE lines: tire barriers (already placed)

Your task: identify **guardrails, concrete barriers, and fences** visible in the
aerial photo that are NOT already covered by the overlays.
These include:
- Metal guardrails along track edges
- Concrete jersey barriers
- Fence lines between track and spectator areas
- Low walls or permanent structural barriers

Rules:
- ONLY identify clearly visible physical barriers in the photo.
- Do NOT duplicate any existing overlay (green/red/orange).
- Barriers must NOT cross the track surface (the dark asphalt).
- Coordinates are pixel [x, y] on the image (origin = top-left).
- Use at least 5 points per barrier for smooth shapes.
- If no guardrails/barriers are visible, return an empty list.

Output JSON:
{{
  "additional_barriers": [
    {{
      "description": "concrete barrier along back straight",
      "points": [[x1,y1], [x2,y2], ...],
      "closed": false
    }}
  ]
}}
"""


def _detect_llm_barriers(
    aerial_np: np.ndarray,
    road_mask_path: Union[str, Path],
    existing_walls: List[Dict[str, Any]],
    prompt_template: str,
    barrier_label: str,
    api_key: str,
    model_name: str = "gemini-2.5-pro",
    temperature: float = 0.2,
    output_dir: Optional[Union[str, Path]] = None,
    hires_aerial: Optional[np.ndarray] = None,
    context_filename: str = "walls_context.png",
) -> List[Dict[str, Any]]:
    """Shared LLM barrier detection with optional high-res input.

    If *hires_aerial* is provided, walls are drawn on the high-res image for
    sharper VLM input.  Returned coordinates are scaled back to modelscale
    (aerial_np) space before filtering.
    """
    from gemini_client import GeminiClient

    # Determine LLM image and coordinate scale factors
    if hires_aerial is not None:
        sx = hires_aerial.shape[1] / aerial_np.shape[1]
        sy = hires_aerial.shape[0] / aerial_np.shape[0]
        scaled_walls = _scale_wall_coords(existing_walls, sx, sy)
        context_img = _render_walls_on_image(hires_aerial, scaled_walls)
    else:
        sx = sy = 1.0
        context_img = _render_walls_on_image(aerial_np, existing_walls)

    context_pil = Image.fromarray(context_img)

    if output_dir:
        ctx_path = Path(str(output_dir)) / context_filename
        context_pil.save(str(ctx_path))
        logger.info("Saved %s context image: %s (%dx%d)",
                     barrier_label, ctx_path, context_pil.size[0], context_pil.size[1])

    # Scale down for LLM if still too large
    llm_w, llm_h = context_pil.size
    llm_scale = 1.0
    if max(llm_w, llm_h) > 2048:
        llm_scale = 2048 / max(llm_w, llm_h)
        context_pil = context_pil.resize(
            (max(1, int(llm_w * llm_scale)), max(1, int(llm_h * llm_scale))),
            Image.LANCZOS)
        llm_w, llm_h = context_pil.size

    client = GeminiClient(api_key=api_key, model_name=model_name)
    prompt = prompt_template.format(width=llm_w, height=llm_h)

    try:
        result = client.generate_json(prompt, image=context_pil, temperature=temperature)
        barriers = result.get("additional_barriers", [])
        if not isinstance(barriers, list):
            return []
        valid = [b for b in barriers
                 if isinstance(b.get("points"), list) and len(b["points"]) >= 2]

        # Scale coordinates: LLM space → hires space → modelscale space
        total_inv_sx = 1.0 / (sx * llm_scale)
        total_inv_sy = 1.0 / (sy * llm_scale)
        for b in valid:
            b["points"] = [[x * total_inv_sx, y * total_inv_sy]
                           for x, y in b["points"]]

        # Filter against road mask (modelscale resolution)
        outer_pts = None
        for ew in existing_walls:
            if ew.get("type") == "outer":
                outer_pts = ew["points"]
                break
        valid = _filter_barriers_against_mask(
            valid, road_mask_path, outer_wall_points=outer_pts)
        logger.info("LLM detected %d %s(s)", len(valid), barrier_label)
        return valid
    except Exception as e:
        logger.warning("%s LLM detection failed (non-critical): %s", barrier_label, e)
        return []


def _detect_tire_barriers_llm(
    aerial_np: np.ndarray,
    road_mask_path: Union[str, Path],
    existing_walls: List[Dict[str, Any]],
    api_key: str,
    model_name: str = "gemini-2.5-pro",
    temperature: float = 0.2,
    output_dir: Optional[Union[str, Path]] = None,
    hires_aerial: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """Use LLM to detect tire barriers, with existing walls drawn on image."""
    return _detect_llm_barriers(
        aerial_np, road_mask_path, existing_walls,
        prompt_template=_TIRE_BARRIER_PROMPT,
        barrier_label="tire barrier",
        api_key=api_key, model_name=model_name,
        temperature=temperature, output_dir=output_dir,
        hires_aerial=hires_aerial,
        context_filename="walls_context_step3_tires.png",
    )


def _detect_guardrails_llm(
    aerial_np: np.ndarray,
    road_mask_path: Union[str, Path],
    existing_walls: List[Dict[str, Any]],
    api_key: str,
    model_name: str = "gemini-2.5-pro",
    temperature: float = 0.2,
    output_dir: Optional[Union[str, Path]] = None,
    hires_aerial: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """Use LLM to detect guardrails/concrete barriers, with all walls drawn."""
    return _detect_llm_barriers(
        aerial_np, road_mask_path, existing_walls,
        prompt_template=_GUARDRAIL_PROMPT,
        barrier_label="guardrail",
        api_key=api_key, model_name=model_name,
        temperature=temperature, output_dir=output_dir,
        hires_aerial=hires_aerial,
        context_filename="walls_context_step4_guardrails.png",
    )


def _filter_barriers_against_mask(
    barriers: List[Dict[str, Any]],
    road_mask_path: Union[str, Path],
    max_road_overlap: float = 0.3,
    max_median_road_distance: float = 40.0,
    outer_wall_points: Optional[List[List[int]]] = None,
) -> List[Dict[str, Any]]:
    """Remove LLM-detected barriers that are invalid.

    Filters:
      1. Too many points ON the road surface (>max_road_overlap fraction)
      2. Too FAR from the road (median distance > max_median_road_distance px)
      3. Mostly outside the outer wall polygon (if provided)
    """
    road_mask = np.array(Image.open(str(road_mask_path)).convert("L"))
    _, road_binary = cv2.threshold(road_mask, 127, 255, cv2.THRESH_BINARY)
    h, w = road_binary.shape

    # Distance transform: pixel value = distance to nearest road pixel
    road_dist = cv2.distanceTransform(
        cv2.bitwise_not(road_binary), cv2.DIST_L2, 5
    )

    # Outer wall containment mask
    outer_mask = None
    if outer_wall_points is not None:
        outer_mask = np.zeros((h, w), dtype=np.uint8)
        outer_arr = np.array(outer_wall_points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(outer_mask, [outer_arr], 255)

    filtered = []
    for b in barriers:
        pts = b.get("points", [])
        if len(pts) < 2:
            continue
        desc = b.get("description", "?")

        on_road = 0
        distances = []
        outside_wall = 0
        total = len(pts)

        for x, y in pts:
            xi, yi = int(round(x)), int(round(y))
            if 0 <= yi < h and 0 <= xi < w:
                if road_binary[yi, xi] > 0:
                    on_road += 1
                distances.append(road_dist[yi, xi])
                if outer_mask is not None and outer_mask[yi, xi] == 0:
                    outside_wall += 1
            else:
                distances.append(999.0)
                outside_wall += 1

        # Filter 1: Too many points on road
        overlap = on_road / total if total > 0 else 0
        if overlap > max_road_overlap:
            logger.info("Filtered '%s': %.0f%% points on road", desc, overlap * 100)
            continue

        # Filter 2: Too far from road (median distance)
        median_dist = float(np.median(distances)) if distances else 999.0
        if median_dist > max_median_road_distance:
            logger.info("Filtered '%s': median road distance=%.0fpx (max=%d)",
                        desc, median_dist, max_median_road_distance)
            continue

        # Filter 3: Mostly outside outer wall
        if outer_mask is not None and outside_wall > total * 0.5:
            logger.info("Filtered '%s': %d/%d points outside outer wall",
                        desc, outside_wall, total)
            continue

        filtered.append(b)
    return filtered


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

_VALID_WALL_TYPES = ("outer", "inner", "barrier", "separation", "tire_barrier", "guardrail")


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
# Main generation entry point
# ---------------------------------------------------------------------------

def generate_walls(
    image_path: Union[str, Path],
    mask_image_path: Optional[Union[str, Path]] = None,
    *,
    trees_mask_path: Optional[Union[str, Path]] = None,
    grass_mask_path: Optional[Union[str, Path]] = None,
    api_key: Optional[str] = None,
    model_name: str = "gemini-2.5-pro",
    track_description: Optional[str] = None,
    temperature: float = 0.2,
    use_llm_barriers: bool = True,
    output_dir: Optional[Union[str, Path]] = None,
    hires_aerial_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Generate wall polygons using 4-step multi-type approach.

    Steps:
      1. Outer wall (programmatic contour)
      2. Separation walls (programmatic — grass between track sections)
      3. Tire barriers (focused LLM with visual context)
      4. Guardrails (focused LLM with visual context)

    Args:
        image_path: Path to the aerial/modelscale image.
        mask_image_path: Path to the merged road binary mask.
        trees_mask_path: Path to the trees mask (optional).
        grass_mask_path: Path to the grass mask (optional).
        api_key: Gemini API key (for LLM steps 3-4).
        model_name: Gemini model name.
        track_description: Optional track description (LLM-only fallback).
        temperature: LLM temperature.
        use_llm_barriers: Whether to run LLM steps 3-4.
        output_dir: Directory for intermediate context images.

    Returns:
        Validated wall JSON dict.
    """
    if mask_image_path is not None and Path(str(mask_image_path)).is_file():
        logger.info("Generating walls via 4-step multi-type approach")
        result = generate_walls_from_masks(
            image_path, mask_image_path,
            trees_mask_path=trees_mask_path,
            grass_mask_path=grass_mask_path,
            api_key=api_key, model_name=model_name,
            temperature=temperature,
            use_llm_barriers=use_llm_barriers,
            output_dir=output_dir,
            hires_aerial_path=hires_aerial_path,
        )
    else:
        logger.info("No mask available, falling back to LLM-only wall generation")
        result = _generate_walls_llm_only(
            image_path, api_key=api_key, model_name=model_name,
            track_description=track_description, temperature=temperature,
        )

    errors = validate_walls_json(result)
    if errors:
        raise ValueError(
            f"Wall JSON validation failed: {errors}\n"
            f"Raw: {json.dumps(result, indent=2)[:500]}"
        )

    return result


def _generate_walls_llm_only(
    image_path: Union[str, Path],
    *,
    api_key: Optional[str] = None,
    model_name: str = "gemini-2.5-pro",
    track_description: Optional[str] = None,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """Fallback: LLM-only wall generation when no mask is available."""
    from gemini_client import GeminiClient

    if api_key is None:
        api_key = "***REDACTED_GEMINI_KEY***"

    client = GeminiClient(api_key=api_key, model_name=model_name)
    img = prepare_image_for_llm(image_path)
    w, h = img.size

    prompt = (
        f"You are a racing game level designer. Analyze this {w}x{h} aerial "
        f"photo of a racing track and generate virtual wall polygons.\n\n"
        f"Output JSON with 'walls' array. Each wall has 'type' (outer/inner), "
        f"'points' ([[x,y],...]), 'closed' (true). One outer wall surrounding "
        f"the entire playable area, inner walls for grass islands between track "
        f"sections. Pixel coordinates, origin=top-left."
    )
    if track_description:
        prompt += f"\nTrack info: {track_description}"

    return client.generate_json(prompt, image=img, temperature=temperature)
