"""
AI-powered game object generation for Assetto Corsa tracks.

Hybrid approach:
- VLM (Gemini) places layout-dependent objects: hotlap_start, pit boxes, start grid, timing_0.
- Programmatic analysis places timing points at bend exits using road centerline.

Per-type VLM generation with mask-based validation and retry.

Pure Python -- no Blender dependency.
"""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("sam3_pipeline.game_objects")


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def _load_image(path: Union[str, Path], max_size: int = 3072) -> Image.Image:
    """Load and optionally downscale an image for VLM input."""
    img = Image.open(str(path)).convert("RGB")
    if max(img.size) > max_size:
        ratio = max_size / max(img.size)
        img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
    return img


# ---------------------------------------------------------------------------
# Geo metadata → pixel_size_m
# ---------------------------------------------------------------------------

def _compute_pixel_size_m(geo_metadata_path: str, image_shape: Tuple[int, int]) -> float:
    """Compute average pixel size in meters from geo metadata.

    Args:
        geo_metadata_path: Path to result_masks.json (stage 2 output).
        image_shape: (height, width) of the mask image.

    Returns:
        Average meters per pixel.
    """
    with open(geo_metadata_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # result_masks.json has meta.model_scale_size and meta.geo.bounds
    # geo_metadata.json has image_width/height and bounds
    if "meta" in data:
        meta = data["meta"]
        model_scale = meta.get("model_scale_size", {})
        w = model_scale.get("width", image_shape[1])
        h = model_scale.get("height", image_shape[0])
        geo_bounds = meta.get("geo", {}).get("bounds", {})
    else:
        w = data.get("image_width", image_shape[1])
        h = data.get("image_height", image_shape[0])
        geo_bounds = data.get("bounds", {})

    left = geo_bounds.get("left", geo_bounds.get("west", 0))
    right = geo_bounds.get("right", geo_bounds.get("east", 0))
    bottom = geo_bounds.get("bottom", geo_bounds.get("south", 0))
    top = geo_bounds.get("top", geo_bounds.get("north", 0))

    # Convert extent to meters (handle both geographic and projected CRS)
    geo_w = abs(right - left)
    geo_h = abs(top - bottom)

    if (-180 <= left <= 180 and -180 <= right <= 180
            and -90 <= bottom <= 90 and -90 <= top <= 90):
        # Geographic CRS (degrees)
        lat_mid = (top + bottom) / 2.0
        cos_lat = math.cos(math.radians(lat_mid))
        width_m = geo_w * 111_320.0 * cos_lat
        height_m = geo_h * 111_320.0
    else:
        # Projected CRS (already in metres)
        width_m = geo_w
        height_m = geo_h

    px_w = width_m / max(w, 1)
    px_h = height_m / max(h, 1)
    pixel_size_m = (px_w + px_h) / 2.0

    logger.info("pixel_size_m = %.4f  (image %dx%d, extent %.1f x %.1f m)",
                pixel_size_m, w, h, width_m, height_m)
    return pixel_size_m


# ---------------------------------------------------------------------------
# ValidationMasks
# ---------------------------------------------------------------------------

class ValidationMasks:
    """Mask-based validation for game object positions, using physical-unit parameters."""

    def __init__(
        self,
        road_mask: np.ndarray,
        near_road_mask: np.ndarray,
        concrete_mask: Optional[np.ndarray],
        grass_mask: Optional[np.ndarray],
        trees_mask: Optional[np.ndarray],
        water_mask: Optional[np.ndarray],
        pixel_size_m: float,
    ):
        self.road = road_mask            # layout mask: >127 = road surface
        self.near_road = near_road_mask  # dilated road: >127 = near road
        self.concrete = concrete_mask    # paved surfaces (valid for pits)
        self.grass = grass_mask
        self.trees = trees_mask
        self.water = water_mask
        self.pixel_size_m = pixel_size_m
        self.h, self.w = road_mask.shape[:2]

    @staticmethod
    def load(
        layout_mask_path: str,
        mask_full_map_dir: str,
        geo_metadata_path: str,
        pit_max_distance_m: float = 30.0,
    ) -> "ValidationMasks":
        """Load all masks + compute pixel_size_m from geo metadata."""
        road = cv2.imread(layout_mask_path, cv2.IMREAD_GRAYSCALE)
        if road is None:
            raise FileNotFoundError(f"Cannot read layout mask: {layout_mask_path}")

        # Compute pixel_size_m
        pixel_size_m = _compute_pixel_size_m(geo_metadata_path, road.shape)

        # Dilate road mask by pit_max_distance_m (in pixels)
        dilation_px = int(pit_max_distance_m / pixel_size_m)
        kernel_size = max(dilation_px * 2 + 1, 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        near = cv2.dilate(road, kernel)

        # Load surface masks
        def _load_mask(name: str) -> Optional[np.ndarray]:
            p = os.path.join(mask_full_map_dir, name)
            if os.path.isfile(p):
                return cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            return None

        return ValidationMasks(
            road_mask=road,
            near_road_mask=near,
            concrete_mask=_load_mask("concrete_mask.png"),
            grass_mask=_load_mask("grass_mask.png"),
            trees_mask=_load_mask("trees_mask.png"),
            water_mask=_load_mask("water_mask.png"),
            pixel_size_m=pixel_size_m,
        )

    def _in_bounds(self, x: float, y: float) -> bool:
        ix, iy = int(round(x)), int(round(y))
        return 0 <= ix < self.w and 0 <= iy < self.h

    def is_on_road(self, x: float, y: float) -> bool:
        """Check if position is on the road surface (layout mask)."""
        if not self._in_bounds(x, y):
            return False
        ix, iy = int(round(x)), int(round(y))
        return self.road[iy, ix] > 127

    def is_near_road(self, x: float, y: float) -> bool:
        """Check if position is within pit_max_distance of road."""
        if not self._in_bounds(x, y):
            return False
        ix, iy = int(round(x)), int(round(y))
        return self.near_road[iy, ix] > 127

    def is_on_paved(self, x: float, y: float) -> bool:
        """On concrete/asphalt surface (valid for pits)."""
        if not self._in_bounds(x, y):
            return False
        ix, iy = int(round(x)), int(round(y))
        if self.concrete is not None and self.concrete[iy, ix] > 127:
            return True
        return False

    def is_on_invalid_surface(self, x: float, y: float) -> bool:
        """On grass, trees, or water (NOT valid for pits)."""
        if not self._in_bounds(x, y):
            return False
        ix, iy = int(round(x)), int(round(y))
        for m in (self.grass, self.trees, self.water):
            if m is not None and m[iy, ix] > 127:
                return True
        return False


# ---------------------------------------------------------------------------
# Per-Type VLM Prompts
# ---------------------------------------------------------------------------

_HOTLAP_PROMPT = """\
You are an expert Assetto Corsa track designer. You will receive a top-down \
2D image of a racing track (image size: {w} x {h} pixels). A second image is \
the segmentation mask (white = racing surface).

The track is driven in a **{direction}** direction.

Place exactly 1 object: **AC_HOTLAP_START_0**
- It must be ON the track surface (white area in the mask image).
- Place it on the main straight, just after the last corner exit \
(in {direction} driving direction) and before the start/finish line.
- orientation_z is a 2D unit vector [dx, dy] pointing in the forward driving direction.
- Coordinates are pixel positions [x, y] on the image (origin = top-left).

Return ONLY valid JSON, no markdown fences:
{{
  "objects": [
    {{
      "name": "AC_HOTLAP_START_0",
      "position": [x, y],
      "orientation_z": [dx, dy],
      "type": "hotlap_start"
    }}
  ]
}}
"""

_PIT_PROMPT = """\
You are an expert Assetto Corsa track designer. You will receive a top-down \
2D image of a racing track (image size: {w} x {h} pixels). A second image is \
the segmentation mask (white = racing surface).

The track is driven in a **{direction}** direction.

Place exactly {pit_count} pit boxes: **AC_PIT_0** through **AC_PIT_{pit_max}**.
- Pit boxes must be in the pit lane — a paved concrete/asphalt area BESIDE \
the main straight.
- They must NOT be on the racing surface (white area in the mask).
- They must NOT be on grass, trees, or water.
- They must be on a hard, flat paved surface adjacent to the track.
- Space them evenly along a straight line, all facing the same driving direction.
- orientation_z is a 2D unit vector [dx, dy] pointing in the forward driving direction.
- Coordinates are pixel positions [x, y] on the image (origin = top-left).

Return ONLY valid JSON, no markdown fences:
{{
  "objects": [
    {{
      "name": "AC_PIT_0",
      "position": [x, y],
      "orientation_z": [dx, dy],
      "type": "pit"
    }},
    ...
  ]
}}
"""

_START_PROMPT = """\
You are an expert Assetto Corsa track designer. You will receive a top-down \
2D image of a racing track (image size: {w} x {h} pixels). A second image is \
the segmentation mask (white = racing surface).

The track is driven in a **{direction}** direction.

Place exactly {start_count} start grid positions: **AC_START_0** through \
**AC_START_{start_max}**.
- All must be ON the racing surface (white area in the mask).
- AC_START_0 is pole position (closest to the start/finish line).
- Arrange them in a staggered 2-wide grid pattern along the driving direction.
- orientation_z is a 2D unit vector [dx, dy] pointing in the forward driving direction.
- Coordinates are pixel positions [x, y] on the image (origin = top-left).

Return ONLY valid JSON, no markdown fences:
{{
  "objects": [
    {{
      "name": "AC_START_0",
      "position": [x, y],
      "orientation_z": [dx, dy],
      "type": "start"
    }},
    ...
  ]
}}
"""

_TIMING0_PROMPT = """\
You are an expert Assetto Corsa track designer. You will receive a top-down \
2D image of a racing track (image size: {w} x {h} pixels). A second image is \
the segmentation mask (white = racing surface).

The track is driven in a **{direction}** direction.

Place exactly 1 object: **AC_TIME_0** — the start/finish line position.
- It should be on or very near the main straight, where a start/finish line \
would typically be placed.
- The start/finish line is usually near the pit exit, on a straight section.
- Just provide one position [x, y] — the program will compute L/R points from this.
- orientation_z is a 2D unit vector [dx, dy] pointing in the forward driving direction.
- Coordinates are pixel positions [x, y] on the image (origin = top-left).

Return ONLY valid JSON, no markdown fences:
{{
  "objects": [
    {{
      "name": "AC_TIME_0",
      "position": [x, y],
      "orientation_z": [dx, dy],
      "type": "timing_0"
    }}
  ]
}}
"""


def _build_per_type_prompt(
    obj_type: str,
    image_size: Tuple[int, int],
    track_direction: str = "clockwise",
    pit_count: int = 8,
    start_count: int = 8,
    start_point_hint: Optional[Dict[str, Any]] = None,
) -> str:
    """Build VLM prompt for a specific object type.

    Args:
        start_point_hint: Optional dict with "position" [x,y] and "direction" [dx,dy]
            of the user-specified start/finish line. Injected into pit/start prompts.
    """
    w, h = image_size
    fmt = dict(
        w=w, h=h, direction=track_direction,
        pit_count=pit_count, pit_max=pit_count - 1,
        start_count=start_count, start_max=start_count - 1,
    )

    templates = {
        "hotlap": _HOTLAP_PROMPT,
        "pit": _PIT_PROMPT,
        "start": _START_PROMPT,
        "timing_0": _TIMING0_PROMPT,
    }

    template = templates.get(obj_type)
    if template is None:
        raise ValueError(f"Unknown object type for VLM prompt: {obj_type}")

    prompt = template.format(**fmt)

    # Inject start point context for pit and start prompts
    if start_point_hint and obj_type in ("pit", "start"):
        pos = start_point_hint["position"]
        dir_ = start_point_hint["direction"]
        if obj_type == "pit":
            prompt += (
                f"\n\nIMPORTANT CONTEXT: The start/finish line is at pixel position "
                f"[{pos[0]:.0f}, {pos[1]:.0f}] with driving direction "
                f"[{dir_[0]:.3f}, {dir_[1]:.3f}]. Place pit boxes in the pit lane area "
                f"near the start/finish line."
            )
        else:  # start
            prompt += (
                f"\n\nIMPORTANT CONTEXT: The start/finish line is at pixel position "
                f"[{pos[0]:.0f}, {pos[1]:.0f}] with driving direction "
                f"[{dir_[0]:.3f}, {dir_[1]:.3f}]. AC_START_0 (pole position) should be "
                f"just before this line, with the grid extending behind it in the "
                f"driving direction."
            )

    return prompt


# ---------------------------------------------------------------------------
# Schema validation (unchanged from before)
# ---------------------------------------------------------------------------

_REQUIRED_TYPES = {"hotlap_start"}
_KNOWN_TYPES = {"hotlap_start", "pit", "start", "timing_left", "timing_right", "timing_0"}

_NAME_TYPE_MAP = {
    "AC_HOTLAP_START": "hotlap_start",
    "AC_PIT": "pit",
    "AC_START": "start",
    "AC_TIME": "timing_left",
}


def _infer_type(obj: Dict[str, Any]) -> str:
    """Infer object type from name if type field is missing."""
    existing = obj.get("type")
    if existing and existing in _KNOWN_TYPES:
        return existing
    name = obj.get("name", "")
    if "_L" in name:
        return "timing_left"
    if "_R" in name:
        return "timing_right"
    for prefix, otype in _NAME_TYPE_MAP.items():
        if name.startswith(prefix):
            return otype
    return existing or "unknown"


def validate_game_objects_json(data: Any) -> List[str]:
    """Return a list of validation error strings (empty = valid)."""
    errors: List[str] = []
    if not isinstance(data, dict):
        errors.append("Root must be a dict")
        return errors

    direction = data.get("track_direction")
    if not isinstance(direction, str) or direction not in ("clockwise", "counterclockwise"):
        errors.append(f"'track_direction' must be 'clockwise' or 'counterclockwise', got {direction!r}")

    objects = data.get("objects")
    if not isinstance(objects, list) or len(objects) == 0:
        errors.append("'objects' must be a non-empty list")
        return errors

    found_types: set = set()
    for i, obj in enumerate(objects):
        if not isinstance(obj, dict):
            errors.append(f"objects[{i}] is not a dict")
            continue
        name = obj.get("name")
        if not isinstance(name, str) or not name:
            errors.append(f"objects[{i}].name must be a non-empty string")

        pos = obj.get("position")
        if not isinstance(pos, (list, tuple)) or len(pos) != 2:
            errors.append(f"objects[{i}].position must be [x, y]")
        elif not all(isinstance(v, (int, float)) for v in pos):
            errors.append(f"objects[{i}].position contains non-numeric values")

        orient = obj.get("orientation_z")
        if not isinstance(orient, (list, tuple)) or len(orient) != 2:
            errors.append(f"objects[{i}].orientation_z must be [dx, dy]")
        elif not all(isinstance(v, (int, float)) for v in orient):
            errors.append(f"objects[{i}].orientation_z contains non-numeric values")

        otype = obj.get("type")
        if isinstance(otype, str):
            found_types.add(otype)

    for rt in _REQUIRED_TYPES:
        if rt not in found_types:
            errors.append(f"Missing required object type: {rt}")

    return errors


# ---------------------------------------------------------------------------
# Per-type validation functions
# ---------------------------------------------------------------------------

def _validate_hotlap(objects: List[Dict], masks: ValidationMasks) -> Dict[str, Any]:
    """Hotlap must be ON road."""
    total = len(objects)
    passed = 0
    for obj in objects:
        x, y = obj["position"]
        if masks.is_on_road(x, y):
            passed += 1
    return {"total": total, "passed": passed, "rule": "on_road"}


def _validate_pits(objects: List[Dict], masks: ValidationMasks) -> Dict[str, Any]:
    """Pits: near road, NOT on racing surface, preferably on paved, NOT on invalid surface."""
    total = len(objects)
    passed = 0
    for obj in objects:
        x, y = obj["position"]
        on_road = masks.is_on_road(x, y)
        near_road = masks.is_near_road(x, y)
        on_invalid = masks.is_on_invalid_surface(x, y)
        # Valid: near road AND NOT on racing surface AND NOT on invalid surface
        if near_road and not on_road and not on_invalid:
            passed += 1
    return {"total": total, "passed": passed, "rule": "near_road+not_on_road+not_invalid"}


def _validate_starts(objects: List[Dict], masks: ValidationMasks) -> Dict[str, Any]:
    """Starts must be ON road."""
    total = len(objects)
    passed = 0
    for obj in objects:
        x, y = obj["position"]
        if masks.is_on_road(x, y):
            passed += 1
    return {"total": total, "passed": passed, "rule": "on_road"}


def _validate_timing0(objects: List[Dict], masks: ValidationMasks) -> Dict[str, Any]:
    """Timing_0 should be near road (will be snapped)."""
    total = len(objects)
    passed = 0
    for obj in objects:
        x, y = obj["position"]
        if masks.is_near_road(x, y):
            passed += 1
    return {"total": total, "passed": passed, "rule": "near_road"}


_VALIDATORS: Dict[str, Callable] = {
    "hotlap": _validate_hotlap,
    "pit": _validate_pits,
    "start": _validate_starts,
    "timing_0": _validate_timing0,
}


# ---------------------------------------------------------------------------
# Snap-to-road post-processing
# ---------------------------------------------------------------------------

def _snap_to_road(
    objects: List[Dict[str, Any]],
    obj_type: str,
    masks: "ValidationMasks",
    max_search_m: float = 30.0,
) -> int:
    """Snap object positions to the nearest valid surface (in-place).

    For hotlap/start: snap to nearest road pixel.
    For pit: snap to nearest near-road, not-on-road, not-invalid pixel.

    Returns:
        Number of objects that were snapped.
    """
    max_search_px = int(max_search_m / masks.pixel_size_m)
    snapped = 0

    for obj in objects:
        pos = obj.get("position")
        if not pos or len(pos) < 2:
            continue
        x, y = float(pos[0]), float(pos[1])

        # Check if already valid
        if obj_type in ("hotlap", "start"):
            if masks.is_on_road(x, y):
                continue
            target_mask = masks.road
            threshold = 127
        elif obj_type == "pit":
            on_road = masks.is_on_road(x, y)
            near_road = masks.is_near_road(x, y)
            on_invalid = masks.is_on_invalid_surface(x, y)
            if near_road and not on_road and not on_invalid:
                continue
            # For pits: snap to near_road area that's not on road and not invalid
            target_mask = masks.near_road
            threshold = 127
        else:
            continue

        # Search for nearest valid pixel using expanding square rings
        best_dist = float("inf")
        best_pos = None
        ix, iy = int(round(x)), int(round(y))

        for r in range(1, max_search_px + 1):
            if best_dist <= r:
                break  # can't find closer
            for dy in range(-r, r + 1):
                for dx in [-r, r] if abs(dy) < r else range(-r, r + 1):
                    nx, ny = ix + dx, iy + dy
                    if not (0 <= nx < masks.w and 0 <= ny < masks.h):
                        continue
                    if target_mask[ny, nx] <= threshold:
                        continue
                    # Extra checks for pits
                    if obj_type == "pit":
                        if masks.road[ny, nx] > 127:
                            continue  # don't snap pit onto road
                        if masks.is_on_invalid_surface(float(nx), float(ny)):
                            continue
                    dist = math.sqrt(dx * dx + dy * dy)
                    if dist < best_dist:
                        best_dist = dist
                        best_pos = (nx, ny)

        if best_pos is not None:
            old_pos = [round(x), round(y)]
            obj["position"] = [best_pos[0], best_pos[1]]
            snapped += 1
            dist_m = best_dist * masks.pixel_size_m
            logger.info("Snapped %s from %s to %s (%.1fm)",
                        obj.get("name", "?"), old_pos, obj["position"], dist_m)

    return snapped


# ---------------------------------------------------------------------------
# VLM call with retry + snap
# ---------------------------------------------------------------------------

def _call_vlm_with_retry(
    client: Any,
    prompt: str,
    images: List[Image.Image],
    obj_type: str,
    masks: Optional["ValidationMasks"],
    temperature: float = 0.2,
    pass_threshold: float = 0.7,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Call VLM, validate result, retry once if below threshold, snap if needed.

    Returns:
        (objects, validation_result)
    """
    def _do_call() -> List[Dict[str, Any]]:
        if len(images) == 1:
            result = client.generate_json(prompt, image=images[0], temperature=temperature)
        else:
            result = client.generate_json(prompt, images=images, temperature=temperature)
        objects = result.get("objects", [])
        for obj in objects:
            obj["type"] = _infer_type(obj)
        return objects

    validator = _VALIDATORS.get(obj_type)

    # Attempt 1
    logger.info("VLM call [%s] attempt 1...", obj_type)
    objects_1 = _do_call()
    logger.info("VLM [%s] attempt 1 returned %d objects", obj_type, len(objects_1))

    if masks is None or validator is None:
        return objects_1, {"total": len(objects_1), "passed": len(objects_1), "rule": "none"}

    val_1 = validator(objects_1, masks)
    pass_rate_1 = val_1["passed"] / max(val_1["total"], 1)
    logger.info("VLM [%s] attempt 1 validation: %d/%d (%.0f%%)",
                obj_type, val_1["passed"], val_1["total"], pass_rate_1 * 100)

    if pass_rate_1 >= pass_threshold:
        return objects_1, val_1

    # Attempt 2
    logger.info("VLM call [%s] attempt 2 (validation below %.0f%%)...", obj_type, pass_threshold * 100)
    try:
        objects_2 = _do_call()
        logger.info("VLM [%s] attempt 2 returned %d objects", obj_type, len(objects_2))

        val_2 = validator(objects_2, masks)
        pass_rate_2 = val_2["passed"] / max(val_2["total"], 1)
        logger.info("VLM [%s] attempt 2 validation: %d/%d (%.0f%%)",
                    obj_type, val_2["passed"], val_2["total"], pass_rate_2 * 100)

        # Use whichever attempt had better pass rate
        if pass_rate_2 >= pass_rate_1:
            best_objects, best_val = objects_2, val_2
        else:
            best_objects, best_val = objects_1, val_1
    except Exception as e:
        logger.warning("VLM [%s] attempt 2 failed: %s, using attempt 1 result", obj_type, e)
        best_objects, best_val = objects_1, val_1

    # Snap-to-road post-processing for objects that failed validation
    best_rate = best_val["passed"] / max(best_val["total"], 1)
    if best_rate < 1.0 and obj_type in ("hotlap", "start", "pit"):
        n_snapped = _snap_to_road(best_objects, obj_type, masks)
        if n_snapped > 0:
            best_val = validator(best_objects, masks)
            new_rate = best_val["passed"] / max(best_val["total"], 1)
            logger.info("VLM [%s] after snap: %d/%d (%.0f%%), snapped %d objects",
                        obj_type, best_val["passed"], best_val["total"],
                        new_rate * 100, n_snapped)

    return best_objects, best_val


# ---------------------------------------------------------------------------
# Sequential orchestrator (per-type VLM calls)
# ---------------------------------------------------------------------------

def generate_all_vlm_sequential(
    image_path: Union[str, Path],
    mask_path: Optional[Union[str, Path]],
    track_direction: str,
    masks: Optional[ValidationMasks],
    *,
    pit_count: int = 8,
    start_count: int = 8,
    api_key: Optional[str] = None,
    model_name: str = "gemini-2.5-pro",
    temperature: float = 0.2,
    modelscale_size: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    """Generate all VLM objects sequentially with per-type validation + retry.

    Args:
        modelscale_size: (w, h) of the modelscale image. When the VLM input
            image is higher resolution (vlmscale), coordinates are scaled
            back to modelscale space so downstream masks/centerline stay
            consistent.

    Returns:
        dict with keys: hotlap, pits, starts, timing_0_raw, validation
    """
    from gemini_client import GeminiClient

    if api_key is None:
        api_key = "***REDACTED_GEMINI_KEY***"

    client = GeminiClient(api_key=api_key, model_name=model_name)

    img = _load_image(image_path)
    actual_size = (img.width, img.height)

    # Use modelscale dims in prompts so VLM returns modelscale-range coords
    # even when seeing a higher-resolution vlmscale image.
    prompt_size = modelscale_size if modelscale_size is not None else actual_size
    if modelscale_size is not None and actual_size != modelscale_size:
        logger.info("VLM image %s, prompt dims %s (modelscale)", actual_size, prompt_size)

    images: List[Image.Image] = [img]
    if mask_path is not None:
        mask_img = _load_image(mask_path)
        # Resize mask to match base image so VLM sees consistent visuals
        if mask_img.size != img.size:
            mask_img = mask_img.resize(img.size, Image.NEAREST)
        images.append(mask_img)

    validation_results = {}

    # 1. Hotlap
    logger.info("=== Generating hotlap_start ===")
    prompt = _build_per_type_prompt("hotlap", prompt_size, track_direction)
    if mask_path:
        prompt += "\n\nA second image is the segmentation mask of this track."
    hotlap_objs, val = _call_vlm_with_retry(
        client, prompt, images, "hotlap", masks, temperature)
    validation_results["hotlap"] = val

    # 2. Pits
    logger.info("=== Generating pit boxes ===")
    prompt = _build_per_type_prompt("pit", prompt_size, track_direction, pit_count=pit_count)
    if mask_path:
        prompt += "\n\nA second image is the segmentation mask. White = racing surface. " \
                  "Pit boxes go on paved area BESIDE the white area, not on it."
    pit_objs, val = _call_vlm_with_retry(
        client, prompt, images, "pit", masks, temperature)
    validation_results["pit"] = val

    # 3. Starts
    logger.info("=== Generating start grid ===")
    prompt = _build_per_type_prompt("start", prompt_size, track_direction, start_count=start_count)
    if mask_path:
        prompt += "\n\nA second image is the segmentation mask. White = racing surface. " \
                  "All start positions must be ON the white area."
    start_objs, val = _call_vlm_with_retry(
        client, prompt, images, "start", masks, temperature)
    validation_results["start"] = val

    # 4. Timing_0 (start/finish line)
    logger.info("=== Generating timing_0 (start/finish line) ===")
    prompt = _build_per_type_prompt("timing_0", prompt_size, track_direction)
    if mask_path:
        prompt += "\n\nA second image is the segmentation mask. White = racing surface."
    timing0_objs, val = _call_vlm_with_retry(
        client, prompt, images, "timing_0", masks, temperature)
    validation_results["timing_0"] = val

    logger.info("Sequential VLM generation complete: %d hotlap, %d pits, %d starts, %d timing_0",
                len(hotlap_objs), len(pit_objs), len(start_objs), len(timing0_objs))

    return {
        "hotlap": hotlap_objs,
        "pits": pit_objs,
        "starts": start_objs,
        "timing_0_raw": timing0_objs,
        "validation": validation_results,
    }


def generate_single_type_vlm(
    obj_type: str,
    image_path: Union[str, Path],
    mask_path: Optional[Union[str, Path]],
    track_direction: str,
    masks: Optional[ValidationMasks],
    *,
    pit_count: int = 8,
    start_count: int = 8,
    api_key: Optional[str] = None,
    model_name: str = "gemini-2.5-pro",
    temperature: float = 0.2,
    modelscale_size: Optional[Tuple[int, int]] = None,
    start_point_hint: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Generate a single type of VLM objects with validation + retry.

    Args:
        obj_type: "hotlap", "pit", "start", or "timing_0".
        modelscale_size: (w, h) for coordinate scaling from vlmscale to modelscale.
        start_point_hint: Optional dict with "position" and "direction" for
            injecting start/finish context into pit/start prompts.

    Returns:
        (objects, validation_result)
    """
    from gemini_client import GeminiClient

    if api_key is None:
        api_key = "***REDACTED_GEMINI_KEY***"

    client = GeminiClient(api_key=api_key, model_name=model_name)

    img = _load_image(image_path)
    actual_size = (img.width, img.height)
    prompt_size = modelscale_size if modelscale_size is not None else actual_size

    images: List[Image.Image] = [img]
    if mask_path is not None:
        mask_img = _load_image(mask_path)
        if mask_img.size != img.size:
            mask_img = mask_img.resize(img.size, Image.NEAREST)
        images.append(mask_img)

    prompt = _build_per_type_prompt(obj_type, prompt_size, track_direction,
                                     pit_count=pit_count, start_count=start_count,
                                     start_point_hint=start_point_hint)
    if mask_path:
        prompt += "\n\nA second image is the segmentation mask. White = racing surface."

    return _call_vlm_with_retry(client, prompt, images, obj_type, masks, temperature)


# ---------------------------------------------------------------------------
# Legacy VLM generation (backward compat)
# ---------------------------------------------------------------------------

_VLM_SYSTEM_PROMPT = """\
You are an expert Assetto Corsa track designer.  You will receive a top-down
2D image of a racing track.  Your job is to place game objects on the track.

CRITICAL: The track is driven in a **{direction}** direction.  All object
orientations (orientation_z = forward driving direction as a unit vector) and
positional logic MUST follow this direction.

Objects to place:
1. **AC_HOTLAP_START_0** -- exactly 1.  Place it on the track surface, just
   after a corner exit and before the start/finish line, facing the driving
   direction.
2. **AC_PIT_0** through **AC_PIT_N** -- at least 8 pit boxes.  Place them
   along a straight section beside the track (the pit lane).  Space them
   evenly.  All face the same driving direction.
3. **AC_START_0**, **AC_START_1**, ... -- same count as pit boxes.  These are
   the grid positions on the start/finish straight.  AC_START_0 is pole
   position (closest to the start line).  Arrange them in a staggered 2-wide
   grid pattern along the driving direction.

Do NOT place timing points (AC_TIME_*) -- those will be generated separately.

Rules:
- Coordinates are **pixel positions** [x, y] on the image (origin = top-left).
- **orientation_z** is a 2D unit vector [dx, dy] pointing in the forward
  driving direction at that location.
- Return ONLY valid JSON, no markdown fences, no commentary.

Output schema:
{{
  "track_direction": "{direction}",
  "objects": [
    {{
      "name": "AC_HOTLAP_START_0",
      "position": [x, y],
      "orientation_z": [dx, dy],
      "type": "hotlap_start"
    }},
    ...
  ]
}}
"""


def _build_vlm_prompt(
    track_direction: str = "clockwise",
    pit_count: Optional[int] = None,
    start_count: Optional[int] = None,
) -> str:
    if pit_count is not None:
        pit_text = f"exactly {pit_count} pit boxes"
    else:
        pit_text = "at least 8 pit boxes"

    if start_count is not None:
        start_text = f"exactly {start_count} grid positions"
    else:
        start_text = "same count as pit boxes"

    system = _VLM_SYSTEM_PROMPT.format(direction=track_direction)
    system = system.replace("at least 8 pit boxes", pit_text)
    system = system.replace("same count as pit boxes", start_text)

    user_lines = [
        "Analyze this top-down track image and place the required Assetto Corsa game objects.",
        f"The track is driven in a {track_direction} direction.",
        f"Place ONLY: AC_HOTLAP_START_0, AC_PIT_* ({pit_text}), and AC_START_* ({start_text}).",
        "Return ONLY the JSON object described above.",
    ]
    return system + "\n\n" + "\n".join(user_lines)


def generate_vlm_objects(
    image_path: Union[str, Path],
    mask_path: Optional[Union[str, Path]] = None,
    track_direction: str = "clockwise",
    *,
    api_key: Optional[str] = None,
    model_name: str = "gemini-2.5-pro",
    temperature: float = 0.2,
    pit_count: Optional[int] = None,
    start_count: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Call Gemini to place hotlap_start, pits, and start grid (legacy single-call)."""
    from gemini_client import GeminiClient

    if api_key is None:
        api_key = "***REDACTED_GEMINI_KEY***"

    client = GeminiClient(api_key=api_key, model_name=model_name)

    img = _load_image(image_path)
    prompt = _build_vlm_prompt(track_direction, pit_count=pit_count, start_count=start_count)

    images: List[Image.Image] = [img]
    if mask_path is not None:
        mask_img = _load_image(mask_path)
        images.append(mask_img)
        prompt += (
            "\n\nA second image is the segmentation mask of this track. "
            "Use it to better identify track surfaces and boundaries."
        )

    if len(images) == 1:
        result = client.generate_json(prompt, image=images[0], temperature=temperature)
    else:
        result = client.generate_json(prompt, images=images, temperature=temperature)

    objects = result.get("objects", [])
    for obj in objects:
        obj["type"] = _infer_type(obj)

    logger.info("VLM returned %d objects", len(objects))
    return objects


# ---------------------------------------------------------------------------
# Programmatic timing point generation
# ---------------------------------------------------------------------------

def generate_programmatic_timing(
    road_mask_path: Union[str, Path],
    track_direction: str = "clockwise",
) -> Dict[str, Any]:
    """From road mask, extract centerline and generate timing points."""
    from road_centerline import process_road_mask

    result = process_road_mask(str(road_mask_path), track_direction)
    logger.info("Programmatic: %d centerline points, %d bends, %d timing objects",
                len(result["centerline"]), len(result["bends"]), len(result["timing_objects"]))
    return result


# ---------------------------------------------------------------------------
# Hybrid entry point (legacy, kept for backward compat)
# ---------------------------------------------------------------------------

def generate_game_objects(
    image_path: Union[str, Path],
    mask_path: Optional[Union[str, Path]] = None,
    road_mask_path: Optional[Union[str, Path]] = None,
    track_direction: str = "clockwise",
    *,
    api_key: Optional[str] = None,
    model_name: str = "gemini-2.5-pro",
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """Hybrid game object generation: VLM + programmatic (legacy)."""
    vlm_objects = generate_vlm_objects(
        image_path=image_path,
        mask_path=mask_path,
        track_direction=track_direction,
        api_key=api_key,
        model_name=model_name,
        temperature=temperature,
    )

    centerline_data = None
    timing_objects = []
    if road_mask_path and Path(road_mask_path).is_file():
        try:
            centerline_data = generate_programmatic_timing(road_mask_path, track_direction)
            timing_objects = centerline_data.get("timing_objects", [])
        except Exception as e:
            logger.warning("Programmatic timing generation failed: %s", e)

    all_objects = vlm_objects + timing_objects

    result = {
        "track_direction": track_direction,
        "objects": all_objects,
    }

    if centerline_data:
        result["_centerline_data"] = {
            "centerline": centerline_data["centerline"],
            "bends": centerline_data["bends"],
        }

    errors = validate_game_objects_json(result)
    if errors:
        logger.warning("Validation issues (non-fatal): %s", errors)

    return result


def generate_game_objects_prompt(track_direction: str = "clockwise") -> str:
    """Build the full prompt — kept for backward compatibility."""
    return _build_vlm_prompt(track_direction)
