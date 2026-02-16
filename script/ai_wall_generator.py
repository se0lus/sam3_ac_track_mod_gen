"""
AI-powered virtual wall generation for Assetto Corsa tracks.

Uses Gemini to analyze a 2D track map image and produce wall polygon data
(outer walls around the driveable area, inner walls blocking inaccessible areas).

Pure Python -- no Blender dependency.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image


# ---------------------------------------------------------------------------
# Image preparation
# ---------------------------------------------------------------------------

def prepare_image_for_llm(image_path: Union[str, Path], max_size: int = 2048) -> Image.Image:
    """Scale a track map image so its longest side is at most *max_size* pixels.

    Returns a PIL Image (RGB) suitable for sending to a vision LLM.
    """
    img = Image.open(str(image_path)).convert("RGB")
    w, h = img.size
    if max(w, h) <= max_size:
        return img
    scale = max_size / max(w, h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return img.resize((new_w, new_h), Image.LANCZOS)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_WALL_SYSTEM_PROMPT = """\
You are an expert racing-track analyst.  You will receive a top-down 2D image
of a racing track.  Your job is to output a JSON object describing virtual
walls that prevent cars from driving off the track.

Rules:
1. The **outer wall** is a single closed polygon that surrounds the entire
   driveable area (road + kerbs + runoff).  It should follow the outer
   boundary of the track map tightly.
2. **Inner walls** are closed polygons that block large inaccessible interior
   regions (e.g. the infield of an oval, or buildings/ponds in the middle).
   Only add inner walls where a significant non-driveable region exists.
3. Coordinates are **pixel positions** on the image (origin = top-left,
   x = right, y = down).
4. Return ONLY valid JSON, no markdown fences, no commentary.

Output schema:
{
  "walls": [
    {
      "type": "outer",          // exactly one outer wall
      "points": [[x1,y1], [x2,y2], ...],
      "closed": true
    },
    {
      "type": "inner",          // zero or more inner walls
      "points": [[x1,y1], [x2,y2], ...],
      "closed": true
    }
  ]
}
"""


def generate_wall_prompt(track_description: Optional[str] = None) -> str:
    """Build the user-side prompt sent alongside the image."""
    parts = [
        "Analyze this top-down track image and generate virtual wall polygons.",
        "Provide one outer wall polygon that tightly surrounds all driveable surface,",
        "and inner wall polygons for any large inaccessible interior areas.",
        "Output coordinates as pixel positions [x, y] on this image.",
        "Use enough points to follow curves accurately (at least 20 points for the outer wall).",
    ]
    if track_description:
        parts.append(f"Additional context: {track_description}")
    parts.append("Return ONLY the JSON object described in the system prompt.")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

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
        if wtype not in ("outer", "inner"):
            errors.append(f"walls[{i}].type must be 'outer' or 'inner', got {wtype!r}")
        if wtype == "outer":
            outer_count += 1
        pts = wall.get("points")
        if not isinstance(pts, list) or len(pts) < 3:
            errors.append(f"walls[{i}].points must be a list with >= 3 points")
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
    api_key: Optional[str] = None,
    model_name: str = "gemini-2.0-flash",
    track_description: Optional[str] = None,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """Call Gemini to generate wall polygons for the given track image.

    Args:
        image_path: Path to the 2D track map image.
        mask_image_path: Optional mask image (if provided, both images are
            sent to the model for better context).
        api_key: Gemini API key.  Falls back to a default if not given.
        model_name: Gemini model to use.
        track_description: Optional extra textual description of the track.
        temperature: LLM temperature.

    Returns:
        Validated wall JSON dict.
    """
    from gemini_client import GeminiClient

    if api_key is None:
        api_key = "***REDACTED_GEMINI_KEY***"

    client = GeminiClient(api_key=api_key, model_name=model_name)

    img = prepare_image_for_llm(image_path)
    user_prompt = _WALL_SYSTEM_PROMPT + "\n\n" + generate_wall_prompt(track_description)

    images: List[Image.Image] = [img]
    if mask_image_path is not None:
        mask_img = prepare_image_for_llm(mask_image_path)
        images.append(mask_img)
        user_prompt += (
            "\n\nA second image is the segmentation mask of this track. "
            "The coloured regions represent driveable surfaces. "
            "Use it to better identify the track boundary."
        )

    if len(images) == 1:
        result = client.generate_json(user_prompt, image=images[0], temperature=temperature)
    else:
        result = client.generate_json(user_prompt, images=images, temperature=temperature)

    errors = validate_walls_json(result)
    if errors:
        raise ValueError(f"Gemini returned invalid wall JSON: {errors}\nRaw: {json.dumps(result, indent=2)}")

    return result
