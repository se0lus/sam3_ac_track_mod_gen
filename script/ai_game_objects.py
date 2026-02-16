"""
AI-powered game object generation for Assetto Corsa tracks.

Uses Gemini to place invisible game objects (timing lines, pit boxes,
start positions, hotlap start) on a 2D track map image.

Pure Python -- no Blender dependency.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PIL import Image

from ai_wall_generator import prepare_image_for_llm


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_GAME_OBJECTS_SYSTEM_PROMPT = """\
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
4. **AC_TIME_0_L** and **AC_TIME_0_R** -- a left/right pair that forms the
   start/finish timing line (sector 0).  Place them on opposite edges of the
   track at the start/finish line.
   If the track has multiple sectors, add AC_TIME_1_L / AC_TIME_1_R, etc.

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


def generate_game_objects_prompt(track_direction: str = "clockwise") -> str:
    """Build the full prompt (system + user) for game object placement."""
    system = _GAME_OBJECTS_SYSTEM_PROMPT.format(direction=track_direction)
    user_lines = [
        "Analyze this top-down track image and place all required Assetto Corsa game objects.",
        f"The track is driven in a {track_direction} direction.",
        "Return ONLY the JSON object described above.",
    ]
    return system + "\n\n" + "\n".join(user_lines)


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

_REQUIRED_TYPES = {"hotlap_start"}
_KNOWN_TYPES = {"hotlap_start", "pit", "start", "timing_left", "timing_right"}


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
# Main generation entry point
# ---------------------------------------------------------------------------

def generate_game_objects(
    image_path: Union[str, Path],
    mask_path: Optional[Union[str, Path]] = None,
    track_direction: str = "clockwise",
    *,
    api_key: Optional[str] = None,
    model_name: str = "gemini-2.0-flash",
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """Call Gemini to generate game object placements for the given track image.

    Args:
        image_path: Path to the 2D track map image.
        mask_path: Optional segmentation mask image.
        track_direction: 'clockwise' or 'counterclockwise'.
        api_key: Gemini API key.
        model_name: Gemini model to use.
        temperature: LLM temperature.

    Returns:
        Validated game objects JSON dict.
    """
    from gemini_client import GeminiClient

    if api_key is None:
        api_key = "***REDACTED_GEMINI_KEY***"

    client = GeminiClient(api_key=api_key, model_name=model_name)

    img = prepare_image_for_llm(image_path)
    prompt = generate_game_objects_prompt(track_direction)

    images: List[Image.Image] = [img]
    if mask_path is not None:
        mask_img = prepare_image_for_llm(mask_path)
        images.append(mask_img)
        prompt += (
            "\n\nA second image is the segmentation mask of this track. "
            "Use it to better identify track surfaces and boundaries."
        )

    if len(images) == 1:
        result = client.generate_json(prompt, image=images[0], temperature=temperature)
    else:
        result = client.generate_json(prompt, images=images, temperature=temperature)

    errors = validate_game_objects_json(result)
    if errors:
        raise ValueError(f"Gemini returned invalid game objects JSON: {errors}\nRaw: {json.dumps(result, indent=2)}")

    return result
