"""Center-hole detection and Gemini-based inpainting for aerial survey images.

Aerial survey images often have black holes in the interior where the drone
failed to capture data.  These are distinct from the normal black borders of
non-rectangular survey extents.  This module detects interior holes and uses
the Gemini image model to fill them with plausible terrain.
"""

from __future__ import annotations

import io
import logging
import time
from typing import Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("sam3_pipeline.inpainter")


def detect_center_holes(
    image: Image.Image,
    brightness_threshold: int = 20,
    min_hole_ratio: float = 0.001,
    dilate_px: int = 5,
) -> Optional[np.ndarray]:
    """Detect black holes in the interior of an aerial survey image.

    Uses a flood-fill-from-edges approach:
    1. Build a binary mask of all "black" pixels (below *brightness_threshold*).
    2. Flood-fill from every edge-touching black pixel to mark the border-
       connected black region.
    3. Whatever black pixels remain are interior ("center") holes.
    4. Clean with morphological close + small-component removal.
    5. Return ``None`` if the hole area is below *min_hole_ratio* of the image.

    Parameters
    ----------
    image : PIL.Image.Image
        Input image (RGB or RGBA).
    brightness_threshold : int
        Pixels with max(R,G,B) below this value are considered "black".
    min_hole_ratio : float
        Minimum fraction of total pixels that must be holes to trigger
        inpainting (default 0.1 %).
    dilate_px : int
        Dilate the detected mask by this many pixels to cover dark boundary
        pixels and prevent visible seam lines after compositing.

    Returns
    -------
    np.ndarray or None
        uint8 mask (255 = hole, 0 = keep), or ``None`` if no significant
        interior holes are found.
    """
    img_np = np.array(image.convert("RGB"))
    gray = np.max(img_np, axis=2)  # max-channel brightness

    black_mask = (gray < brightness_threshold).astype(np.uint8)

    # Flood fill from edges to mark border-connected black regions
    h, w = black_mask.shape
    edge_visited = np.zeros_like(black_mask)

    # Collect edge-touching black seed pixels
    # floodFill needs a mask 2 pixels larger than the image
    ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

    # Top and bottom edges
    for x in range(w):
        if black_mask[0, x] and not edge_visited[0, x]:
            cv2.floodFill(black_mask, ff_mask, (x, 0), 0)
            edge_visited[black_mask == 0] = 0  # already cleared by floodFill
        if black_mask[h - 1, x] and not edge_visited[h - 1, x]:
            cv2.floodFill(black_mask, ff_mask, (x, h - 1), 0)

    # Left and right edges
    for y in range(h):
        if black_mask[y, 0] and not edge_visited[y, 0]:
            cv2.floodFill(black_mask, ff_mask, (0, y), 0)
        if black_mask[y, w - 1] and not edge_visited[y, w - 1]:
            cv2.floodFill(black_mask, ff_mask, (w - 1, y), 0)

    # After flood-filling, remaining 1s in black_mask are interior holes
    center_holes = black_mask

    # Morphological close to fill small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    center_holes = cv2.morphologyEx(center_holes, cv2.MORPH_CLOSE, kernel)

    # Remove tiny connected components (< 100 px)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        center_holes, connectivity=8
    )
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < 100:
            center_holes[labels == i] = 0

    # Dilate to cover semi-dark boundary pixels (prevents seam lines)
    if dilate_px > 0:
        dilate_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1)
        )
        center_holes = cv2.dilate(center_holes, dilate_kernel, iterations=1)

    hole_ratio = np.sum(center_holes > 0) / center_holes.size
    if hole_ratio < min_hole_ratio:
        return None

    return (center_holes * 255).astype(np.uint8)


def inpaint_holes(
    image: Image.Image,
    hole_mask: np.ndarray,
    api_key: str,
    model_name: str = "gemini-2.5-flash-image",
    max_retries: int = 3,
) -> Image.Image:
    """Fill interior holes using the Gemini image generation model.

    Sends the image with a prompt asking Gemini to fill only the black
    patches.  Then composites the result: only pixels in *hole_mask* are
    taken from the Gemini output, keeping all other pixels untouched.

    Parameters
    ----------
    image : PIL.Image.Image
        Original image with holes.
    hole_mask : np.ndarray
        uint8 mask (255 = hole pixels to fill).
    api_key : str
        Google Gemini API key.
    model_name : str
        Gemini image model to use.
    max_retries : int
        Number of API call attempts before raising.

    Returns
    -------
    PIL.Image.Image
        Image with holes filled.
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    prompt = (
        "This aerial survey photo has black patches where data is missing. "
        "Fill ONLY the black holes with terrain (grass, paths, etc.) matching "
        "the surrounding areas. Keep all non-black areas exactly unchanged."
    )

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[prompt, image],
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                ),
            )

            # Extract generated image from response
            gemini_img: Image.Image | None = None
            for part in response.candidates[0].content.parts:
                if part.inline_data is not None:
                    gemini_img = Image.open(
                        io.BytesIO(part.inline_data.data)
                    ).convert("RGB")
                    break

            if gemini_img is None:
                raise ValueError("Gemini returned no image in response")

            # Composite: only replace hole pixels
            original_np = np.array(image.convert("RGB"))
            gemini_np = np.array(gemini_img.resize(image.size, Image.LANCZOS))

            mask_bool = hole_mask > 0
            result = original_np.copy()
            result[mask_bool] = gemini_np[mask_bool]

            logger.info("Inpainting succeeded on attempt %d", attempt)
            return Image.fromarray(result)

        except Exception as e:
            last_error = e
            logger.warning(
                "Inpainting attempt %d/%d failed: %s", attempt, max_retries, e
            )
            if attempt < max_retries:
                time.sleep(2.0 * attempt)

    raise RuntimeError(
        f"Gemini inpainting failed after {max_retries} attempts: {last_error}"
    )
