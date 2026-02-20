"""Stage 11: Track packaging -- assemble final Assetto Corsa track mod folder.

Reads Stage 10 KN5 outputs + layout/game-object metadata and produces a
ready-to-use AC track folder with the canonical structure (all lowercase):

    {track_folder}/                       (lowercase track name)
        models_{layout}.ini               (per layout, e.g. models_layoutcw.ini)
        {shared}.kn5                      (terrain, collision, env -- shared)
        go_{layout}.kn5                   (game objects -- per layout)
        {layout}/map.png                  (TrackMapGenerator output)
        {layout}/data/map.ini             (map display parameters)
        {layout}/data/cameras.ini         (TV cameras)
        ui/{layout}/ui_track.json         (metadata)
        ui/{layout}/preview.png           (preview image)
        ui/{layout}/outline.png           (track outline)

Layout folder names are the layout name lowercased (e.g. LayoutCW -> layoutcw).
"""
from __future__ import annotations

import json
import logging
import math
import os
import shutil
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("sam3_pipeline.s11")

_script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from pipeline_config import PipelineConfig


# ---------------------------------------------------------------------------
# Gemini LLM helpers
# ---------------------------------------------------------------------------
def _generate_description_llm(
    config: PipelineConfig,
    track_name: str,
    country: str,
    city: str,
    length_m: float,
    direction: str,
    layout_display: str,
) -> Optional[str]:
    """Call Gemini to generate a track description from metadata.

    Returns the description string, or None on failure.
    """
    if not config.gemini_api_key:
        return None

    length_info = f"approximately {int(round(length_m))} metres long" if length_m > 0 else "length unknown"
    location = ", ".join(filter(None, [city, country])) or "unknown location"

    prompt = (
        f"Write a short, engaging description (2-3 sentences, in English) for a racing "
        f"track mod in Assetto Corsa.\n\n"
        f"Track name: {track_name}\n"
        f"Layout: {layout_display}\n"
        f"Location: {location}\n"
        f"Direction: {direction}\n"
        f"Length: {length_info}\n\n"
        f"The description should sound professional, mention the location and "
        f"key characteristics. Do NOT include the track name at the start. "
        f"Return ONLY the description text, no quotes or extra formatting."
    )

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=config.gemini_api_key)
        response = client.models.generate_content(
            model=config.gemini_model,
            contents=[prompt],
            config=types.GenerateContentConfig(temperature=0.7),
        )
        text = response.text.strip().strip('"').strip()
        if text and len(text) > 20:
            logger.info("LLM generated description (%d chars)", len(text))
            return text
    except Exception as e:
        logger.warning("LLM description generation failed: %s", e)

    return None


def _find_modelscale_image(config: PipelineConfig) -> Optional[str]:
    """Find the *_modelscale.png aerial image from Stage 2 output."""
    for base in [config.mask_full_map_result, config.mask_full_map_dir]:
        if not os.path.isdir(base):
            continue
        for f in os.listdir(base):
            if f.endswith("_modelscale.png"):
                return os.path.join(base, f)
    return None


def _find_visualization_image(config: PipelineConfig) -> Optional[str]:
    """Find the results_visualization.png from Stage 2 (segmentation overlay)."""
    for base in [config.mask_full_map_result, config.mask_full_map_dir]:
        if not os.path.isdir(base):
            continue
        p = os.path.join(base, "results_visualization.png")
        if os.path.isfile(p):
            return p
    return None


def _generate_preview_llm(
    config: PipelineConfig,
    output_path: str,
    track_name: str,
    country: str,
    city: str,
    length_m: float = 0.0,
    direction: str = "",
    tags: Optional[List[str]] = None,
    pitboxes: int = 0,
    layout_display: str = "",
) -> bool:
    """Call Gemini image model (img2img) to generate a preview from the aerial photo.

    Uses the Stage 2 modelscale aerial image as reference input, optionally
    accompanied by the segmentation visualization, combined with a rich text
    prompt, to produce a stylised track cover image.
    Returns True on success.
    """
    if not config.gemini_api_key:
        return False

    aerial_path = _find_modelscale_image(config)
    if not aerial_path:
        logger.info("No modelscale image found, skipping LLM preview generation")
        return False

    # --- Build rich text prompt with all available metadata ---
    location = ", ".join(filter(None, [city, country])) or ""
    location_desc = f" in {location}" if location else ""

    # Metadata lines
    meta_lines: List[str] = []
    if layout_display:
        meta_lines.append(f"Layout: {layout_display}")
    if direction:
        meta_lines.append(f"Driving direction: {direction}")
    if length_m > 0:
        meta_lines.append(f"Track length: approximately {int(round(length_m))} metres")
    if pitboxes > 0:
        meta_lines.append(f"Pit boxes: {pitboxes}")
    if tags:
        meta_lines.append(f"Track type: {', '.join(tags)}")

    meta_block = ""
    if meta_lines:
        meta_block = (
            "\n\nAdditional track information:\n"
            + "\n".join(f"- {line}" for line in meta_lines)
            + "\n"
        )

    prompt = (
        f"This is an aerial / satellite top-down photo of a racing circuit "
        f"called \"{track_name}\"{location_desc}. "
        f"{meta_block}"
        f"Based on this bird's-eye view, generate a beautiful, photorealistic "
        f"cover image for this track. "
        f"Transform the perspective: imagine standing at the side of the track "
        f"or from an elevated grandstand viewpoint, looking across the circuit. "
        f"Keep the real landscape features (roads, buildings, vegetation) visible "
        f"in the photo but make it look like a professional motorsport promotional "
        f"image with vivid colours, good lighting, and a dramatic sky. "
        f"Output a single landscape-oriented image."
    )

    try:
        import io
        from PIL import Image
        from google import genai
        from google.genai import types

        # Load aerial image
        aerial_img = Image.open(aerial_path).convert("RGB")
        logger.info("LLM preview: using aerial image %s (%dx%d)",
                     os.path.basename(aerial_path), *aerial_img.size)

        # Build contents list: prompt + aerial photo + optional segmentation viz
        contents: list = [prompt, aerial_img]

        viz_path = _find_visualization_image(config)
        if viz_path:
            viz_img = Image.open(viz_path).convert("RGB")
            contents.append(
                "This second image shows the same area with colour-coded "
                "segmentation: road surfaces, grass, trees, and buildings are "
                "highlighted. Use it to understand the track layout."
            )
            contents.append(viz_img)
            logger.info("LLM preview: also using segmentation viz %s (%dx%d)",
                         os.path.basename(viz_path), *viz_img.size)

        client = genai.Client(api_key=config.gemini_api_key)
        response = client.models.generate_content(
            model=config.inpaint_model,  # gemini-2.5-flash-image
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
            ),
        )
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                img = Image.open(io.BytesIO(part.inline_data.data)).convert("RGB")
                img = img.resize((355, 266), Image.LANCZOS)
                img.save(output_path)
                logger.info("LLM generated preview image -> %s", output_path)
                return True
        logger.warning("LLM preview: no image in response")
    except Exception as e:
        logger.warning("LLM preview image generation failed: %s", e)

    return False


# ---------------------------------------------------------------------------
# Preview image acquisition (multi-strategy)
# ---------------------------------------------------------------------------
def _acquire_preview_image(
    config: PipelineConfig,
    output_path: str,
    track_name: str,
    length_m: float = 0.0,
    direction: str = "",
    tags: Optional[List[str]] = None,
    pitboxes: int = 0,
    layout_display: str = "",
) -> bool:
    """Try to find/generate a preview image for the track.

    Strategy order:
    1. User-provided preview in output dir
    2. LLM image generation (Gemini)
    3. Crop from aerial GeoTIFF
    4. Gradient placeholder (last resort)
    """
    # Strategy 1: user-provided image in output
    for candidate in [
        os.path.join(config.output_dir, "preview.png"),
        os.path.join(config.output_dir, "preview.jpg"),
    ]:
        if os.path.isfile(candidate):
            shutil.copy2(candidate, output_path)
            logger.info("Using user-provided preview image: %s", candidate)
            return True

    # Strategy 2: LLM image generation (if enabled)
    if config.s11_llm_preview and _generate_preview_llm(
        config, output_path, track_name,
        config.s11_track_country, config.s11_track_city,
        length_m=length_m,
        direction=direction,
        tags=tags,
        pitboxes=pitboxes,
        layout_display=layout_display,
    ):
        return True

    # Strategy 3: crop from the aerial/satellite image (GeoTIFF rendered)
    aerial_candidates = [
        os.path.join(config.mask_full_map_dir, "geotiff_preview.png"),
        os.path.join(config.mask_full_map_dir, "merged_mask.png"),
    ]
    for img_path in aerial_candidates:
        if os.path.isfile(img_path):
            try:
                from PIL import Image
                img = Image.open(img_path).convert("RGB")
                w, h = img.size
                target_ratio = 4 / 3
                current_ratio = w / h
                if current_ratio > target_ratio:
                    new_w = int(h * target_ratio)
                    left = (w - new_w) // 2
                    img = img.crop((left, 0, left + new_w, h))
                else:
                    new_h = int(w / target_ratio)
                    top = (h - new_h) // 2
                    img = img.crop((0, top, w, top + new_h))
                img = img.resize((355, 266), Image.LANCZOS)
                img.save(output_path)
                logger.info("Generated preview from aerial image: %s", img_path)
                return True
            except Exception as e:
                logger.warning("Failed to process aerial image for preview: %s", e)

    # Strategy 4: gradient placeholder
    try:
        from PIL import Image, ImageDraw, ImageFont
        img = Image.new("RGB", (355, 266), (26, 26, 46))
        draw = ImageDraw.Draw(img)
        for y in range(266):
            alpha = int(40 + 30 * y / 266)
            draw.line([(0, y), (354, y)], fill=(alpha, alpha, alpha + 20))
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except OSError:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), track_name, font=font)
        tw = bbox[2] - bbox[0]
        draw.text(((355 - tw) // 2, 120), track_name, fill=(229, 231, 235), font=font)
        img.save(output_path)
        logger.info("Generated placeholder preview image")
        return True
    except Exception as e:
        logger.warning("Failed to generate placeholder preview: %s", e)

    return False


# ---------------------------------------------------------------------------
# KN5 classification
# ---------------------------------------------------------------------------
def _classify_kn5_files(export_dir: str, layouts: List[Dict]) -> Dict[str, List[str]]:
    """Classify Stage 10 KN5 files into shared vs per-layout categories.

    Returns dict with keys:
        "shared"   : list of shared KN5 basenames (terrain, collision, env)
        layout_name: list of layout-specific KN5 basenames (game objects)
    """
    layout_names = {lay["name"] for lay in layouts}
    kn5_files = sorted(
        f for f in os.listdir(export_dir) if f.endswith(".kn5")
    )

    result: Dict[str, List[str]] = {"shared": []}
    for ln in layout_names:
        result[ln] = []

    for kn5 in kn5_files:
        matched_layout = None
        for ln in layout_names:
            if f"_go_{ln}" in kn5:
                matched_layout = ln
                break
        if matched_layout:
            result[matched_layout].append(kn5)
        else:
            result["shared"].append(kn5)

    return result


def _layout_short_name(layout: Dict) -> str:
    """Derive a folder name from the layout name, lowercased.

    E.g. LayoutCW -> layoutcw, LayoutCCW -> layoutccw.
    """
    name = layout.get("name", "default")
    return name.lower()


# ---------------------------------------------------------------------------
# models_*.ini generation
# ---------------------------------------------------------------------------
def _generate_models_ini(
    shared_kn5_names: List[str],
    layout_kn5_names: List[str],
) -> str:
    """Generate a models_{layout}.ini file content."""
    all_models = list(shared_kn5_names) + list(layout_kn5_names)
    lines = []
    for i, name in enumerate(all_models):
        lines.append(f"[MODEL_{i}]")
        lines.append(f"FILE={name}")
        lines.append("POSITION=0,0,0")
        lines.append("ROTATION=0,0,0")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# map.ini generation (from geo_metadata and centerline)
# ---------------------------------------------------------------------------
def _compute_track_bounds_from_centerline(
    centerline_path: str,
) -> Optional[Tuple[float, float, float, float]]:
    """Compute bounding box (min_x, min_z, max_x, max_z) from centerline points.

    Centerline coords are in Blender space (x, z) where z maps to AC Z axis.
    """
    if not os.path.isfile(centerline_path):
        return None
    with open(centerline_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    points = data.get("centerline", [])
    if not points:
        return None
    xs = [p[0] for p in points]
    zs = [p[1] for p in points]
    return (min(xs), min(zs), max(xs), max(zs))


def _generate_map_ini(
    centerline_path: str,
    margin: int = 20,
) -> str:
    """Generate map.ini content with map display parameters."""
    bounds = _compute_track_bounds_from_centerline(centerline_path)
    if bounds is None:
        # Fallback defaults
        return (
            "[PARAMETERS]\n"
            "WIDTH=200\n"
            "HEIGHT=200\n"
            "X_OFFSET=100\n"
            "Z_OFFSET=100\n"
            "MARGIN=20\n"
            "SCALE_FACTOR=1\n"
            "DRAWING_SIZE=10\n\n"
        )

    min_x, min_z, max_x, max_z = bounds
    width = max_x - min_x + margin * 2
    height = max_z - min_z + margin * 2
    # Offsets: distance from track origin to the left/top edge of the map
    x_offset = -min_x + margin
    z_offset = -min_z + margin

    return (
        "[PARAMETERS]\n"
        f"WIDTH={width:.4f}\n"
        f"HEIGHT={height:.4f}\n"
        f"X_OFFSET={x_offset:.4f}\n"
        f"Z_OFFSET={z_offset:.4f}\n"
        f"MARGIN={margin}\n"
        "SCALE_FACTOR=1\n"
        "DRAWING_SIZE=10\n\n"
    )


# ---------------------------------------------------------------------------
# cameras.ini generation (basic default cameras from centerline)
# ---------------------------------------------------------------------------
def _generate_cameras_ini(centerline_path: str, num_cameras: int = 5) -> str:
    """Generate a basic cameras.ini with evenly spaced cameras along the track."""
    bounds = _compute_track_bounds_from_centerline(centerline_path)
    if bounds is None:
        # Minimal fallback
        return "[HEADER]\nVERSION=3\nCAMERA_COUNT=0\nSET_NAME=TV1\n\n"

    with open(centerline_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    points = data.get("centerline", [])
    if len(points) < num_cameras:
        num_cameras = max(1, len(points))

    # Pick evenly spaced points along centerline
    step = len(points) / num_cameras
    lines = [
        "[HEADER]",
        "VERSION=3",
        f"CAMERA_COUNT={num_cameras}",
        "SET_NAME=TV1",
        "",
    ]

    total_points = len(points)
    for i in range(num_cameras):
        idx = int(i * step) % total_points
        # Camera position: slightly above and to the side
        px, pz = points[idx]
        # Next point for forward direction
        next_idx = (idx + max(1, total_points // 50)) % total_points
        nx, nz = points[next_idx]
        dx, dz = nx - px, nz - pz
        length = math.sqrt(dx * dx + dz * dz)
        if length > 0:
            dx /= length
            dz /= length
        else:
            dx, dz = 1.0, 0.0

        # Camera height offset (elevated view)
        height = 8.0
        # Side offset (perpendicular to track direction)
        side_offset = 15.0
        cam_x = px + (-dz) * side_offset
        cam_z = pz + dx * side_offset

        # Distribute camera in_point / out_point along the track
        in_point = round(i / num_cameras, 2)
        out_point = round((i + 1) / num_cameras, 2) if i < num_cameras - 1 else 0.1

        lines.extend([
            f"[CAMERA_{i}]",
            f"NAME={i + 1}",
            f"POSITION={cam_x:.3f} ,{-height:.3f} ,{cam_z:.3f}",
            f"FORWARD={dx:.6f} ,-0.15 ,{dz:.6f}",
            f"UP=0 ,1 ,0",
            "MIN_FOV=10",
            "MAX_FOV=60",
            f"IN_POINT={in_point}",
            f"OUT_POINT={out_point}",
            "SHADOW_SPLIT0=1.8",
            "SHADOW_SPLIT1=20",
            "SHADOW_SPLIT2=180",
            "NEAR_PLANE=0.1",
            "FAR_PLANE=5000",
            "MIN_EXPOSURE=0",
            "MAX_EXPOSURE=10000",
            "DOF_FACTOR=10",
            "DOF_RANGE=10000",
            "DOF_FOCUS=0",
            "DOF_MANUAL=0",
            "SPLINE=",
            "SPLINE_ROTATION=0",
            "FOV_GAMMA=0",
            "SPLINE_ANIMATION_LENGTH=15",
            "IS_FIXED=0",
            "",
        ])

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# ui_track.json generation
# ---------------------------------------------------------------------------
def _compute_pixel_size_m(config: PipelineConfig) -> float:
    """Compute metres-per-pixel from Stage 2 geo metadata.

    Reads result_masks.json from 02_result (or 02_mask_full_map fallback).
    Returns 0.0 if unavailable.
    """
    for base in [config.mask_full_map_result, config.mask_full_map_dir]:
        p = os.path.join(base, "result_masks.json") if os.path.isdir(base) else ""
        if p and os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                meta = data.get("meta", {})
                ms = meta.get("model_scale_size", {})
                w = ms.get("width", 1)
                h = ms.get("height", 1)
                geo = meta.get("geo", {}).get("bounds", {})
                left = geo.get("left", 0)
                right = geo.get("right", 0)
                bottom = geo.get("bottom", 0)
                top = geo.get("top", 0)
                lat_mid = (top + bottom) / 2.0
                cos_lat = math.cos(math.radians(lat_mid))
                width_m = abs(right - left) * 111_320.0 * cos_lat
                height_m = abs(top - bottom) * 111_320.0
                px_w = width_m / max(w, 1)
                px_h = height_m / max(h, 1)
                return (px_w + px_h) / 2.0
            except Exception as e:
                logger.warning("Failed to read pixel_size_m: %s", e)
    return 0.0


def _compute_track_length(centerline_path: str, pixel_size_m: float) -> float:
    """Compute track length in metres from centerline points.

    Centerline coordinates are in pixel space (modelscale), so we
    multiply the geometric length by *pixel_size_m* to get metres.
    """
    if not os.path.isfile(centerline_path):
        return 0.0
    with open(centerline_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    points = data.get("centerline", [])
    if len(points) < 2:
        return 0.0
    total_px = 0.0
    for i in range(1, len(points)):
        dx = points[i][0] - points[i - 1][0]
        dz = points[i][1] - points[i - 1][1]
        total_px += math.sqrt(dx * dx + dz * dz)
    if pixel_size_m > 0:
        return total_px * pixel_size_m
    # Fallback: return pixel length (better than 0)
    logger.warning("pixel_size_m unavailable, track length in pixels: %.0f", total_px)
    return total_px


def _count_pitboxes(game_objects_path: str) -> int:
    """Count pit objects from game_objects.json."""
    if not os.path.isfile(game_objects_path):
        return 0
    with open(game_objects_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return sum(1 for obj in data.get("objects", []) if obj.get("type") == "pit")


def _parse_layout_display_names(raw: str) -> Dict[str, str]:
    """Parse 'layoutcw:Clockwise;layoutccw:Counter-clockwise' into a dict."""
    result = {}
    if not raw:
        return result
    for pair in raw.split(";"):
        pair = pair.strip()
        if ":" in pair:
            key, val = pair.split(":", 1)
            result[key.strip().lower()] = val.strip()
    return result


def _generate_ui_track_json(
    config: PipelineConfig,
    layout: Dict,
    centerline_path: str,
    game_objects_path: str,
    pixel_size_m: float,
) -> Dict[str, Any]:
    """Generate ui_track.json content for a given layout."""
    short = _layout_short_name(layout)
    direction = layout.get("track_direction", "clockwise")
    run_label = "clockwise" if direction == "clockwise" else "anti-clockwise"

    # Compute track length (centerline is in pixel space → multiply by pixel_size_m)
    length_m = _compute_track_length(centerline_path, pixel_size_m)
    length_str = str(int(round(length_m))) if length_m > 0 else ""

    # Count pitboxes
    pitboxes = _count_pitboxes(game_objects_path)
    if pitboxes == 0:
        pitboxes = config.s11_pitboxes

    # Track name
    track_name = config.s11_track_name
    if not track_name:
        if config.geotiff_path:
            track_name = os.path.splitext(os.path.basename(config.geotiff_path))[0]
        else:
            track_name = "Generated Track"

    # Layout display name: user-configured (full name) > auto "{track_name} {short}"
    display_names = _parse_layout_display_names(config.s11_layout_display_names)
    if short in display_names:
        # User provided the full display name for this layout
        name = display_names[short]
        layout_display = name
    else:
        layout_display = short
        name = f"{track_name} {short}".strip()

    year = config.s11_track_year
    if year == 0:
        year = datetime.now().year

    # Description: user-provided > LLM generated (if enabled) > template fallback
    description = config.track_description
    if not description and config.s11_llm_description:
        description = _generate_description_llm(
            config, track_name,
            config.s11_track_country, config.s11_track_city,
            length_m, run_label, layout_display,
        )
    if not description:
        length_info = f"Track length: ~{int(round(length_m))}m. " if length_m > 0 else ""
        description = (
            f"Auto-generated {run_label} layout of {track_name}. {length_info}"
        ).strip()

    result: Dict[str, Any] = {
        "name": name,
        "description": description,
        "tags": list(config.s11_track_tags),
        "country": config.s11_track_country,
        "city": config.s11_track_city,
        "length": length_str,
        "width": "10",
        "pitboxes": str(pitboxes),
        "run": run_label,
        "year": year,
        "author": config.s11_track_author,
    }
    if config.s11_track_url:
        result["url"] = config.s11_track_url

    return result


# ---------------------------------------------------------------------------
# Track outline image generation (simple PNG from centerline)
# ---------------------------------------------------------------------------
def _generate_outline_png(
    centerline_path: str, output_path: str, size: int = 512
) -> bool:
    """Generate a simple track outline PNG from centerline points."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        logger.warning("Pillow not installed, skipping outline generation")
        return False

    if not os.path.isfile(centerline_path):
        return False

    with open(centerline_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    points = data.get("centerline", [])
    if len(points) < 3:
        return False

    # Compute bounds
    xs = [p[0] for p in points]
    zs = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_z, max_z = min(zs), max(zs)
    range_x = max_x - min_x or 1
    range_z = max_z - min_z or 1

    # Scale to fit image with margin
    margin = 30
    draw_size = size - margin * 2
    scale = min(draw_size / range_x, draw_size / range_z)

    # Center in image
    cx = size / 2
    cz = size / 2
    center_x = (min_x + max_x) / 2
    center_z = (min_z + max_z) / 2

    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Map points to image coordinates
    img_points = []
    for px, pz in points:
        ix = cx + (px - center_x) * scale
        iz = cz + (pz - center_z) * scale
        img_points.append((ix, iz))

    # Close the loop
    img_points.append(img_points[0])

    # Draw track outline
    draw.line(img_points, fill=(255, 255, 255, 255), width=3)

    # Mark start/finish with a dot
    if img_points:
        sx, sz = img_points[0]
        draw.ellipse(
            [sx - 5, sz - 5, sx + 5, sz + 5],
            fill=(255, 80, 80, 255),
        )

    img.save(output_path)
    return True


# ---------------------------------------------------------------------------
# TrackMapGenerator integration
# ---------------------------------------------------------------------------
_TRACK_MAP_GEN_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "ac_toolbox", "TrackMapGenerator",
)


def _run_track_map_generator(kn5_path: str, dest_dir: str) -> Optional[Tuple[str, str]]:
    """Run TrackMapGenerator.exe on a KN5 file.

    The tool does NOT exit automatically after generation, so we:
    1. Clean previous output files before starting
    2. Copy the KN5 into the tool directory
    3. Launch the process (non-blocking)
    4. Poll until map.png + data/map.ini both appear, or 5s timeout
    5. Kill the process
    6. Move generated files to *dest_dir*

    Returns (map_png_path, map_ini_path) inside *dest_dir*, or None.
    """
    import subprocess
    import time

    exe = os.path.join(_TRACK_MAP_GEN_DIR, "TrackMapGenerator.exe")
    if not os.path.isfile(exe):
        logger.warning("TrackMapGenerator.exe not found at %s", exe)
        return None
    if not os.path.isfile(kn5_path):
        logger.warning("KN5 file not found: %s", kn5_path)
        return None

    kn5_name = os.path.basename(kn5_path)
    work_kn5 = os.path.join(_TRACK_MAP_GEN_DIR, kn5_name)
    out_map = os.path.join(_TRACK_MAP_GEN_DIR, "map.png")
    out_ini_dir = os.path.join(_TRACK_MAP_GEN_DIR, "data")
    out_ini = os.path.join(out_ini_dir, "map.ini")

    proc = None
    try:
        # 1. Clean previous output + stale KN5
        for f in [out_map, out_ini, work_kn5]:
            if os.path.isfile(f):
                os.remove(f)

        # 2. Copy KN5 to TrackMapGenerator directory
        shutil.copy2(kn5_path, work_kn5)
        logger.info("TrackMapGenerator: copied %s", kn5_name)

        # 3. Launch (non-blocking) — tool stays alive after generation
        proc = subprocess.Popen(
            [exe, f".\\{kn5_name}"],
            cwd=_TRACK_MAP_GEN_DIR,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # 4. Poll for output files (check every 0.25s, max 5s)
        deadline = time.monotonic() + 5.0
        generated = False
        while time.monotonic() < deadline:
            if os.path.isfile(out_map) and os.path.isfile(out_ini):
                # Brief extra wait so the tool finishes writing
                time.sleep(0.3)
                generated = True
                break
            time.sleep(0.25)

        # 5. Kill the process (it won't exit on its own)
        try:
            proc.kill()
            proc.wait(timeout=3)
        except Exception:
            pass
        proc = None

        if not generated:
            logger.warning("TrackMapGenerator timed out — output not found")
            return None

        # 6. Move generated files to dest_dir
        os.makedirs(dest_dir, exist_ok=True)
        dest_data = os.path.join(dest_dir, "data")
        os.makedirs(dest_data, exist_ok=True)

        dst_map = os.path.join(dest_dir, "map.png")
        dst_ini = os.path.join(dest_data, "map.ini")
        shutil.move(out_map, dst_map)
        shutil.move(out_ini, dst_ini)
        logger.info("TrackMapGenerator -> %s/map.png + data/map.ini", dest_dir)
        return (dst_map, dst_ini)

    except Exception as e:
        logger.warning("TrackMapGenerator error: %s", e)
        return None
    finally:
        # Kill process if still alive
        if proc is not None:
            try:
                proc.kill()
                proc.wait(timeout=3)
            except Exception:
                pass
        # Clean up copied KN5
        if os.path.isfile(work_kn5):
            try:
                os.remove(work_kn5)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Main run function
# ---------------------------------------------------------------------------
def run(config: PipelineConfig) -> None:
    """Execute Stage 11: Track Packaging."""
    logger.info("=" * 60)
    logger.info("Stage 11: Track Packaging")
    logger.info("=" * 60)

    export_dir = config.export_dir  # 10_model_export
    if not os.path.isdir(export_dir):
        raise FileNotFoundError(
            f"Stage 10 output not found: {export_dir}. Run Stage 10 first."
        )

    # Load layout information
    layouts_json = config.track_layouts_json
    if os.path.isfile(layouts_json):
        with open(layouts_json, "r", encoding="utf-8") as f:
            layouts_data = json.load(f)
        layouts = layouts_data.get("layouts", [])
    else:
        # Single default layout
        layouts = [{
            "name": "Default",
            "track_direction": config.track_direction,
        }]
    if not layouts:
        raise ValueError("No layouts found. Configure layouts in Stage 2a.")

    logger.info("Layouts: %s", [l["name"] for l in layouts])

    # Compute pixel_size_m once (centerline coords are pixel-space)
    pixel_size_m = _compute_pixel_size_m(config)
    if pixel_size_m > 0:
        logger.info("pixel_size_m = %.4f", pixel_size_m)
    else:
        logger.warning("pixel_size_m unavailable — track length will be approximate")

    # Determine track name (folder name must be all lowercase)
    track_name = config.s11_track_name
    if not track_name:
        if config.geotiff_path:
            track_name = os.path.splitext(os.path.basename(config.geotiff_path))[0]
        else:
            track_name = "generated_track"
    track_folder = track_name.lower().replace(" ", "_")
    logger.info("Track name: %s (folder: %s)", track_name, track_folder)

    # Output directory (all lowercase)
    out_dir = os.path.join(config.packaging_dir, track_folder)
    os.makedirs(out_dir, exist_ok=True)
    logger.info("Output: %s", out_dir)

    # -----------------------------------------------------------------------
    # 1. Classify and copy KN5 files (keep original names from Stage 10)
    # -----------------------------------------------------------------------
    classified = _classify_kn5_files(export_dir, layouts)

    logger.info("KN5 files found: %d shared, %s per-layout",
                len(classified["shared"]),
                {k: len(v) for k, v in classified.items() if k != "shared"})

    all_kn5 = list(classified["shared"])
    for lay in layouts:
        all_kn5.extend(classified.get(lay["name"], []))

    for kn5 in all_kn5:
        src = os.path.join(export_dir, kn5)
        dst = os.path.join(out_dir, kn5)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            logger.info("  Copied %s", kn5)
        else:
            logger.warning("  KN5 not found: %s", src)

    # -----------------------------------------------------------------------
    # 2. Generate models_*.ini for each layout
    # -----------------------------------------------------------------------
    layout_short_map = {}
    for lay in layouts:
        short = _layout_short_name(lay)
        layout_short_map[lay["name"]] = short

    for lay in layouts:
        ln = lay["name"]
        short = layout_short_map[ln]
        ini_content = _generate_models_ini(
            classified["shared"], classified.get(ln, []),
        )
        ini_path = os.path.join(out_dir, f"models_{short}.ini")
        with open(ini_path, "w", encoding="utf-8") as f:
            f.write(ini_content)
        logger.info("Generated %s", os.path.basename(ini_path))

    # -----------------------------------------------------------------------
    # 3. Run TrackMapGenerator once (shared road KN5 → map.png + data/map.ini)
    #    The tool does NOT auto-exit, so we poll for output then kill it.
    #    Output is moved into a staging dir, then copied to each layout.
    # -----------------------------------------------------------------------
    track_core_kn5 = None
    for kn5 in classified["shared"]:
        if "track_core" in kn5:
            track_core_kn5 = os.path.join(out_dir, kn5)
            break
    if not track_core_kn5 and classified["shared"]:
        track_core_kn5 = os.path.join(out_dir, classified["shared"][0])

    tmg_staging = os.path.join(out_dir, "_tmg_staging")
    tmg_result = None
    if track_core_kn5:
        tmg_result = _run_track_map_generator(track_core_kn5, tmg_staging)

    # -----------------------------------------------------------------------
    # 4. Create per-layout directories
    # -----------------------------------------------------------------------
    game_objects_base = config.game_objects_result_dir
    if not os.path.isdir(game_objects_base):
        game_objects_base = config.stage_dir("ai_game_objects")

    for lay in layouts:
        ln = lay["name"]
        short = layout_short_map[ln]

        # Layout data directory
        data_dir = os.path.join(out_dir, short, "data")
        os.makedirs(data_dir, exist_ok=True)

        # AI directory (empty placeholder)
        ai_dir = os.path.join(out_dir, short, "ai")
        os.makedirs(ai_dir, exist_ok=True)

        # Centerline path for this layout
        centerline_path = os.path.join(game_objects_base, ln, "centerline.json")
        game_objects_path = os.path.join(game_objects_base, ln, "game_objects.json")

        # 4a. map.png and data/map.ini — prefer TrackMapGenerator output
        layout_map_path = os.path.join(out_dir, short, "map.png")
        layout_map_ini_path = os.path.join(data_dir, "map.ini")

        if tmg_result:
            shutil.copy2(tmg_result[0], layout_map_path)
            shutil.copy2(tmg_result[1], layout_map_ini_path)
            logger.info("Copied TrackMapGenerator output -> %s/map.png + data/map.ini", short)
        else:
            # Fallback: generate from centerline
            map_ini = _generate_map_ini(centerline_path)
            with open(layout_map_ini_path, "w", encoding="utf-8") as f:
                f.write(map_ini)
            _generate_outline_png(centerline_path, layout_map_path)
            logger.info("Generated fallback %s/map.png + data/map.ini from centerline", short)

        # 4b. Generate cameras.ini
        cameras_ini = _generate_cameras_ini(centerline_path)
        with open(os.path.join(data_dir, "cameras.ini"), "w", encoding="utf-8") as f:
            f.write(cameras_ini)
        logger.info("Generated %s/data/cameras.ini", short)

    # Clean up TrackMapGenerator staging dir
    if os.path.isdir(tmg_staging):
        shutil.rmtree(tmg_staging, ignore_errors=True)

    # -----------------------------------------------------------------------
    # 5. Generate UI metadata
    # -----------------------------------------------------------------------
    for lay in layouts:
        ln = lay["name"]
        short = layout_short_map[ln]

        ui_dir = os.path.join(out_dir, "ui", short)
        os.makedirs(ui_dir, exist_ok=True)

        centerline_path = os.path.join(game_objects_base, ln, "centerline.json")
        game_objects_path = os.path.join(game_objects_base, ln, "game_objects.json")

        # 5a. ui_track.json
        ui_data = _generate_ui_track_json(config, lay, centerline_path, game_objects_path, pixel_size_m)
        ui_json_path = os.path.join(ui_dir, "ui_track.json")
        with open(ui_json_path, "w", encoding="utf-8") as f:
            json.dump(ui_data, f, indent=2, ensure_ascii=False)
        logger.info("Generated ui/%s/ui_track.json", short)

        # 5b. outline.png — use TrackMapGenerator map.png if available
        outline_path = os.path.join(ui_dir, "outline.png")
        layout_map_path = os.path.join(out_dir, short, "map.png")
        if tmg_result and os.path.isfile(layout_map_path):
            shutil.copy2(layout_map_path, outline_path)
            logger.info("Copied map.png -> ui/%s/outline.png", short)
        else:
            _generate_outline_png(centerline_path, outline_path)
            logger.info("Generated ui/%s/outline.png from centerline", short)

        # 5c. Preview image
        preview_path = os.path.join(ui_dir, "preview.png")
        preview_src = os.path.join(game_objects_base, ln, "game_objects_preview.png")

        # Gather context for LLM preview generation
        lay_direction = lay.get("track_direction", config.track_direction)
        lay_run = "clockwise" if lay_direction == "clockwise" else "anti-clockwise"
        lay_length = _compute_track_length(centerline_path, pixel_size_m)
        lay_pitboxes = _count_pitboxes(game_objects_path)
        if lay_pitboxes == 0:
            lay_pitboxes = config.s11_pitboxes
        # Layout display name (same logic as ui_track.json)
        display_names = _parse_layout_display_names(config.s11_layout_display_names)
        lay_display = display_names.get(short, short)

        if os.path.isfile(preview_src):
            shutil.copy2(preview_src, preview_path)
            logger.info("Copied game_objects_preview -> ui/%s/preview.png", short)
        elif not _acquire_preview_image(
            config, preview_path, track_name,
            length_m=lay_length,
            direction=lay_run,
            tags=list(config.s11_track_tags),
            pitboxes=lay_pitboxes,
            layout_display=lay_display,
        ):
            if os.path.isfile(outline_path):
                shutil.copy2(outline_path, preview_path)
                logger.info("Using outline as preview for ui/%s", short)
            else:
                logger.warning("No preview image available for ui/%s", short)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 60)
    logger.info("Track packaging complete!")
    logger.info("Output: %s", out_dir)
    logger.info("")

    # List final structure
    total_size = 0
    file_count = 0
    for root, dirs, files in os.walk(out_dir):
        for f in files:
            fp = os.path.join(root, f)
            sz = os.path.getsize(fp)
            total_size += sz
            file_count += 1
            rel = os.path.relpath(fp, out_dir)
            logger.info("  %s (%.1f MB)" if sz > 1e6 else "  %s (%.1f KB)",
                        rel, sz / 1e6 if sz > 1e6 else sz / 1e3)

    logger.info("")
    logger.info("Total: %d files, %.1f MB", file_count, total_size / 1e6)
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Stage 11: Track Packaging")
    parser.add_argument("--output-dir", default="output", help="Pipeline output directory")
    parser.add_argument("--track-name", default="", help="Track folder name")
    parser.add_argument("--track-author", default="", help="Track author")
    parser.add_argument("--track-country", default="", help="Country")
    parser.add_argument("--track-city", default="", help="City")
    args = parser.parse_args()

    cfg = PipelineConfig(output_dir=args.output_dir)
    if args.track_name:
        cfg.s11_track_name = args.track_name
    if args.track_author:
        cfg.s11_track_author = args.track_author
    if args.track_country:
        cfg.s11_track_country = args.track_country
    if args.track_city:
        cfg.s11_track_city = args.track_city
    cfg.resolve()
    run(cfg)
