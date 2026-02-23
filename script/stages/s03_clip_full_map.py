"""Stage 3: Clip the full map into tiles for per-tile processing."""
from __future__ import annotations

import argparse
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger("sam3_pipeline.s03")

_script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from pipeline_config import PipelineConfig
from progress import ProgressTracker


def run(config: PipelineConfig) -> None:
    """Execute Stage 3: Clip the full map into tiles.

    Reads the merged mask from stage 2's output, writes clips to
    ``config.clips_dir`` (``output/03_clip_full_map/``).
    """
    logger.info("=== Stage 3: Clip full map into tiles ===")

    if not config.geotiff_path:
        raise ValueError("geotiff_path is required for clip_full_map stage")

    # Read from 02_result junction (points to 02 or 02a)
    result_dir = config.mask_full_map_result
    if not os.path.isdir(result_dir):
        # Fallback: direct stage 2 output (no junction set up)
        result_dir = config.mask_full_map_dir
    from progress import report_progress
    report_progress(2, "Loading GeoTIFF...")
    _clip_full_map(config.geotiff_path, config.clips_dir, result_dir,
                   max_workers=config.max_workers)
    report_progress(100, "Clipping complete")
    logger.info("Clipping complete. Clips saved to %s", config.clips_dir)


def _union_layout_masks(layouts_json_path: str, layouts_dir: str):
    """Load all layout masks and merge them with pixel-wise OR.

    Returns a PIL Image (mode 'L') representing the union of all layout masks.
    """
    import json
    from PIL import Image
    import numpy as np

    with open(layouts_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    layouts = data.get("layouts", [])
    union_arr = None

    for layout in layouts:
        mask_file = layout.get("mask_file", "")
        if not mask_file:
            continue
        mask_path = os.path.join(layouts_dir, mask_file)
        if not os.path.isfile(mask_path):
            logger.warning("Layout mask not found: %s", mask_path)
            continue

        mask_img = Image.open(mask_path).convert("L")
        mask_arr = np.array(mask_img)

        if union_arr is None:
            union_arr = mask_arr.copy()
        else:
            if union_arr.shape == mask_arr.shape:
                union_arr = np.maximum(union_arr, mask_arr)
            else:
                logger.warning("Layout mask size mismatch: %s vs %s, skipping",
                               union_arr.shape, mask_arr.shape)

    if union_arr is None:
        return None
    return Image.fromarray(union_arr, mode="L")


def _clip_full_map(src_img_file: str, clips_output_dir: str, mask_full_map_dir: str,
                   max_workers: int = 1) -> None:
    """Clip the full GeoTIFF into smaller tiles based on the mask.

    All outputs (clips, visualization) go to *clips_output_dir*.
    The mask/modelscale images are read from *mask_full_map_dir* (stage 2 output).
    """
    from PIL import Image
    from geo_sam3_utils2 import generate_clip_boxes2, visualize_clip_boxes
    from geo_sam3_image import GeoSam3Image

    os.makedirs(clips_output_dir, exist_ok=True)

    geo_image = GeoSam3Image(src_img_file)
    if not geo_image.has_model_scale_image():
        geo_image.generate_model_scale_image()

    # Read layouts.json from the result directory (02_result junction)
    # The junction points to either 02 (auto) or 02a (manual),
    # both of which contain layouts.json + layout mask PNGs.
    layouts_json = os.path.join(mask_full_map_dir, "layouts.json")
    merged_mask = None

    if os.path.isfile(layouts_json):
        merged_mask = _union_layout_masks(layouts_json, mask_full_map_dir)
        if merged_mask is not None:
            import json
            with open(layouts_json, "r", encoding="utf-8") as f:
                n_layouts = len(json.load(f).get("layouts", []))
            logger.info("Using merged mask from %d track layout(s)", n_layouts)

    # Fall back to merged_mask.png in the same directory
    if merged_mask is None:
        merged_mask_path = os.path.join(mask_full_map_dir, "merged_mask.png")
        if os.path.isfile(merged_mask_path):
            merged_mask = Image.open(merged_mask_path)
            logger.info("Loaded merged mask from %s", merged_mask_path)
        else:
            merged_mask = geo_image.merge_all_masks(mode='union')

    if merged_mask is None:
        raise RuntimeError(
            f"No merged mask available for clipping.  "
            f"Expected at {os.path.join(mask_full_map_dir, 'merged_mask.png')}.  Run stage 2 first."
        )

    target_clip_size_in_meters = 40
    geo_image_width_in_meters = geo_image.geo_image.get_gsd()[0] * geo_image.geo_image.width / 100
    geo_image_height_in_meters = geo_image.geo_image.get_gsd()[1] * geo_image.geo_image.height / 100

    target_clip_width = target_clip_size_in_meters / geo_image_width_in_meters
    target_clip_height = target_clip_size_in_meters / geo_image_height_in_meters
    clip_boxes = generate_clip_boxes2(merged_mask, (target_clip_width, target_clip_height), 0.1)
    logger.info("Generated %d clip boxes", len(clip_boxes))

    # Save visualization to output dir (not source dir)
    visualize_clip_boxes(merged_mask, clip_boxes, show_plot=False,
                         save_path=os.path.join(clips_output_dir, "clip_boxes_visualization.png"))

    total = len(clip_boxes)
    source_gsd = geo_image.geo_image.get_gsd()[0]
    max_workers = max(1, max_workers)
    _tracker = ProgressTracker(total=total, pct_start=5, pct_end=95)

    def _process_clip(idx_box):
        """Process a single clip: crop, generate modelscale, save, close."""
        i, box = idx_box
        logger.info("Clipping %d/%d ...", i + 1, total)
        _tracker.update(i + 1, f"Clipping {i+1}/{total}")
        cropped = geo_image.crop_and_scale_to_gsd(
            box, source_gsd,
            dst_image_path=os.path.join(clips_output_dir, f"clip_{i}.tif"),
        )
        cropped.generate_model_scale_image()
        cropped.save(save_masks=False, output_dir=clips_output_dir)
        cropped.geo_image.close()
        return i

    if max_workers == 1:
        for i, box in enumerate(clip_boxes):
            _process_clip((i, box))
    else:
        logger.info("Clipping %d boxes with %d workers", total, max_workers)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_process_clip, (i, box)): i
                for i, box in enumerate(clip_boxes)
            }
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    clip_idx = futures[future]
                    logger.error("Failed to clip %d: %s", clip_idx, e)
                    raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Stage 3: Clip full map into tiles")
    p.add_argument("--geotiff", required=True, help="Path to GeoTIFF image")
    p.add_argument("--output-dir", default="output", help="Output base directory")
    args = p.parse_args()
    config = PipelineConfig(geotiff_path=args.geotiff, output_dir=args.output_dir).resolve()
    run(config)
