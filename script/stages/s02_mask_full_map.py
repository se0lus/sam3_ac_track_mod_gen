"""Stage 2: Run SAM3 segmentation on the full GeoTIFF.

Generates per-tag masks at full-map level for wall generation reference
(road, trees, grass) in addition to the merged road mask.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
from PIL import Image

logger = logging.getLogger("sam3_pipeline.s02")

_script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from pipeline_config import PipelineConfig


def run(config: PipelineConfig) -> None:
    """Execute Stage 2: Full-map SAM3 segmentation.

    Outputs go to ``config.mask_full_map_dir``:
    - result_modelscale.png
    - result_mask0_prob(...).png
    - result_masks.json
    - merged_mask.png
    - results_visualization.png
    - {tag}_mask.png for each tag in sam3_fullmap_tags
    """
    logger.info("=== Stage 2: Full map SAM3 segmentation ===")

    if not config.geotiff_path:
        raise ValueError("geotiff_path is required for mask_full_map stage")

    geo_image = _mask_full_map(config.geotiff_path, config.mask_full_map_dir, config)
    if geo_image is None:
        raise RuntimeError("Failed to generate mask for full map")

    # Generate per-tag masks for wall generation reference
    _generate_fullmap_tag_masks(
        geo_image,
        config.mask_full_map_dir,
        config.sam3_prompts,
        config.sam3_fullmap_tags,
    )

    # Generate VLM-scale image (higher resolution for VLM input in stage 8)
    _generate_vlmscale_image(geo_image, config)

    # Generate default layout + geo_metadata for downstream stages
    _generate_default_layout(config)
    _generate_geo_metadata(config)

    logger.info("Full map segmentation complete. Output: %s", config.mask_full_map_dir)


def _mask_full_map(src_img_file: str, output_dir: str, config: PipelineConfig | None = None):
    """Run SAM3 segmentation on the full GeoTIFF and return the GeoSam3Image.

    All outputs (modelscale image, masks, visualization, merged mask) are
    written to *output_dir* instead of the source directory.
    """
    import sam3
    import matplotlib.pyplot as plt
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.visualization_utils import plot_results
    from geo_sam3_image import GeoSam3Image

    os.makedirs(output_dir, exist_ok=True)

    sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")

    geo_image = GeoSam3Image(src_img_file)
    if not geo_image.has_model_scale_image():
        geo_image.generate_model_scale_image()

    # Detect and inpaint center holes before SAM3 inference
    if config and config.inpaint_center_holes:
        from image_inpainter import detect_center_holes, inpaint_holes

        hole_mask = detect_center_holes(
            geo_image.model_scale_image,
            min_hole_ratio=config.inpaint_min_hole_ratio,
        )
        if hole_mask is not None:
            hole_pct = np.sum(hole_mask > 0) / hole_mask.size * 100
            logger.info(
                "Center holes detected (%.1f%%), inpainting with %s...",
                hole_pct, config.inpaint_model,
            )
            # Save debug artifacts
            Image.fromarray(hole_mask).save(
                os.path.join(output_dir, "holes_mask.png")
            )
            geo_image.model_scale_image.save(
                os.path.join(output_dir, "result_modelscale_original.png")
            )
            # Inpaint and replace
            geo_image.model_scale_image = inpaint_holes(
                geo_image.model_scale_image,
                hole_mask,
                api_key=config.gemini_api_key,
                model_name=config.inpaint_model,
            )
            geo_image.model_scale_image.save(
                os.path.join(output_dir, "result_modelscale_inpainted.png")
            )
            logger.info("Inpainting complete")
        else:
            logger.info("No center holes detected, skipping inpainting")

    need_inference = not geo_image.has_masks()
    inference_state = None

    if need_inference:
        bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
        checkpoint_path = f"{sam3_root}/../model/sam3.pt"
        model = build_sam3_image_model(bpe_path=bpe_path, checkpoint_path=checkpoint_path, load_from_HF=False)

        image = geo_image.model_scale_image
        processor = Sam3Processor(model, confidence_threshold=0.2)
        inference_state = processor.set_image(image)
        processor.reset_all_prompts(inference_state)
        inference_state = processor.set_text_prompt(state=inference_state, prompt="race track surface")

        geo_image.set_masks_from_inference_state(inference_state)

    # Always save modelscale + masks to output_dir (even if pre-existing in source)
    geo_image.save(save_masks=True, output_dir=output_dir)

    # Save visualization (only if we ran fresh inference)
    if need_inference and inference_state is not None:
        plot_results(geo_image.model_scale_image, inference_state)
        plt.axis('off')
        plt.tight_layout()
        save_path = os.path.join(output_dir, "results_visualization.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=150, pad_inches=0.1)
        logger.info("Visualization saved: %s", save_path)
        plt.close()

    # Always save merged mask to output_dir
    merged_mask = geo_image.merge_all_masks(mode='union')
    if merged_mask is None:
        logger.error("No merged mask produced")
        return None
    merged_mask.save(os.path.join(output_dir, "merged_mask.png"))

    return geo_image


def _generate_fullmap_tag_masks(
    geo_image,
    output_dir: str,
    prompts: list,
    fullmap_tags: list,
) -> None:
    """Run SAM3 with additional prompts and save per-tag masks at full-map level.

    For each tag in *fullmap_tags* (except "road" which is already generated),
    loads the SAM3 model, runs inference, and saves ``{tag}_mask.png``.
    """
    # Build prompt lookup
    prompt_lookup = {p["tag"]: p for p in prompts}

    # Check which tags still need generation
    tags_to_generate = []
    for tag in fullmap_tags:
        if tag == "road":
            continue  # already generated as merged_mask.png
        if tag not in prompt_lookup:
            logger.warning("Tag '%s' in sam3_fullmap_tags but not in sam3_prompts, skipping", tag)
            continue
        mask_path = os.path.join(output_dir, f"{tag}_mask.png")
        if os.path.isfile(mask_path):
            logger.info("Tag '%s' mask already exists: %s", tag, mask_path)
            continue
        tags_to_generate.append(tag)

    if not tags_to_generate:
        logger.info("All full-map tag masks already exist, skipping")
        return

    logger.info("Generating full-map masks for tags: %s", tags_to_generate)

    import sam3
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
    bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
    checkpoint_path = f"{sam3_root}/../model/sam3.pt"
    model = build_sam3_image_model(bpe_path=bpe_path, checkpoint_path=checkpoint_path, load_from_HF=False)

    image = geo_image.model_scale_image

    for tag in tags_to_generate:
        cfg = prompt_lookup[tag]
        threshold = cfg.get("threshold", 0.4)
        prompt_text = cfg["prompt"]

        logger.info("Running SAM3 for tag '%s' (prompt='%s', threshold=%.2f)", tag, prompt_text, threshold)

        processor = Sam3Processor(model, confidence_threshold=threshold)
        inference_state = processor.set_image(image)
        processor.reset_all_prompts(inference_state)
        inference_state = processor.set_text_prompt(state=inference_state, prompt=prompt_text)

        # Extract masks from inference state and merge into a single tag mask
        masks = inference_state.get("masks")
        scores = inference_state.get("scores", [])

        if masks is None or len(masks) == 0:
            logger.warning("No masks generated for tag '%s'", tag)
            continue

        # Convert masks to numpy, merge all masks for this tag
        import torch
        merged = None
        for i, mask in enumerate(masks):
            score = float(scores[i]) if i < len(scores) else 0.0
            if score < threshold:
                continue
            if isinstance(mask, torch.Tensor):
                mask_np = mask.cpu().numpy()
            else:
                mask_np = np.array(mask)
            if mask_np.ndim > 2:
                mask_np = mask_np.squeeze()
            if mask_np.dtype == bool:
                mask_np = mask_np.astype(np.uint8) * 255
            elif mask_np.max() > 1.0:
                mask_np = (mask_np / mask_np.max() * 255).astype(np.uint8)
            else:
                mask_np = (mask_np * 255).astype(np.uint8)

            if merged is None:
                merged = mask_np.astype(np.float32)
            else:
                merged = np.maximum(merged, mask_np.astype(np.float32))

        if merged is not None:
            merged_img = Image.fromarray(merged.astype(np.uint8), mode='L')
            mask_path = os.path.join(output_dir, f"{tag}_mask.png")
            merged_img.save(mask_path)
            logger.info("Saved %s mask: %s (size=%s)", tag, mask_path, merged_img.size)
        else:
            logger.warning("No masks above threshold for tag '%s'", tag)


def _generate_vlmscale_image(geo_image, config: PipelineConfig) -> None:
    """Generate a higher-resolution image for VLM input (stage 8).

    Uses the same scaling + inpainting pipeline as modelscale but with
    ``config.vlm_max_size`` (default 3072) to give the VLM more detail.
    """
    output_dir = config.mask_full_map_dir
    basename = os.path.splitext(os.path.basename(config.geotiff_path))[0]
    vlm_path = os.path.join(output_dir, f"{basename}_vlmscale.png")

    if os.path.isfile(vlm_path):
        logger.info("VLM-scale image already exists: %s", vlm_path)
        return

    vlm_img = geo_image.geo_image.scale_to_max_size(max_size=config.vlm_max_size)
    if vlm_img.mode != "RGB":
        vlm_img = vlm_img.convert("RGB")

    # Inpaint center holes on vlmscale image (same pipeline as modelscale)
    if config.inpaint_center_holes:
        from image_inpainter import detect_center_holes, inpaint_holes

        hole_mask = detect_center_holes(
            vlm_img, min_hole_ratio=config.inpaint_min_hole_ratio
        )
        if hole_mask is not None:
            hole_pct = np.sum(hole_mask > 0) / hole_mask.size * 100
            logger.info(
                "VLM-scale: center holes detected (%.1f%%), inpainting...", hole_pct
            )
            vlm_img.save(os.path.join(output_dir, f"{basename}_vlmscale_original.png"))
            vlm_img = inpaint_holes(
                vlm_img,
                hole_mask,
                api_key=config.gemini_api_key,
                model_name=config.inpaint_model,
            )

    vlm_img.save(vlm_path)
    logger.info("VLM-scale image saved: %s (%s)", vlm_path, vlm_img.size)


def _generate_default_layout(config: PipelineConfig) -> None:
    """Generate Default layout mask + layouts.json in stage 2 output.

    The merged_mask.png is copied as ``Default.png`` and a ``layouts.json``
    is written so that downstream stages (3, 5, 7, 8) can read layouts
    directly from the 02_result junction without requiring stage 2a.
    """
    import json, shutil

    output_dir = config.mask_full_map_dir
    merged_mask = os.path.join(output_dir, "merged_mask.png")
    default_png = os.path.join(output_dir, "Default.png")
    layouts_json = os.path.join(output_dir, "layouts.json")

    if not os.path.isfile(merged_mask):
        logger.warning("merged_mask.png not found, skipping default layout generation")
        return

    # Copy merged_mask as Default.png (only if not already present)
    if not os.path.isfile(default_png):
        shutil.copy2(merged_mask, default_png)
        logger.info("Default layout mask: %s", default_png)

    # Write layouts.json (only if not already present)
    if not os.path.isfile(layouts_json):
        data = {
            "layouts": [{
                "name": "Default",
                "mask_file": "Default.png",
                "track_direction": config.track_direction,
            }]
        }
        with open(layouts_json, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info("Default layouts.json written: %s", layouts_json)


def _generate_geo_metadata(config: PipelineConfig) -> None:
    """Extract geo metadata from result_masks.json and write geo_metadata.json."""
    import json

    output_dir = config.mask_full_map_dir
    masks_json = os.path.join(output_dir, "result_masks.json")
    geo_meta_path = os.path.join(output_dir, "geo_metadata.json")

    if os.path.isfile(geo_meta_path):
        return  # already exists

    if not os.path.isfile(masks_json):
        logger.warning("result_masks.json not found, skipping geo_metadata.json")
        return

    try:
        with open(masks_json, "r", encoding="utf-8") as f:
            masks_data = json.load(f)

        meta = masks_data.get("meta", {})
        model_scale = meta.get("model_scale_size", {})
        geo_bounds = meta.get("geo", {}).get("bounds", {})

        geo_metadata = {
            "image_width": model_scale.get("width", 0),
            "image_height": model_scale.get("height", 0),
            "bounds": {
                "north": geo_bounds.get("top", 0),
                "south": geo_bounds.get("bottom", 0),
                "east": geo_bounds.get("right", 0),
                "west": geo_bounds.get("left", 0),
            },
        }

        with open(geo_meta_path, "w", encoding="utf-8") as f:
            json.dump(geo_metadata, f, indent=2, ensure_ascii=False)
        logger.info("geo_metadata.json written: %s", geo_meta_path)
    except Exception as e:
        logger.warning("geo_metadata.json generation failed: %s", e)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Stage 2: Full map SAM3 segmentation")
    p.add_argument("--geotiff", required=True, help="Path to GeoTIFF image")
    p.add_argument("--output-dir", default="output", help="Output base directory")
    args = p.parse_args()
    config = PipelineConfig(geotiff_path=args.geotiff, output_dir=args.output_dir).resolve()
    run(config)
