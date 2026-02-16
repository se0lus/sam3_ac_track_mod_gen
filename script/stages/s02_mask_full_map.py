"""Stage 2: Run SAM3 segmentation on the full GeoTIFF."""
from __future__ import annotations

import argparse
import logging
import os
import sys

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
    """
    logger.info("=== Stage 2: Full map SAM3 segmentation ===")

    if not config.geotiff_path:
        raise ValueError("geotiff_path is required for mask_full_map stage")

    geo_image = _mask_full_map(config.geotiff_path, config.mask_full_map_dir)
    if geo_image is None:
        raise RuntimeError("Failed to generate mask for full map")
    logger.info("Full map segmentation complete. Output: %s", config.mask_full_map_dir)


def _mask_full_map(src_img_file: str, output_dir: str):
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Stage 2: Full map SAM3 segmentation")
    p.add_argument("--geotiff", required=True, help="Path to GeoTIFF image")
    p.add_argument("--output-dir", default="output", help="Output base directory")
    args = p.parse_args()
    config = PipelineConfig(geotiff_path=args.geotiff, output_dir=args.output_dir).resolve()
    run(config)
