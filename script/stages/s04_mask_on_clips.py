"""Stage 4: Run SAM3 segmentation on each clip tile."""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys

logger = logging.getLogger("sam3_pipeline.s04")

_script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from pipeline_config import PipelineConfig


def run(config: PipelineConfig) -> None:
    """Execute Stage 4: Per-clip SAM3 segmentation.

    Reads clips from ``config.clips_dir`` (stage 3 output),
    writes per-tag masks to ``config.mask_on_clips_dir`` (``output/04_mask_on_clips/``).
    """
    logger.info("=== Stage 4: Per-clip SAM3 segmentation ===")

    if not config.clips_dir or not os.path.isdir(config.clips_dir):
        raise ValueError(f"clips_dir not found: {config.clips_dir}")

    _generate_mask_on_clips(
        clips_dir=config.clips_dir,
        output_dir=config.mask_on_clips_dir,
        prompts=config.sam3_prompts,
    )
    logger.info("Per-clip segmentation complete. Output: %s", config.mask_on_clips_dir)


def _generate_mask_on_clips(
    clips_dir: str,
    output_dir: str,
    prompts: list | None = None,
) -> None:
    """Run SAM3 on each clip tile for multiple material tags.

    Reads clip TIF files from *clips_dir* (stage 3 output).
    Writes per-tag masks to *output_dir*/{tag}/ (stage 4 output).

    When a prompt has ``fallback_prompts``, clips that produce no masks
    with the primary prompt are retried using each fallback prompt in order
    until a detection succeeds.
    """
    import sam3
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    from geo_sam3_image import GeoSam3Image

    os.makedirs(output_dir, exist_ok=True)

    sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")

    if prompts is None:
        prompts = [
            {"tag": "road", "prompt": "race track surface"},
            {"tag": "grass", "prompt": "grass"},
            {"tag": "sand", "prompt": "sand surface"},
            {"tag": "kerb", "prompt": "race track curb", "threshold": 0.2},
        ]

    clips = [f for f in os.listdir(clips_dir) if re.match(r'clip_\d+\.tif', f)]
    if len(clips) == 0:
        logger.warning("No clips found in %s", clips_dir)
        return

    logger.info("Found %d clips", len(clips))
    bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
    checkpoint_path = f"{sam3_root}/../model/sam3.pt"
    model = build_sam3_image_model(bpe_path=bpe_path, checkpoint_path=checkpoint_path, load_from_HF=False)

    for inp in prompts:
        tag = inp["tag"]
        threshold = inp.get("threshold", 0.4)
        fallback_prompts = inp.get("fallback_prompts", [])
        target_path = os.path.join(output_dir, tag)
        os.makedirs(target_path, exist_ok=True)

        # Track clips that got no masks from the primary prompt
        no_mask_clips = []

        for clip in clips:
            geo_image = GeoSam3Image(os.path.join(clips_dir, clip))
            if not geo_image.has_model_scale_image():
                geo_image.generate_model_scale_image()

            image = geo_image.model_scale_image
            processor = Sam3Processor(model, confidence_threshold=threshold)
            inference_state = processor.set_image(image)
            processor.reset_all_prompts(inference_state)
            inference_state = processor.set_text_prompt(
                state=inference_state, prompt=inp["prompt"]
            )

            geo_image.set_masks_from_inference_state(inference_state, tag=tag)

            has_masks = _inference_has_masks(inference_state, threshold)
            if not has_masks and fallback_prompts:
                no_mask_clips.append(clip)

            geo_image.save(save_masks=True, overwrite=True, output_dir=target_path)

        # Fallback: retry clips with no masks using alternative prompts
        if no_mask_clips and fallback_prompts:
            logger.info(
                "[%s] %d/%d clips have no masks, trying fallback prompts: %s",
                tag, len(no_mask_clips), len(clips), fallback_prompts,
            )
            _run_fallback(
                clips_dir, target_path, no_mask_clips, tag,
                fallback_prompts, threshold, model,
            )


def _inference_has_masks(inference_state: dict, threshold: float) -> bool:
    """Check whether an inference state contains any masks above *threshold*."""
    masks = inference_state.get("masks")
    scores = inference_state.get("scores", [])
    if masks is None or len(masks) == 0:
        return False
    for s in scores:
        if float(s) >= threshold:
            return True
    return False


def _run_fallback(
    clips_dir: str,
    target_path: str,
    no_mask_clips: list,
    tag: str,
    fallback_prompts: list,
    threshold: float,
    model,
) -> None:
    """Try fallback prompts on clips that produced no masks.

    For each clip, tries each fallback prompt in order and uses the first
    one that produces masks above threshold.
    """
    from sam3.model.sam3_image_processor import Sam3Processor
    from geo_sam3_image import GeoSam3Image

    recovered = 0
    for clip in no_mask_clips:
        clip_name = os.path.splitext(clip)[0]
        found = False

        for fb_prompt in fallback_prompts:
            geo_image = GeoSam3Image(os.path.join(clips_dir, clip))
            if not geo_image.has_model_scale_image():
                geo_image.generate_model_scale_image()

            image = geo_image.model_scale_image
            processor = Sam3Processor(model, confidence_threshold=threshold)
            inference_state = processor.set_image(image)
            processor.reset_all_prompts(inference_state)
            inference_state = processor.set_text_prompt(
                state=inference_state, prompt=fb_prompt
            )

            if _inference_has_masks(inference_state, threshold):
                geo_image.set_masks_from_inference_state(inference_state, tag=tag)
                geo_image.save(save_masks=True, overwrite=True, output_dir=target_path)
                recovered += 1
                found = True

                # Log the best score
                scores = inference_state.get("scores", [])
                best = max(float(s) for s in scores) if len(scores) > 0 else 0
                logger.info(
                    "[%s] %s recovered via fallback '%s' (best_score=%.2f)",
                    tag, clip_name, fb_prompt, best,
                )
                break

        if not found:
            logger.warning(
                "[%s] %s: no masks from any fallback prompt", tag, clip_name
            )

    logger.info(
        "[%s] Fallback complete: %d/%d clips recovered",
        tag, recovered, len(no_mask_clips),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Stage 4: Per-clip SAM3 segmentation")
    p.add_argument("--clips-dir", required=True, help="Directory with clip TIF files (stage 3 output)")
    p.add_argument("--output-dir", default="output", help="Output base directory")
    args = p.parse_args()
    config = PipelineConfig(output_dir=args.output_dir).resolve()
    # Override clips_dir for standalone usage
    config.clips_dir = os.path.abspath(args.clips_dir)
    run(config)
