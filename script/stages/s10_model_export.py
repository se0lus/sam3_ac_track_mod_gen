"""Stage 10: Model export — split and export from final Blender file.

Reads ``09_result/final_track.blend`` and exports split models.
Currently a skeleton — actual export logic is not yet implemented.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

logger = logging.getLogger("sam3_pipeline.s10")

_script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from pipeline_config import PipelineConfig


def run(config: PipelineConfig) -> None:
    """Execute Stage 10: Model export (skeleton).

    Reads:
    - ``config.blender_result_dir / final_track.blend`` from stage 9

    Writes to ``config.export_dir`` (``output/10_model_export/``).
    """
    logger.info("=== Stage 10: Model export ===")

    # Locate input blend file
    blend_input = os.path.join(config.blender_result_dir, "final_track.blend")
    if not os.path.isfile(blend_input):
        raise FileNotFoundError(
            f"Input blend file not found: {blend_input}\n"
            "Run Stage 9 (blender_automate) first."
        )

    # Create output directory
    os.makedirs(config.export_dir, exist_ok=True)

    logger.info("Input:  %s", blend_input)
    logger.info("Output: %s", config.export_dir)
    logger.info("[placeholder] Model export logic not yet implemented.")
    logger.info("=== Stage 10 complete (skeleton) ===")


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 10: Model export")
    parser.add_argument("--output-dir", default="output", help="Output base directory")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = PipelineConfig(output_dir=args.output_dir).resolve()
    run(config)


if __name__ == "__main__":
    main()
