"""
SAM3 Track Segmentation Pipeline -- Main Entry Point

Full pipeline:
  1. B3DM -> GLB conversion
  2. mask_full_map() -- SAM3 segmentation on full GeoTIFF
  3. clip_full_map() -- clip image into tiles for per-tile processing
  4. generate_mask_on_clips() -- SAM3 per-clip segmentation
  5. convert_mask_to_blender_input() -- geo -> blender coords + consolidation
  6. AI wall generation -- generates wall JSON + preview
  7. AI game object generation -- generates objects JSON + preview
  8. Blender polygon generation with mesh conversion -- via subprocess
  9. Blender headless automation (load tiles, refine, extract, import, save)

Usage:
    python sam3_track_gen.py --geotiff path/to/result.tif --tiles-dir path/to/b3dm --output-dir output
    python sam3_track_gen.py --stage b3dm_convert --tiles-dir path/to/b3dm --output-dir output
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Callable, Dict, List, Optional

# ---------------------------------------------------------------------------
# Ensure script/ is on sys.path
# ---------------------------------------------------------------------------
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from pipeline_config import PipelineConfig, PIPELINE_STAGES, _DEFAULT_OUTPUT_DIR, _PROJECT_ROOT
from stages import (
    s01_b3dm_convert,
    s02_mask_full_map,
    s03_clip_full_map,
    s04_mask_on_clips,
    s05_convert_to_blender,
    s06_ai_walls,
    s07_ai_game_objects,
    s08_blender_polygons,
    s09_blender_automate,
    s10_model_export,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sam3_pipeline")

# ---------------------------------------------------------------------------
# Stage registry
# ---------------------------------------------------------------------------
STAGE_FUNCTIONS: Dict[str, Callable[[PipelineConfig], None]] = {
    "b3dm_convert": s01_b3dm_convert.run,
    "mask_full_map": s02_mask_full_map.run,
    "clip_full_map": s03_clip_full_map.run,
    "mask_on_clips": s04_mask_on_clips.run,
    "convert_to_blender": s05_convert_to_blender.run,
    "ai_walls": s06_ai_walls.run,
    "ai_game_objects": s07_ai_game_objects.run,
    "blender_polygons": s08_blender_polygons.run,
    "blender_automate": s09_blender_automate.run,
    "model_export": s10_model_export.run,
}


# ---------------------------------------------------------------------------
# Pipeline Runner
# ---------------------------------------------------------------------------
def _load_manual_stages_config(config: PipelineConfig) -> Dict[str, bool]:
    """Read manual_stages from webtools_config.json if available."""
    config_json = os.path.join(config.output_dir, "webtools_config.json")
    if os.path.isfile(config_json):
        try:
            import json
            with open(config_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("manual_stages", {})
        except Exception:
            pass
    return {}


def run_pipeline(config: PipelineConfig, stages: Optional[List[str]] = None) -> None:
    """Run the pipeline, either all stages or a subset."""
    if stages is None:
        stages = list(PIPELINE_STAGES)

    # Set up result directory junctions before running any stage
    manual_cfg = _load_manual_stages_config(config)
    config.setup_result_junctions(manual_cfg)

    logger.info("Pipeline starting with %d stages: %s", len(stages), stages)
    logger.info("Output directory: %s", config.output_dir)

    for stage_name in stages:
        func = STAGE_FUNCTIONS.get(stage_name)
        if func is None:
            raise ValueError(
                f"Unknown stage: {stage_name!r}. "
                f"Available: {sorted(STAGE_FUNCTIONS.keys())}"
            )
        try:
            func(config)
        except Exception as e:
            logger.error("Stage '%s' failed: %s", stage_name, e)
            raise RuntimeError(f"Pipeline failed at stage '{stage_name}': {e}") from e

    logger.info("Pipeline complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    p = argparse.ArgumentParser(
        description="SAM3 Track Segmentation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run the full pipeline
  python sam3_track_gen.py \\
      --geotiff test_images_shajing/result.tif \\
      --tiles-dir test_images_shajing/b3dm \\
      --output-dir output \\
      --blender-exe /path/to/blender

  # Run only B3DM conversion
  python sam3_track_gen.py --stage b3dm_convert \\
      --tiles-dir test_images_shajing/b3dm --output-dir output

  # Run only AI stages
  python sam3_track_gen.py --stage ai_walls --stage ai_game_objects \\
      --geotiff test_images_shajing/result.tif --output-dir output

Available stages: """ + ", ".join(PIPELINE_STAGES)
    )

    p.add_argument("--geotiff", default="", help="Path to the GeoTIFF image (result.tif)")
    p.add_argument("--tiles-dir", default="", help="Directory with b3dm files and tileset.json")
    p.add_argument("--output-dir", default=_DEFAULT_OUTPUT_DIR,
                    help=f"Output directory (default: {_DEFAULT_OUTPUT_DIR})")
    p.add_argument("--blender-exe", default="", help="Path to the Blender executable")
    p.add_argument(
        "--stage", action="append", dest="stages", default=None,
        help="Run specific stage(s). Can be repeated. If omitted, runs all.",
    )
    p.add_argument("--gemini-api-key", default="", help="Gemini API key (default: built-in key)")
    p.add_argument("--gemini-model", default="gemini-2.5-pro", help="Gemini model name")
    p.add_argument(
        "--track-direction", default="clockwise", choices=["clockwise", "counterclockwise"],
        help="Track driving direction (default: clockwise)",
    )
    p.add_argument("--track-description", default="", help="Optional track description for AI prompts")
    p.add_argument(
        "--inpaint-model", default="",
        help='Inpainting model (default: gemini-2.5-flash-image). Use "disabled" to skip.',
    )

    # --- Stage 8 options ---
    p.add_argument("--generate-curves", action="store_true",
                    help="Generate diagnostic 2D curves in Stage 8 (default: skip)")
    p.add_argument("--no-gap-fill", action="store_true",
                    help="Skip Stage 8 mask gap auto-filling")
    p.add_argument("--gap-fill-threshold", type=float, default=0.20,
                    help="Gap fill threshold in metres (default: 0.20)")
    p.add_argument("--gap-fill-default-tag", default="road2",
                    help="Default fill tag for remaining voids (default: road2)")

    # --- Stage 9 options ---
    p.add_argument("--base-level", type=int, default=0,
                    help="Base tile level for Blender import (default: use config)")
    p.add_argument("--target-level", type=int, default=0,
                    help="Target refinement level (default: use config)")
    p.add_argument("--no-walls", action="store_true",
                    help="Skip wall import in Stage 9")
    p.add_argument("--no-game-objects", action="store_true",
                    help="Skip game object import in Stage 9")
    p.add_argument("--no-surfaces", action="store_true",
                    help="Skip surface extraction in Stage 9")
    p.add_argument("--no-textures", action="store_true",
                    help="Skip texture processing in Stage 9")
    p.add_argument("--no-background", action="store_true",
                    help="Run Blender without --background (show GUI)")
    p.add_argument("--refine-tags", default="",
                    help="Comma-separated mask tags for tile refinement (default: road)")

    # --- Surface extraction options ---
    p.add_argument("--edge-simplify", type=float, default=0.0,
                    help="Edge simplification epsilon in metres (0 = no simplification)")
    p.add_argument("--density-road", type=float, default=0.0,
                    help="Sampling density for road surfaces in metres (0 = use config default)")
    p.add_argument("--density-kerb", type=float, default=0.0,
                    help="Sampling density for kerb surfaces in metres (0 = use config default)")
    p.add_argument("--density-grass", type=float, default=0.0,
                    help="Sampling density for grass surfaces in metres (0 = use config default)")
    p.add_argument("--density-sand", type=float, default=0.0,
                    help="Sampling density for sand surfaces in metres (0 = use config default)")
    p.add_argument("--density-road2", type=float, default=0.0,
                    help="Sampling density for road2 surfaces in metres (0 = use config default)")

    # --- Mesh simplification options ---
    p.add_argument("--mesh-simplify", action="store_true",
                    help="Enable weld + decimate for terrain collision meshes (road/kerb)")
    p.add_argument("--mesh-weld-distance", type=float, default=0.0,
                    help="Weld distance in metres (0 = use config default 0.01)")
    p.add_argument("--mesh-decimate-ratio", type=float, default=0.0,
                    help="Decimate ratio 0-1 (0 = use config default 0.5)")

    # --- Stage 10 options ---
    p.add_argument("--max-vertices", type=int, default=0,
                    help="Max vertices per exported mesh object (0 = use config default 21000)")
    p.add_argument("--max-batch-mb", type=int, default=0,
                    help="Max FBX file size in MB (0 = use config default 100)")
    p.add_argument("--fbx-scale", type=float, default=0.0,
                    help="FBX export global scale (0 = use config default 0.01)")
    p.add_argument("--ks-ambient", type=float, default=-1.0,
                    help="ksAmbient for visible materials (-1 = use config default)")
    p.add_argument("--ks-diffuse", type=float, default=-1.0,
                    help="ksDiffuse for visible materials (-1 = use config default)")
    p.add_argument("--ks-emissive", type=float, default=-1.0,
                    help="ksEmissive for visible materials (-1 = use config default)")
    p.add_argument("--kseditor-exe", default="",
                    help="Path to ksEditorAT.exe for FBX→KN5 conversion")

    return p


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    """Build a PipelineConfig from parsed CLI arguments."""
    config = PipelineConfig(
        geotiff_path=args.geotiff,
        tiles_dir=args.tiles_dir,
        output_dir=args.output_dir,
        track_direction=args.track_direction,
        track_description=args.track_description,
        gemini_model=args.gemini_model,
    )
    if args.blender_exe:
        config.blender_exe = args.blender_exe
    if args.gemini_api_key:
        config.gemini_api_key = args.gemini_api_key
    if args.inpaint_model:
        if args.inpaint_model == "disabled":
            config.inpaint_center_holes = False
        else:
            config.inpaint_model = args.inpaint_model

    # Stage 8 options
    config.s8_generate_curves = args.generate_curves
    config.s8_gap_fill_enabled = not args.no_gap_fill
    config.s8_gap_fill_threshold_m = args.gap_fill_threshold
    config.s8_gap_fill_default_tag = args.gap_fill_default_tag

    # Stage 9 options
    if args.base_level:
        config.base_level = args.base_level
    if args.target_level:
        config.target_fine_level = args.target_level
    config.s9_no_walls = args.no_walls
    config.s9_no_game_objects = args.no_game_objects
    config.s9_no_surfaces = args.no_surfaces
    config.s9_no_textures = args.no_textures
    config.s9_no_background = args.no_background
    if args.refine_tags:
        config.s9_refine_tags = [t.strip() for t in args.refine_tags.split(",") if t.strip()]

    # Surface extraction options
    if args.edge_simplify > 0:
        config.surface_edge_simplify = args.edge_simplify
    if args.density_road > 0:
        config.surface_density_road = args.density_road
    if args.density_kerb > 0:
        config.surface_density_kerb = args.density_kerb
    if args.density_grass > 0:
        config.surface_density_grass = args.density_grass
    if args.density_sand > 0:
        config.surface_density_sand = args.density_sand
    if args.density_road2 > 0:
        config.surface_density_road2 = args.density_road2

    # Mesh simplification options
    if args.mesh_simplify:
        config.s9_mesh_simplify = True
    if args.mesh_weld_distance > 0:
        config.s9_mesh_weld_distance = args.mesh_weld_distance
    if args.mesh_decimate_ratio > 0:
        config.s9_mesh_decimate_ratio = args.mesh_decimate_ratio

    # Stage 10 options
    if args.max_vertices > 0:
        config.s10_max_vertices = args.max_vertices
    if args.max_batch_mb > 0:
        config.s10_max_batch_mb = args.max_batch_mb
    if args.fbx_scale > 0:
        config.s10_fbx_scale = args.fbx_scale
    if args.ks_ambient >= 0:
        config.s10_ks_ambient = args.ks_ambient
    if args.ks_diffuse >= 0:
        config.s10_ks_diffuse = args.ks_diffuse
    if args.ks_emissive >= 0:
        config.s10_ks_emissive = args.ks_emissive
    if args.kseditor_exe:
        config.s10_kseditor_exe = args.kseditor_exe

    return config.resolve()


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    config = config_from_args(args)
    run_pipeline(config, stages=args.stages)


if __name__ == "__main__":
    main()
