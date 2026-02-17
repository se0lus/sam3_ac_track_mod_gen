"""
Unified pipeline configuration for SAM3 Track Segmentation.

Single source of truth for ALL config.  Replaces both:
- PipelineConfig in sam3_track_gen.py
- blender_scripts/config.py hardcoded values
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Project root (sam3_track_seg/)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "output")

# ---------------------------------------------------------------------------
# Stage ordering -- used by stage_dir() and the pipeline runner
# ---------------------------------------------------------------------------
STAGE_ORDER: Dict[str, int] = {
    "b3dm_convert": 1,
    "mask_full_map": 2,
    "clip_full_map": 3,
    "mask_on_clips": 4,
    "convert_to_blender": 5,
    "blender_polygons": 6,
    "ai_walls": 7,
    "ai_game_objects": 8,
    "blender_automate": 9,
}

PIPELINE_STAGES: List[str] = list(STAGE_ORDER.keys())


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------
@dataclass
class PipelineConfig:
    """Centralised configuration for the full pipeline."""

    # --- Input ---
    geotiff_path: str = ""
    tiles_dir: str = ""  # directory with b3dm files + tileset.json

    # --- Output ---
    output_dir: str = _DEFAULT_OUTPUT_DIR

    # --- Tools ---
    blender_exe: str = r"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe"

    # --- AI ---
    gemini_api_key: str = "***REDACTED_GEMINI_KEY***"
    gemini_model: str = "gemini-2.5-pro"

    # --- Inpainting (center hole repair) ---
    inpaint_center_holes: bool = True
    inpaint_model: str = "gemini-3-pro-image-preview"
    inpaint_min_hole_ratio: float = 0.001  # 0.1% of image area threshold

    # --- VLM input ---
    vlm_max_size: int = 3072  # VLM input image max dimension (Gemini 2.5 Pro supports up to 3072)

    # --- Track metadata ---
    track_direction: str = "clockwise"
    track_description: str = ""

    # --- Tile levels ---
    base_level: int = 17
    target_fine_level: int = 22

    # --- Blender collections ---
    root_curve_collection_name: str = "mask_curve2D_collection"
    root_polygon_collection_name: str = "mask_polygon_collection"
    collision_collection_name: str = "collision"

    # --- Surface sampling densities (metres between grid samples) ---
    surface_density_road: float = 0.5
    surface_density_grass: float = 2.0
    surface_density_kerb: float = 0.5
    surface_density_sand: float = 2.0
    surface_density_default: float = 1.0

    # --- SAM3 segmentation prompts ---
    sam3_prompts: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"tag": "road", "prompt": "race track surface", "threshold": 0.25,
         "fallback_prompts": ["asphalt road", "concrete road"]},
        {"tag": "grass", "prompt": "grass", "threshold": 0.4},
        {"tag": "sand", "prompt": "sand surface", "threshold": 0.4},
        {"tag": "kerb", "prompt": "race track curb", "threshold": 0.2},
        {"tag": "trees", "prompt": "forest", "threshold": 0.3},
        {"tag": "building", "prompt": "building structure", "threshold": 0.4},
        {"tag": "water", "prompt": "water pond", "threshold": 0.4},
        {"tag": "concrete", "prompt": "concrete paved ground", "threshold": 0.4},
    ])

    # Tags to segment at full-map level (stage 2) for wall generation reference
    sam3_fullmap_tags: List[str] = field(default_factory=lambda: [
        "road", "trees", "grass", "kerb", "sand", "building", "water", "concrete",
    ])

    # --- Derived paths (populated by resolve()) ---
    glb_dir: str = ""
    clips_dir: str = ""
    mask_full_map_dir: str = ""
    mask_on_clips_dir: str = ""
    blender_clips_dir: str = ""
    blend_file: str = ""
    walls_json: str = ""
    walls_preview: str = ""
    game_objects_json: str = ""
    game_objects_preview: str = ""
    final_blend_file: str = ""
    mask_image_path: str = ""  # merged road mask
    trees_mask_path: str = ""  # trees mask for wall generation
    grass_mask_path: str = ""  # grass mask for wall generation
    kerb_mask_path: str = ""  # kerb mask for wall generation
    sand_mask_path: str = ""  # sand mask for wall generation
    building_mask_path: str = ""  # building mask for wall generation
    water_mask_path: str = ""  # water mask for wall generation
    concrete_mask_path: str = ""  # concrete mask for wall generation
    track_layouts_json: str = ""  # track layouts metadata
    manual_game_objects_json: str = ""  # manual-edited game objects (stage 8a)

    def stage_dir(self, stage_name: str) -> str:
        """Return ``output/NN_stage_name/`` for a given stage."""
        if stage_name == "track_layouts":
            return os.path.join(self.output_dir, "02a_track_layouts")
        if stage_name == "manual_game_objects":
            return os.path.join(self.output_dir, "08a_manual_game_objects")
        idx = STAGE_ORDER.get(stage_name, 0)
        return os.path.join(self.output_dir, f"{idx:02d}_{stage_name}")

    def resolve(self) -> "PipelineConfig":
        """Derive all intermediate paths from the base settings.

        Call this once after construction (and after any overrides).
        """
        self.output_dir = os.path.abspath(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        if self.geotiff_path:
            self.geotiff_path = os.path.abspath(self.geotiff_path)
        if self.tiles_dir:
            self.tiles_dir = os.path.abspath(self.tiles_dir)

        # Per-stage derived paths
        self.glb_dir = os.path.join(self.stage_dir("b3dm_convert"), "glb")
        self.mask_full_map_dir = self.stage_dir("mask_full_map")
        self.clips_dir = self.stage_dir("clip_full_map")
        self.mask_on_clips_dir = self.stage_dir("mask_on_clips")
        self.blender_clips_dir = self.stage_dir("convert_to_blender")
        self.blend_file = os.path.join(
            self.stage_dir("blender_polygons"), "polygons.blend"
        )
        self.walls_json = os.path.join(self.stage_dir("ai_walls"), "walls.json")
        self.walls_preview = os.path.join(
            self.stage_dir("ai_walls"), "walls_preview.png"
        )
        self.game_objects_json = os.path.join(
            self.stage_dir("ai_game_objects"), "game_objects.json"
        )
        self.game_objects_preview = os.path.join(
            self.stage_dir("ai_game_objects"), "game_objects_preview.png"
        )
        self.final_blend_file = os.path.join(
            self.stage_dir("blender_automate"), "final_track.blend"
        )
        self.mask_image_path = os.path.join(
            self.mask_full_map_dir, "merged_mask.png"
        )
        self.trees_mask_path = os.path.join(
            self.mask_full_map_dir, "trees_mask.png"
        )
        self.grass_mask_path = os.path.join(
            self.mask_full_map_dir, "grass_mask.png"
        )
        self.kerb_mask_path = os.path.join(
            self.mask_full_map_dir, "kerb_mask.png"
        )
        self.sand_mask_path = os.path.join(
            self.mask_full_map_dir, "sand_mask.png"
        )
        self.building_mask_path = os.path.join(
            self.mask_full_map_dir, "building_mask.png"
        )
        self.water_mask_path = os.path.join(
            self.mask_full_map_dir, "water_mask.png"
        )
        self.concrete_mask_path = os.path.join(
            self.mask_full_map_dir, "concrete_mask.png"
        )
        self.track_layouts_json = os.path.join(
            self.stage_dir("track_layouts"), "layouts.json"
        )
        self.manual_game_objects_json = os.path.join(
            self.stage_dir("manual_game_objects"), "game_objects.json"
        )

        return self
