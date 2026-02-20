"""
Unified pipeline configuration for SAM3 Track Segmentation.

Single source of truth for ALL config.  Replaces both:
- PipelineConfig in sam3_track_gen.py
- blender_scripts/config.py hardcoded values
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("sam3_pipeline.config")


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
    "ai_walls": 6,
    "ai_game_objects": 7,
    "blender_polygons": 8,
    "blender_automate": 9,
    "model_export": 10,
}

PIPELINE_STAGES: List[str] = [
    "b3dm_convert",
    "mask_full_map",
    "clip_full_map",
    "mask_on_clips",
    "convert_to_blender",
    "ai_walls",
    "ai_game_objects",
    "blender_polygons",
    "blender_automate",
    "model_export",
]

# ---------------------------------------------------------------------------
# Manual stage pairs: auto_stage → manual_stage
# ---------------------------------------------------------------------------
MANUAL_STAGE_PAIRS: Dict[str, str] = {
    "mask_full_map": "track_layouts",
    "convert_to_blender": "manual_surface_masks",
    "ai_walls": "manual_walls",
    "ai_game_objects": "manual_game_objects",
    "blender_automate": "manual_blender",  # 09 → 09a
}


# ---------------------------------------------------------------------------
# Junction utilities (Windows mklink /J, POSIX symlink)
# ---------------------------------------------------------------------------
def _is_junction(path: str) -> bool:
    """Check if *path* is a directory junction (Windows) or symlink (POSIX)."""
    if not os.path.lexists(path):
        return False
    if sys.platform == "win32":
        import ctypes
        attrs = ctypes.windll.kernel32.GetFileAttributesW(str(path))
        if attrs == -1:
            return False
        FILE_ATTRIBUTE_REPARSE_POINT = 0x400
        return bool(attrs & FILE_ATTRIBUTE_REPARSE_POINT)
    return os.path.islink(path)


def _create_junction(link: str, target: str) -> None:
    """Create a directory junction (Windows) or symlink (POSIX)."""
    target = os.path.abspath(target)
    link = os.path.abspath(link)
    if sys.platform == "win32":
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", link, target],
            check=True, capture_output=True,
        )
    else:
        os.symlink(target, link, target_is_directory=True)


def _remove_junction_safe(path: str) -> None:
    """Remove a junction/symlink without deleting the target contents.

    IMPORTANT: shutil.rmtree() would follow the junction and delete
    the real directory contents.  Use os.rmdir() for junctions.
    """
    if not os.path.lexists(path):
        return
    if _is_junction(path):
        if sys.platform == "win32":
            # os.rmdir works for junctions on Windows
            os.rmdir(path)
        else:
            os.unlink(path)
    # If it's a real directory, leave it alone


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
    inpaint_model: str = "gemini-2.5-flash-image"
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
    surface_density_road: float = 0.1
    surface_density_grass: float = 2.0
    surface_density_kerb: float = 0.1
    surface_density_sand: float = 2.0
    surface_density_road2: float = 2.0
    surface_density_default: float = 1.0

    # --- Surface extraction edge simplification (metres, 0 = no simplification) ---
    surface_edge_simplify: float = 0.0

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

    # --- Stage 8 options ---
    s8_generate_curves: bool = False  # Generate diagnostic 2D curves (slow, debug only)
    s8_gap_fill_enabled: bool = True  # Auto-fill mask gaps within driveable zone
    s8_gap_fill_threshold_m: float = 0.20  # Small gap threshold in metres
    s8_gap_fill_default_tag: str = "road2"  # Default fill tag for remaining voids

    # --- Stage 9 options ---
    s9_no_walls: bool = False
    s9_no_game_objects: bool = False
    s9_no_surfaces: bool = False
    s9_no_textures: bool = False
    s9_no_background: bool = False
    s9_refine_tags: List[str] = field(default_factory=lambda: ["road"])
    s9_mesh_simplify: bool = False          # Enable post-processing simplification for terrain meshes
    s9_mesh_weld_distance: float = 0.01     # Weld distance in metres (default 0.01)
    s9_mesh_decimate_ratio: float = 0.5     # Decimate ratio 0-1 (default 0.5)

    # --- Stage 10 options ---
    s10_max_vertices: int = 21000           # Max vertices per MESH object
    s10_max_batch_mb: int = 100             # Max FBX file size (MB, geometry only)
    s10_fbx_scale: float = 0.01            # FBX export global scale
    s10_ks_ambient: float = 0.5            # ksAmbient FLOAT1 for visible materials
    s10_ks_diffuse: float = 0.1            # ksDiffuse FLOAT1 for visible materials
    s10_ks_emissive: float = 0.1           # ksEmissive FLOAT3 (same value x3)
    s10_kseditor_exe: str = ""             # Path to ksEditorAT.exe (auto-detected if empty)

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
    manual_surface_masks_dir: str = ""  # manual-edited surface masks (stage 5a)
    manual_walls_json: str = ""  # manual-edited walls (stage 6a)
    manual_game_objects_json: str = ""  # manual-edited game objects (stage 7a)

    def stage_dir(self, stage_name: str) -> str:
        """Return ``output/NN_stage_name/`` for a given stage."""
        if stage_name == "track_layouts":
            return os.path.join(self.output_dir, "02a_track_layouts")
        if stage_name == "manual_surface_masks":
            return os.path.join(self.output_dir, "05a_manual_surface_masks")
        if stage_name == "manual_walls":
            return os.path.join(self.output_dir, "06a_manual_walls")
        if stage_name == "manual_game_objects":
            return os.path.join(self.output_dir, "07a_manual_game_objects")
        if stage_name == "manual_blender":
            return os.path.join(self.output_dir, "09a_manual_blender")
        idx = STAGE_ORDER.get(stage_name, 0)
        return os.path.join(self.output_dir, f"{idx:02d}_{stage_name}")

    def result_dir(self, auto_stage: str) -> str:
        """Return ``output/NN_result/`` — the junction target for downstream readers."""
        idx = STAGE_ORDER.get(auto_stage, 0)
        return os.path.join(self.output_dir, f"{idx:02d}_result")

    def setup_result_junctions(self, manual_stages: Dict[str, bool] | None = None) -> None:
        """Create NN_result/ junctions for each auto/manual stage pair.

        For each pair in MANUAL_STAGE_PAIRS:
        - If the manual stage is enabled AND its directory has data,
          junction points to the manual directory.
        - Otherwise, junction points to the auto directory.
        """
        if manual_stages is None:
            manual_stages = {}

        for auto_name, manual_name in MANUAL_STAGE_PAIRS.items():
            result_path = self.result_dir(auto_name)
            auto_dir = self.stage_dir(auto_name)
            manual_dir = self.stage_dir(manual_name)

            # Determine target
            manual_enabled = manual_stages.get(manual_name, False)
            manual_has_data = os.path.isdir(manual_dir) and bool(os.listdir(manual_dir))
            use_manual = manual_enabled and manual_has_data

            target = manual_dir if use_manual else auto_dir

            # Skip if target doesn't exist
            if not os.path.isdir(target):
                logger.debug("Junction target not found: %s, skipping %s", target, result_path)
                continue

            # Remove existing junction if it points to a different target
            if _is_junction(result_path):
                current = os.path.realpath(result_path)
                if os.path.normcase(os.path.normpath(current)) == os.path.normcase(os.path.normpath(target)):
                    continue  # already correct
                _remove_junction_safe(result_path)

            # Don't clobber a real directory
            if os.path.isdir(result_path) and not _is_junction(result_path):
                logger.warning("Real directory exists at %s, skipping junction", result_path)
                continue

            _create_junction(result_path, target)
            label = "manual" if use_manual else "auto"
            logger.info("Junction %s -> %s (%s)", result_path, target, label)

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
        self.manual_surface_masks_dir = self.stage_dir("manual_surface_masks")
        self.track_layouts_json = os.path.join(
            self.stage_dir("track_layouts"), "layouts.json"
        )
        self.manual_walls_json = os.path.join(
            self.stage_dir("manual_walls"), "walls.json"
        )
        self.manual_game_objects_json = os.path.join(
            self.stage_dir("manual_game_objects"), "game_objects.json"
        )

        # Result directories (junction targets for downstream stages)
        self.mask_full_map_result = self.result_dir("mask_full_map")       # 02_result
        self.blender_clips_result = self.result_dir("convert_to_blender")  # 05_result
        self.walls_result_dir = self.result_dir("ai_walls")                # 06_result
        self.game_objects_result_dir = self.result_dir("ai_game_objects")   # 07_result
        self.manual_blend_file = os.path.join(
            self.stage_dir("manual_blender"), "final_track.blend"
        )
        self.blender_result_dir = self.result_dir("blender_automate")     # 09_result
        self.export_dir = self.stage_dir("model_export")                # 10_model_export

        return self
