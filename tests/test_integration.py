"""
Integration tests for the SAM3 track segmentation pipeline.

Tests the pipeline configuration, stage sequencing, CLI argument parsing,
module interface verification, and error handling.

Note: These tests do NOT require SAM3 model weights, Blender, or GPU.
They verify the orchestration logic, configuration, and module APIs.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from unittest import mock

import pytest

# Ensure the script/ directory is on the path for imports
_script_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "script")
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from sam3_track_gen import (
    BLENDER_STAGES,
    PIPELINE_STAGES,
    PipelineConfig,
    STAGE_FUNCTIONS,
    build_parser,
    config_from_args,
    run_pipeline,
)


# ---------------------------------------------------------------------------
# PipelineConfig tests
# ---------------------------------------------------------------------------

class TestPipelineConfig:
    """Tests for PipelineConfig dataclass and resolve()."""

    def test_default_values(self):
        config = PipelineConfig()
        assert config.geotiff_path == ""
        assert config.tiles_dir == ""
        assert config.output_dir == "output"
        assert config.blender_exe == ""
        assert config.gemini_model == "gemini-2.0-flash"
        assert config.track_direction == "clockwise"

    def test_resolve_creates_output_dir(self, tmp_path):
        out = str(tmp_path / "test_output")
        config = PipelineConfig(output_dir=out)
        config.resolve()
        assert os.path.isdir(out)

    def test_resolve_derives_paths(self, tmp_path):
        geotiff = str(tmp_path / "result.tif")
        tiles = str(tmp_path / "b3dm")
        out = str(tmp_path / "output")
        os.makedirs(tiles, exist_ok=True)
        # create a dummy geotiff file
        with open(geotiff, "w") as f:
            f.write("dummy")

        config = PipelineConfig(
            geotiff_path=geotiff,
            tiles_dir=tiles,
            output_dir=out,
        )
        config.resolve()

        # Verify derived paths
        assert config.geotiff_path == os.path.abspath(geotiff)
        assert config.tiles_dir == os.path.abspath(tiles)
        assert config.output_dir == os.path.abspath(out)
        assert config.clips_dir == os.path.join(os.path.dirname(os.path.abspath(geotiff)), "clips")
        assert config.blender_clips_dir == os.path.join(os.path.abspath(out), "blender_clips")
        assert config.glb_dir == os.path.join(os.path.abspath(out), "glb")
        assert config.blend_file.endswith("polygons.blend")
        assert config.walls_json.endswith("walls.json")
        assert config.walls_preview.endswith("walls_preview.png")
        assert config.game_objects_json.endswith("game_objects.json")
        assert config.game_objects_preview.endswith("game_objects_preview.png")
        assert config.mask_image_path.endswith("merged_mask.png")

    def test_resolve_without_geotiff(self, tmp_path):
        out = str(tmp_path / "output")
        config = PipelineConfig(output_dir=out)
        config.resolve()
        # clips_dir should remain empty when no geotiff
        assert config.clips_dir == ""
        assert config.mask_image_path == ""

    def test_resolve_without_tiles_dir(self, tmp_path):
        out = str(tmp_path / "output")
        config = PipelineConfig(output_dir=out)
        config.resolve()
        assert config.tiles_dir == ""


# ---------------------------------------------------------------------------
# CLI argument parsing tests
# ---------------------------------------------------------------------------

class TestCLI:
    """Tests for CLI argument parsing."""

    def test_build_parser_returns_parser(self):
        parser = build_parser()
        assert parser is not None

    def test_parse_minimal_args(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.geotiff == ""
        assert args.tiles_dir == ""
        assert args.output_dir == "output"
        assert args.stages is None

    def test_parse_full_args(self):
        parser = build_parser()
        args = parser.parse_args([
            "--geotiff", "test.tif",
            "--tiles-dir", "tiles/",
            "--output-dir", "out/",
            "--blender-exe", "/usr/bin/blender",
            "--gemini-api-key", "test-key",
            "--gemini-model", "gemini-pro",
            "--track-direction", "counterclockwise",
            "--track-description", "A test track",
        ])
        assert args.geotiff == "test.tif"
        assert args.tiles_dir == "tiles/"
        assert args.output_dir == "out/"
        assert args.blender_exe == "/usr/bin/blender"
        assert args.gemini_api_key == "test-key"
        assert args.gemini_model == "gemini-pro"
        assert args.track_direction == "counterclockwise"
        assert args.track_description == "A test track"

    def test_parse_multiple_stages(self):
        parser = build_parser()
        args = parser.parse_args([
            "--stage", "b3dm_convert",
            "--stage", "ai_walls",
        ])
        assert args.stages == ["b3dm_convert", "ai_walls"]

    def test_config_from_args(self, tmp_path):
        parser = build_parser()
        out = str(tmp_path / "output")
        args = parser.parse_args([
            "--output-dir", out,
            "--track-direction", "counterclockwise",
        ])
        config = config_from_args(args)
        assert config.track_direction == "counterclockwise"
        assert os.path.isdir(config.output_dir)

    def test_config_from_args_gemini_key(self, tmp_path):
        parser = build_parser()
        out = str(tmp_path / "output")
        args = parser.parse_args([
            "--output-dir", out,
            "--gemini-api-key", "custom-key",
        ])
        config = config_from_args(args)
        assert config.gemini_api_key == "custom-key"

    def test_config_from_args_default_gemini_key(self, tmp_path):
        parser = build_parser()
        out = str(tmp_path / "output")
        args = parser.parse_args(["--output-dir", out])
        config = config_from_args(args)
        # Should use the built-in default key
        assert config.gemini_api_key != ""


# ---------------------------------------------------------------------------
# Pipeline stage registration tests
# ---------------------------------------------------------------------------

class TestPipelineStages:
    """Tests for pipeline stage registration and ordering."""

    def test_all_stages_have_functions(self):
        for stage in PIPELINE_STAGES:
            assert stage in STAGE_FUNCTIONS, f"Stage '{stage}' missing from STAGE_FUNCTIONS"

    def test_stage_functions_are_callable(self):
        for name, func in STAGE_FUNCTIONS.items():
            assert callable(func), f"STAGE_FUNCTIONS['{name}'] is not callable"

    def test_pipeline_stages_ordered(self):
        """Verify that stage order is logical."""
        stage_order = {name: i for i, name in enumerate(PIPELINE_STAGES)}
        # b3dm_convert should come before convert_to_blender
        assert stage_order["b3dm_convert"] < stage_order["convert_to_blender"]
        # mask stages come before blender conversion
        assert stage_order["mask_full_map"] < stage_order["convert_to_blender"]
        # blender polygons comes after conversion
        assert stage_order["convert_to_blender"] < stage_order["blender_polygons"]
        # AI stages are after mask stages
        assert stage_order["mask_on_clips"] < stage_order["ai_walls"]

    def test_blender_stages_defined(self):
        assert len(BLENDER_STAGES) > 0
        for stage in BLENDER_STAGES:
            assert isinstance(stage, str)


# ---------------------------------------------------------------------------
# Pipeline runner tests
# ---------------------------------------------------------------------------

class TestRunPipeline:
    """Tests for the run_pipeline orchestration."""

    def test_unknown_stage_raises(self, tmp_path):
        config = PipelineConfig(output_dir=str(tmp_path / "out"))
        config.resolve()
        with pytest.raises(ValueError, match="Unknown stage"):
            run_pipeline(config, stages=["nonexistent_stage"])

    def test_stage_failure_raises_runtime_error(self, tmp_path):
        """When a stage raises, run_pipeline wraps it in RuntimeError."""
        config = PipelineConfig(output_dir=str(tmp_path / "out"))
        config.resolve()
        # b3dm_convert with empty tiles_dir will just skip (warning), so use mask_full_map
        # which requires geotiff_path
        with pytest.raises(RuntimeError, match="Pipeline failed at stage"):
            run_pipeline(config, stages=["mask_full_map"])

    def test_run_b3dm_convert_no_tiles_dir(self, tmp_path):
        """b3dm_convert with no tiles_dir should skip gracefully."""
        config = PipelineConfig(output_dir=str(tmp_path / "out"))
        config.resolve()
        # Should not raise
        run_pipeline(config, stages=["b3dm_convert"])

    def test_run_blender_polygons_no_blender_exe(self, tmp_path):
        """blender_polygons without blender_exe should raise."""
        config = PipelineConfig(output_dir=str(tmp_path / "out"))
        config.resolve()
        with pytest.raises(RuntimeError, match="blender_exe is required"):
            run_pipeline(config, stages=["blender_polygons"])

    def test_run_convert_to_blender_no_geotiff(self, tmp_path):
        config = PipelineConfig(output_dir=str(tmp_path / "out"))
        config.resolve()
        with pytest.raises(RuntimeError, match="geotiff_path is required"):
            run_pipeline(config, stages=["convert_to_blender"])


# ---------------------------------------------------------------------------
# Module API verification tests
# ---------------------------------------------------------------------------

class TestModuleAPIs:
    """Verify that all pipeline modules are importable and have expected APIs."""

    def test_b3dm_converter_api(self):
        from b3dm_converter import convert_file, convert_directory, B3dmConversionError
        assert callable(convert_file)
        assert callable(convert_directory)
        assert issubclass(B3dmConversionError, Exception)

    def test_geo_sam3_blender_utils_api(self):
        from geo_sam3_blender_utils import map_mask_to_blender, consolidate_clips_by_tag
        assert callable(map_mask_to_blender)
        assert callable(consolidate_clips_by_tag)

    def test_surface_extraction_api(self):
        from surface_extraction import (
            generate_collision_name,
            generate_sampling_grid,
            triangulate_points,
            load_clip_polygons,
            extract_polygon_xz,
            MATERIAL_PREFIXES,
        )
        assert callable(generate_collision_name)
        assert callable(generate_sampling_grid)
        assert callable(triangulate_points)
        assert callable(load_clip_polygons)
        assert callable(extract_polygon_xz)
        assert isinstance(MATERIAL_PREFIXES, dict)
        assert "road" in MATERIAL_PREFIXES

    def test_ai_wall_generator_api(self):
        from ai_wall_generator import (
            generate_walls,
            validate_walls_json,
            prepare_image_for_llm,
            generate_wall_prompt,
        )
        assert callable(generate_walls)
        assert callable(validate_walls_json)
        assert callable(prepare_image_for_llm)
        assert callable(generate_wall_prompt)

    def test_ai_game_objects_api(self):
        from ai_game_objects import (
            generate_game_objects,
            validate_game_objects_json,
            generate_game_objects_prompt,
        )
        assert callable(generate_game_objects)
        assert callable(validate_game_objects_json)
        assert callable(generate_game_objects_prompt)

    def test_ai_visualizer_api(self):
        from ai_visualizer import visualize_walls, visualize_game_objects
        assert callable(visualize_walls)
        assert callable(visualize_game_objects)

    def test_gemini_client_api(self):
        from gemini_client import GeminiClient
        assert callable(GeminiClient)


# ---------------------------------------------------------------------------
# Stage input validation tests
# ---------------------------------------------------------------------------

class TestStageValidation:
    """Tests for individual stage input validation."""

    def test_mask_full_map_requires_geotiff(self, tmp_path):
        from sam3_track_gen import stage_mask_full_map
        config = PipelineConfig(output_dir=str(tmp_path / "out"))
        config.resolve()
        with pytest.raises(ValueError, match="geotiff_path is required"):
            stage_mask_full_map(config)

    def test_clip_full_map_requires_geotiff(self, tmp_path):
        from sam3_track_gen import stage_clip_full_map
        config = PipelineConfig(output_dir=str(tmp_path / "out"))
        config.resolve()
        with pytest.raises(ValueError, match="geotiff_path is required"):
            stage_clip_full_map(config)

    def test_mask_on_clips_requires_clips_dir(self, tmp_path):
        from sam3_track_gen import stage_mask_on_clips
        config = PipelineConfig(output_dir=str(tmp_path / "out"))
        config.resolve()
        with pytest.raises(ValueError, match="clips_dir not found"):
            stage_mask_on_clips(config)

    def test_convert_to_blender_requires_geotiff(self, tmp_path):
        from sam3_track_gen import stage_convert_to_blender
        config = PipelineConfig(output_dir=str(tmp_path / "out"))
        config.resolve()
        with pytest.raises(ValueError, match="geotiff_path is required"):
            stage_convert_to_blender(config)

    def test_convert_to_blender_requires_tiles_dir(self, tmp_path):
        from sam3_track_gen import stage_convert_to_blender
        config = PipelineConfig(
            geotiff_path=str(tmp_path / "result.tif"),
            output_dir=str(tmp_path / "out"),
        )
        config.resolve()
        with pytest.raises(ValueError, match="tiles_dir is required"):
            stage_convert_to_blender(config)

    def test_blender_polygons_requires_blender_exe(self, tmp_path):
        from sam3_track_gen import stage_blender_polygons
        config = PipelineConfig(output_dir=str(tmp_path / "out"))
        config.resolve()
        with pytest.raises(ValueError, match="blender_exe is required"):
            stage_blender_polygons(config)

    def test_ai_walls_requires_geotiff(self, tmp_path):
        from sam3_track_gen import stage_ai_walls
        config = PipelineConfig(output_dir=str(tmp_path / "out"))
        config.resolve()
        with pytest.raises(ValueError, match="geotiff_path is required"):
            stage_ai_walls(config)

    def test_ai_game_objects_requires_geotiff(self, tmp_path):
        from sam3_track_gen import stage_ai_game_objects
        config = PipelineConfig(output_dir=str(tmp_path / "out"))
        config.resolve()
        with pytest.raises(ValueError, match="geotiff_path is required"):
            stage_ai_game_objects(config)


# ---------------------------------------------------------------------------
# B3DM conversion integration test
# ---------------------------------------------------------------------------

class TestB3DMConvertStage:
    """Integration test for the B3DM conversion stage with real test data."""

    def test_b3dm_convert_with_test_data(self, tmp_path):
        """Test B3DM conversion if test data exists."""
        test_tiles = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "test_images_shajing", "b3dm",
        )
        if not os.path.isdir(test_tiles):
            pytest.skip("Test data not available")

        out = str(tmp_path / "out")
        config = PipelineConfig(tiles_dir=test_tiles, output_dir=out)
        config.resolve()
        run_pipeline(config, stages=["b3dm_convert"])
        # Verify at least some GLB files were created
        assert os.path.isdir(config.glb_dir)


# ---------------------------------------------------------------------------
# Surface extraction functional test
# ---------------------------------------------------------------------------

class TestSurfaceExtractionFunctional:
    """Test surface_extraction module functions with concrete data."""

    def test_collision_naming(self):
        from surface_extraction import generate_collision_name
        assert generate_collision_name("road", 0) == "1ROAD_0"
        assert generate_collision_name("grass", 3) == "1GRASS_3"
        assert generate_collision_name("WALL", 1) == "1WALL_1"
        assert generate_collision_name("  sand  ", 5) == "1SAND_5"

    def test_collision_naming_invalid(self):
        from surface_extraction import generate_collision_name
        with pytest.raises(ValueError, match="Unknown material type"):
            generate_collision_name("water", 0)

    def test_sampling_grid(self):
        from surface_extraction import generate_sampling_grid
        polygon = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
        points, boundary = generate_sampling_grid(polygon, density=5.0)
        assert len(points) > 4  # boundary + at least some interior
        assert len(boundary) == 4

    def test_triangulate_points(self):
        from surface_extraction import triangulate_points
        pts_3d = [(0, 0, 0), (10, 0, 0), (10, 0, 10), (0, 0, 10)]
        faces = triangulate_points(pts_3d)
        assert len(faces) >= 2  # should form at least 2 triangles from 4 pts


# ---------------------------------------------------------------------------
# Wall/game object validation functional tests
# ---------------------------------------------------------------------------

class TestAIValidation:
    """Test the validation functions from AI modules."""

    def test_validate_walls_valid(self):
        from ai_wall_generator import validate_walls_json
        data = {
            "walls": [
                {
                    "type": "outer",
                    "points": [[0, 0], [100, 0], [100, 100]],
                    "closed": True,
                },
            ]
        }
        errors = validate_walls_json(data)
        assert errors == []

    def test_validate_walls_invalid_no_outer(self):
        from ai_wall_generator import validate_walls_json
        data = {
            "walls": [
                {
                    "type": "inner",
                    "points": [[0, 0], [100, 0], [100, 100]],
                    "closed": True,
                },
            ]
        }
        errors = validate_walls_json(data)
        assert len(errors) > 0

    def test_validate_game_objects_valid(self):
        from ai_game_objects import validate_game_objects_json
        data = {
            "track_direction": "clockwise",
            "objects": [
                {
                    "name": "AC_HOTLAP_START_0",
                    "position": [100, 200],
                    "orientation_z": [1.0, 0.0],
                    "type": "hotlap_start",
                },
            ],
        }
        errors = validate_game_objects_json(data)
        assert errors == []

    def test_validate_game_objects_missing_hotlap(self):
        from ai_game_objects import validate_game_objects_json
        data = {
            "track_direction": "clockwise",
            "objects": [
                {
                    "name": "AC_PIT_0",
                    "position": [100, 200],
                    "orientation_z": [1.0, 0.0],
                    "type": "pit",
                },
            ],
        }
        errors = validate_game_objects_json(data)
        assert any("hotlap_start" in e for e in errors)


# ---------------------------------------------------------------------------
# Consolidation functional test
# ---------------------------------------------------------------------------

class TestConsolidation:
    """Test clip consolidation with synthetic data."""

    def test_consolidate_clips_by_tag(self, tmp_path):
        from geo_sam3_blender_utils import consolidate_clips_by_tag

        clip_dir = str(tmp_path / "clips")
        os.makedirs(clip_dir)

        # Create two blender JSON files with different tags
        for i, tag in enumerate(["road", "grass"]):
            data = {
                "origin": {"ecef": [0, 0, 0]},
                "frame": {"mode": "enu"},
                "polygons": {
                    "include": [
                        {
                            "tag": tag,
                            "mask_index": 0,
                            "poly_index": 0,
                            "points_xyz": [[1, 0, 2], [3, 0, 4], [5, 0, 6]],
                            "prob": 0.9,
                        }
                    ],
                    "exclude": [],
                },
            }
            with open(os.path.join(clip_dir, f"clip_{i}_blender.json"), "w") as f:
                json.dump(data, f)

        result = consolidate_clips_by_tag(clip_dir)
        assert "road" in result
        assert "grass" in result
        # Verify the consolidated files exist
        assert os.path.isfile(result["road"])
        assert os.path.isfile(result["grass"])

        # Verify content of road consolidated file
        with open(result["road"], "r") as f:
            road_data = json.load(f)
        assert road_data["source_tag"] == "road"
        assert len(road_data["polygons"]["include"]) == 1
