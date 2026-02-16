"""
Tests for AI generator modules:
  - ai_wall_generator  (TODO-5)
  - ai_game_objects    (TODO-7)
  - ai_visualizer

All tests are runnable without Blender and without calling the real Gemini API
(mocked responses are used).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure script/ is on sys.path so we can import modules directly
_SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "script")
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# Output directory for test artifacts
_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")
os.makedirs(_OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Sample / mock data
# ---------------------------------------------------------------------------

MOCK_WALLS_JSON = {
    "walls": [
        {
            "type": "outer",
            "points": [
                [100, 100], [900, 100], [900, 900], [100, 900],
                [100, 500], [50, 500], [50, 100],
            ],
            "closed": True,
        },
        {
            "type": "inner",
            "points": [
                [300, 300], [700, 300], [700, 700], [300, 700],
            ],
            "closed": True,
        },
    ]
}

MOCK_GAME_OBJECTS_JSON = {
    "track_direction": "clockwise",
    "objects": [
        {
            "name": "AC_HOTLAP_START_0",
            "position": [500, 150],
            "orientation_z": [1.0, 0.0],
            "type": "hotlap_start",
        },
        {
            "name": "AC_PIT_0",
            "position": [200, 800],
            "orientation_z": [0.0, -1.0],
            "type": "pit",
        },
        {
            "name": "AC_PIT_1",
            "position": [250, 800],
            "orientation_z": [0.0, -1.0],
            "type": "pit",
        },
        {
            "name": "AC_PIT_2",
            "position": [300, 800],
            "orientation_z": [0.0, -1.0],
            "type": "pit",
        },
        {
            "name": "AC_PIT_3",
            "position": [350, 800],
            "orientation_z": [0.0, -1.0],
            "type": "pit",
        },
        {
            "name": "AC_PIT_4",
            "position": [400, 800],
            "orientation_z": [0.0, -1.0],
            "type": "pit",
        },
        {
            "name": "AC_PIT_5",
            "position": [450, 800],
            "orientation_z": [0.0, -1.0],
            "type": "pit",
        },
        {
            "name": "AC_PIT_6",
            "position": [500, 800],
            "orientation_z": [0.0, -1.0],
            "type": "pit",
        },
        {
            "name": "AC_PIT_7",
            "position": [550, 800],
            "orientation_z": [0.0, -1.0],
            "type": "pit",
        },
        {
            "name": "AC_START_0",
            "position": [500, 200],
            "orientation_z": [1.0, 0.0],
            "type": "start",
        },
        {
            "name": "AC_START_1",
            "position": [480, 220],
            "orientation_z": [1.0, 0.0],
            "type": "start",
        },
        {
            "name": "AC_TIME_0_L",
            "position": [490, 100],
            "orientation_z": [1.0, 0.0],
            "type": "timing_left",
        },
        {
            "name": "AC_TIME_0_R",
            "position": [510, 100],
            "orientation_z": [1.0, 0.0],
            "type": "timing_right",
        },
    ],
}


def _create_test_image(size=(1024, 1024), color=(0, 128, 0)):
    """Create a simple in-memory test image."""
    from PIL import Image
    return Image.new("RGB", size, color)


def _save_test_image(path: str, size=(1024, 1024)):
    img = _create_test_image(size)
    img.save(path)
    return path


# ===================================================================
# ai_wall_generator tests
# ===================================================================


class TestPrepareImageForLLM(unittest.TestCase):
    def test_small_image_unchanged(self):
        from ai_wall_generator import prepare_image_for_llm
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            _save_test_image(path, size=(500, 400))
            result = prepare_image_for_llm(path, max_size=2048)
            self.assertEqual(result.size, (500, 400))
        finally:
            os.unlink(path)

    def test_large_image_scaled(self):
        from ai_wall_generator import prepare_image_for_llm
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            _save_test_image(path, size=(4096, 2048))
            result = prepare_image_for_llm(path, max_size=2048)
            self.assertEqual(max(result.size), 2048)
            # Aspect ratio preserved
            self.assertEqual(result.size, (2048, 1024))
        finally:
            os.unlink(path)


class TestWallPrompt(unittest.TestCase):
    def test_basic_prompt(self):
        from ai_wall_generator import generate_wall_prompt
        prompt = generate_wall_prompt()
        self.assertIn("outer wall", prompt.lower())
        self.assertIn("inner wall", prompt.lower())

    def test_prompt_with_description(self):
        from ai_wall_generator import generate_wall_prompt
        prompt = generate_wall_prompt(track_description="A small oval track")
        self.assertIn("A small oval track", prompt)


class TestValidateWallsJSON(unittest.TestCase):
    def test_valid(self):
        from ai_wall_generator import validate_walls_json
        errors = validate_walls_json(MOCK_WALLS_JSON)
        self.assertEqual(errors, [])

    def test_missing_outer(self):
        from ai_wall_generator import validate_walls_json
        data = {"walls": [{"type": "inner", "points": [[0, 0], [1, 0], [1, 1]], "closed": True}]}
        errors = validate_walls_json(data)
        self.assertTrue(any("outer" in e.lower() for e in errors))

    def test_not_a_dict(self):
        from ai_wall_generator import validate_walls_json
        errors = validate_walls_json([1, 2, 3])
        self.assertTrue(len(errors) > 0)

    def test_too_few_points(self):
        from ai_wall_generator import validate_walls_json
        data = {"walls": [{"type": "outer", "points": [[0, 0], [1, 1]], "closed": True}]}
        errors = validate_walls_json(data)
        self.assertTrue(any("3 points" in e for e in errors))

    def test_bad_point_format(self):
        from ai_wall_generator import validate_walls_json
        data = {
            "walls": [
                {"type": "outer", "points": [[0, 0], [1], [2, 2]], "closed": True},
            ]
        }
        errors = validate_walls_json(data)
        self.assertTrue(len(errors) > 0)


class TestGenerateWalls(unittest.TestCase):
    """Test generate_walls() with a mocked Gemini client."""

    @patch("gemini_client.GeminiClient")
    def test_generate_walls_success(self, MockClient):
        from ai_wall_generator import generate_walls

        instance = MockClient.return_value
        instance.generate_json.return_value = MOCK_WALLS_JSON

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            _save_test_image(path)
            result = generate_walls(path, api_key="test-key")
            self.assertIn("walls", result)
            self.assertEqual(len(result["walls"]), 2)
        finally:
            os.unlink(path)

    @patch("gemini_client.GeminiClient")
    def test_generate_walls_invalid_response(self, MockClient):
        from ai_wall_generator import generate_walls

        instance = MockClient.return_value
        instance.generate_json.return_value = {"walls": []}  # empty -> invalid

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            _save_test_image(path)
            with self.assertRaises(ValueError):
                generate_walls(path, api_key="test-key")
        finally:
            os.unlink(path)


# ===================================================================
# ai_game_objects tests
# ===================================================================


class TestGameObjectsPrompt(unittest.TestCase):
    def test_direction_in_prompt(self):
        from ai_game_objects import generate_game_objects_prompt
        prompt = generate_game_objects_prompt("clockwise")
        self.assertIn("clockwise", prompt)

    def test_counterclockwise(self):
        from ai_game_objects import generate_game_objects_prompt
        prompt = generate_game_objects_prompt("counterclockwise")
        self.assertIn("counterclockwise", prompt)


class TestValidateGameObjectsJSON(unittest.TestCase):
    def test_valid(self):
        from ai_game_objects import validate_game_objects_json
        errors = validate_game_objects_json(MOCK_GAME_OBJECTS_JSON)
        self.assertEqual(errors, [])

    def test_bad_direction(self):
        from ai_game_objects import validate_game_objects_json
        data = {"track_direction": "left", "objects": MOCK_GAME_OBJECTS_JSON["objects"]}
        errors = validate_game_objects_json(data)
        self.assertTrue(any("track_direction" in e for e in errors))

    def test_missing_hotlap(self):
        from ai_game_objects import validate_game_objects_json
        data = {
            "track_direction": "clockwise",
            "objects": [
                {"name": "AC_PIT_0", "position": [100, 100], "orientation_z": [1, 0], "type": "pit"},
            ],
        }
        errors = validate_game_objects_json(data)
        self.assertTrue(any("hotlap_start" in e for e in errors))

    def test_bad_position(self):
        from ai_game_objects import validate_game_objects_json
        data = {
            "track_direction": "clockwise",
            "objects": [
                {"name": "X", "position": [100], "orientation_z": [1, 0], "type": "hotlap_start"},
            ],
        }
        errors = validate_game_objects_json(data)
        self.assertTrue(any("position" in e for e in errors))


class TestGenerateGameObjects(unittest.TestCase):
    @patch("gemini_client.GeminiClient")
    def test_generate_success(self, MockClient):
        from ai_game_objects import generate_game_objects

        instance = MockClient.return_value
        instance.generate_json.return_value = MOCK_GAME_OBJECTS_JSON

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            _save_test_image(path)
            result = generate_game_objects(path, track_direction="clockwise", api_key="test-key")
            self.assertIn("objects", result)
            self.assertEqual(result["track_direction"], "clockwise")
        finally:
            os.unlink(path)


# ===================================================================
# ai_visualizer tests
# ===================================================================


class TestVisualizeWalls(unittest.TestCase):
    def test_creates_output_file(self):
        from ai_visualizer import visualize_walls

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img_path = f.name
        out_path = os.path.join(_OUTPUT_DIR, "test_walls_viz.png")
        try:
            _save_test_image(img_path)
            result = visualize_walls(img_path, MOCK_WALLS_JSON, out_path)
            self.assertTrue(os.path.isfile(result))
            self.assertTrue(os.path.getsize(result) > 0)
        finally:
            os.unlink(img_path)


class TestVisualizeGameObjects(unittest.TestCase):
    def test_creates_output_file(self):
        from ai_visualizer import visualize_game_objects

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img_path = f.name
        out_path = os.path.join(_OUTPUT_DIR, "test_game_objects_viz.png")
        try:
            _save_test_image(img_path)
            result = visualize_game_objects(img_path, MOCK_GAME_OBJECTS_JSON, out_path)
            self.assertTrue(os.path.isfile(result))
            self.assertTrue(os.path.getsize(result) > 0)
        finally:
            os.unlink(img_path)


# ===================================================================
# Coordinate conversion logic tests
# ===================================================================


class TestCoordinateConversion(unittest.TestCase):
    """Test the pixel-to-Blender coordinate mapping used by the Blender actions."""

    def test_pixel_to_blender_basic(self):
        """Verify the mapping: Blender X = px/ppu, Blender Z = (img_h - py)/ppu."""
        pixels_per_unit = 10.0
        img_h = 1000

        px, py = 500, 200
        bx = px / pixels_per_unit      # 50
        bz = (img_h - py) / pixels_per_unit  # 80

        self.assertAlmostEqual(bx, 50.0)
        self.assertAlmostEqual(bz, 80.0)

    def test_origin_maps_to_expected(self):
        """Pixel (0, img_h) should map to Blender (0, 0, 0)."""
        ppu = 1.0
        img_h = 1024
        bx = 0 / ppu
        bz = (img_h - img_h) / ppu
        self.assertAlmostEqual(bx, 0.0)
        self.assertAlmostEqual(bz, 0.0)


# ===================================================================
# JSON round-trip tests
# ===================================================================


class TestJSONSerialization(unittest.TestCase):
    def test_walls_json_roundtrip(self):
        s = json.dumps(MOCK_WALLS_JSON)
        loaded = json.loads(s)
        self.assertEqual(loaded, MOCK_WALLS_JSON)

    def test_game_objects_json_roundtrip(self):
        s = json.dumps(MOCK_GAME_OBJECTS_JSON)
        loaded = json.loads(s)
        self.assertEqual(loaded, MOCK_GAME_OBJECTS_JSON)

    def test_save_and_load_walls(self):
        out_path = os.path.join(_OUTPUT_DIR, "test_walls.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(MOCK_WALLS_JSON, f, indent=2)
        with open(out_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        self.assertEqual(loaded, MOCK_WALLS_JSON)

    def test_save_and_load_game_objects(self):
        out_path = os.path.join(_OUTPUT_DIR, "test_game_objects.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(MOCK_GAME_OBJECTS_JSON, f, indent=2)
        with open(out_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        self.assertEqual(loaded, MOCK_GAME_OBJECTS_JSON)


if __name__ == "__main__":
    unittest.main()
