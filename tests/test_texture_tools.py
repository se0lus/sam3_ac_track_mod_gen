"""
Tests for texture_tools pure-Python helpers.

These tests cover the non-Blender utility functions that can be tested
without the bpy module. Blender operator logic is tested via manual
Blender execution.
"""

import os
import sys
import types
import unittest

# Add the blender_scripts directory to sys.path so we can import the helpers
# without requiring bpy at module level.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
_BLENDER_SCRIPTS = os.path.join(_PROJECT_ROOT, "blender_scripts")
if _BLENDER_SCRIPTS not in sys.path:
    sys.path.insert(0, _BLENDER_SCRIPTS)

# We need a mock bpy module since texture_tools imports bpy at module level.
_mock_bpy = types.ModuleType("bpy")
_mock_bpy.types = types.ModuleType("bpy.types")


class _MockOperator:
    bl_idname = ""
    bl_label = ""
    bl_options = set()


_mock_bpy.types.Operator = _MockOperator
_mock_bpy.types.Context = type("Context", (), {})
sys.modules["bpy"] = _mock_bpy
sys.modules["bpy.types"] = _mock_bpy.types

from sam3_actions.texture_tools import (
    format_to_extension,
    is_jpeg_path,
    jpg_to_png_path,
    texture_output_path,
    try_import_pil,
)

OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


class TestFormatToExtension(unittest.TestCase):
    """Test format_to_extension mapping."""

    def test_png(self):
        self.assertEqual(format_to_extension("PNG"), ".png")

    def test_jpeg(self):
        self.assertEqual(format_to_extension("JPEG"), ".jpg")

    def test_bmp(self):
        self.assertEqual(format_to_extension("BMP"), ".bmp")

    def test_tiff(self):
        self.assertEqual(format_to_extension("TIFF"), ".tiff")

    def test_targa(self):
        self.assertEqual(format_to_extension("TARGA"), ".tga")

    def test_open_exr(self):
        self.assertEqual(format_to_extension("OPEN_EXR"), ".exr")

    def test_hdr(self):
        self.assertEqual(format_to_extension("HDR"), ".hdr")

    def test_case_insensitive(self):
        self.assertEqual(format_to_extension("png"), ".png")
        self.assertEqual(format_to_extension("jpeg"), ".jpg")

    def test_unknown_defaults_to_png(self):
        self.assertEqual(format_to_extension("UNKNOWN"), ".png")
        self.assertEqual(format_to_extension(""), ".png")


class TestIsJpegPath(unittest.TestCase):
    """Test JPEG file detection."""

    def test_jpg_extension(self):
        self.assertTrue(is_jpeg_path("texture/image_0.jpg"))

    def test_jpeg_extension(self):
        self.assertTrue(is_jpeg_path("texture/image_0.jpeg"))

    def test_uppercase_jpg(self):
        self.assertTrue(is_jpeg_path("texture/image_0.JPG"))

    def test_uppercase_jpeg(self):
        self.assertTrue(is_jpeg_path("texture/image_0.JPEG"))

    def test_png_is_not_jpeg(self):
        self.assertFalse(is_jpeg_path("texture/image_0.png"))

    def test_bmp_is_not_jpeg(self):
        self.assertFalse(is_jpeg_path("texture/image_0.bmp"))

    def test_no_extension(self):
        self.assertFalse(is_jpeg_path("texture/image_0"))

    def test_empty_string(self):
        self.assertFalse(is_jpeg_path(""))


class TestJpgToPngPath(unittest.TestCase):
    """Test JPG to PNG path conversion."""

    def test_jpg_to_png(self):
        self.assertEqual(jpg_to_png_path("/tex/img.jpg"), "/tex/img.png")

    def test_jpeg_to_png(self):
        self.assertEqual(jpg_to_png_path("/tex/img.jpeg"), "/tex/img.png")

    def test_uppercase_jpg_to_png(self):
        self.assertEqual(jpg_to_png_path("/tex/img.JPG"), "/tex/img.png")

    def test_png_unchanged(self):
        self.assertEqual(jpg_to_png_path("/tex/img.png"), "/tex/img.png")

    def test_bmp_unchanged(self):
        self.assertEqual(jpg_to_png_path("/tex/img.bmp"), "/tex/img.bmp")

    def test_no_extension_unchanged(self):
        self.assertEqual(jpg_to_png_path("/tex/img"), "/tex/img")


class TestTextureOutputPath(unittest.TestCase):
    """Test output path construction."""

    def test_png_format(self):
        result = texture_output_path("image_0", "PNG", "/blenddir/texture")
        self.assertEqual(result, os.path.join("/blenddir/texture", "image_0.png"))

    def test_jpeg_format(self):
        result = texture_output_path("image_1", "JPEG", "/blenddir/texture")
        self.assertEqual(result, os.path.join("/blenddir/texture", "image_1.jpg"))

    def test_unknown_format_defaults_to_png(self):
        result = texture_output_path("image_2", "WEIRD", "/blenddir/texture")
        self.assertEqual(result, os.path.join("/blenddir/texture", "image_2.png"))


class TestTryImportPil(unittest.TestCase):
    """Test PIL availability detection."""

    def test_returns_module_or_none(self):
        result = try_import_pil()
        # On machines with PIL installed, result should be the Image module.
        # On machines without PIL, result should be None.
        # Either way the function should not raise.
        if result is not None:
            self.assertTrue(hasattr(result, "open"))
        else:
            self.assertIsNone(result)


class TestPilConversionFallback(unittest.TestCase):
    """Test that PIL-based conversion works when PIL is available."""

    def test_pil_jpg_to_png_roundtrip(self):
        """If PIL is available, do a real conversion to output/."""
        PILImage = try_import_pil()
        if PILImage is None:
            self.skipTest("PIL not available")

        # Create a small test JPEG in output/
        jpg_path = os.path.join(OUTPUT_DIR, "test_texture_input.jpg")
        png_path = os.path.join(OUTPUT_DIR, "test_texture_input.png")

        # Create a 4x4 red image and save as JPEG.
        img = PILImage.new("RGB", (4, 4), color=(255, 0, 0))
        img.save(jpg_path, "JPEG")
        self.assertTrue(os.path.exists(jpg_path))

        # Convert to PNG.
        loaded = PILImage.open(jpg_path)
        loaded.save(png_path, "PNG")
        loaded.close()
        self.assertTrue(os.path.exists(png_path))

        # Verify the PNG can be opened and has the right size.
        verify = PILImage.open(png_path)
        self.assertEqual(verify.size, (4, 4))
        verify.close()

        # Cleanup -- use gc to release file handles on Windows
        import gc
        gc.collect()
        try:
            os.remove(jpg_path)
        except (PermissionError, FileNotFoundError):
            pass
        try:
            os.remove(png_path)
        except (PermissionError, FileNotFoundError):
            pass


if __name__ == "__main__":
    # Write results summary to output/
    result_path = os.path.join(OUTPUT_DIR, "test_texture_tools_results.txt")
    with open(result_path, "w") as f:
        runner = unittest.TextTestRunner(stream=f, verbosity=2)
        suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
        result = runner.run(suite)

    # Also run to stdout for CI.
    unittest.main(verbosity=2)
