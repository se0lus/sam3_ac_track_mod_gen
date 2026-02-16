"""Tests for the B3DM to GLB converter module."""

import os
import sys
import struct
import tempfile
import unittest

# Add script directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'script'))

from b3dm_converter import convert_file, convert_directory, B3dmConversionError

# Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
B3DM_DIR = os.path.join(PROJECT_ROOT, 'test_images_shajing', 'b3dm')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output', 'test_b3dm_converter')


def _make_fake_b3dm(glb_payload=b'glTF\x02\x00\x00\x00\x0c\x00\x00\x00',
                    ft_json=b'', ft_bin=b'', bt_json=b'', bt_bin=b''):
    """Build a minimal valid B3DM byte sequence for testing."""
    total = 28 + len(ft_json) + len(ft_bin) + len(bt_json) + len(bt_bin) + len(glb_payload)
    header = struct.pack('<4s6I',
                         b'b3dm', 1, total,
                         len(ft_json), len(ft_bin),
                         len(bt_json), len(bt_bin))
    return header + ft_json + ft_bin + bt_json + bt_bin + glb_payload


class TestConvertFile(unittest.TestCase):
    """Tests for convert_file()."""

    def setUp(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    def test_convert_real_b3dm_file(self):
        """Convert a real B3DM file from test data and verify GLB output."""
        b3dm_path = os.path.join(B3DM_DIR, 'BlockBXAYX', 'BlockBXAYX_L24_1.b3dm')
        if not os.path.isfile(b3dm_path):
            self.skipTest(f"Test data not found: {b3dm_path}")

        glb_path = os.path.join(OUTPUT_DIR, 'BlockBXAYX_L24_1.glb')
        result = convert_file(b3dm_path, glb_path)

        self.assertEqual(result, glb_path)
        self.assertTrue(os.path.isfile(glb_path))

        # Verify GLB magic bytes
        with open(glb_path, 'rb') as f:
            magic = f.read(4)
        self.assertEqual(magic, b'glTF', "Output file should start with glTF magic")

        # Verify GLB has reasonable size
        glb_size = os.path.getsize(glb_path)
        self.assertGreater(glb_size, 12, "GLB file should be larger than minimum header")

    def test_convert_multiple_real_files(self):
        """Convert several real B3DM files to verify consistency."""
        test_files = [
            os.path.join(B3DM_DIR, 'BlockBXAYX', 'BlockBXAYX_L24_1.b3dm'),
            os.path.join(B3DM_DIR, 'BlockBXAYX', 'BlockBXAYX_L25_1.b3dm'),
            os.path.join(B3DM_DIR, 'BlockBXAYX', 'BlockBXAYX_L26_1.b3dm'),
        ]

        for b3dm_path in test_files:
            if not os.path.isfile(b3dm_path):
                continue

            name = os.path.basename(b3dm_path).replace('.b3dm', '.glb')
            glb_path = os.path.join(OUTPUT_DIR, name)
            convert_file(b3dm_path, glb_path)

            with open(glb_path, 'rb') as f:
                magic = f.read(4)
            self.assertEqual(magic, b'glTF', f"Failed for {name}")

    def test_convert_synthetic_b3dm(self):
        """Convert a synthetic B3DM with known payload."""
        glb_payload = b'glTF\x02\x00\x00\x00\x0c\x00\x00\x00'
        b3dm_data = _make_fake_b3dm(glb_payload=glb_payload)

        with tempfile.NamedTemporaryFile(suffix='.b3dm', delete=False) as tmp:
            tmp.write(b3dm_data)
            tmp_path = tmp.name

        try:
            glb_path = os.path.join(OUTPUT_DIR, 'synthetic.glb')
            convert_file(tmp_path, glb_path)

            with open(glb_path, 'rb') as f:
                data = f.read()
            self.assertEqual(data, glb_payload)
        finally:
            os.unlink(tmp_path)

    def test_convert_b3dm_with_tables(self):
        """Convert a B3DM that has non-empty feature and batch tables."""
        ft_json = b'{"BATCH_LENGTH":0}    '  # padded to align
        bt_json = b'{}                    '
        glb_payload = b'glTF\x02\x00\x00\x00\x0c\x00\x00\x00'
        b3dm_data = _make_fake_b3dm(
            glb_payload=glb_payload,
            ft_json=ft_json,
            bt_json=bt_json,
        )

        with tempfile.NamedTemporaryFile(suffix='.b3dm', delete=False) as tmp:
            tmp.write(b3dm_data)
            tmp_path = tmp.name

        try:
            glb_path = os.path.join(OUTPUT_DIR, 'with_tables.glb')
            convert_file(tmp_path, glb_path)

            with open(glb_path, 'rb') as f:
                data = f.read()
            self.assertEqual(data, glb_payload)
        finally:
            os.unlink(tmp_path)

    def test_missing_file_raises_error(self):
        """FileNotFoundError raised for non-existent input."""
        with self.assertRaises(FileNotFoundError):
            convert_file('/nonexistent/path/file.b3dm', os.path.join(OUTPUT_DIR, 'out.glb'))

    def test_invalid_magic_raises_error(self):
        """B3dmConversionError raised for files with wrong magic bytes."""
        bad_data = b'XXXX' + b'\x00' * 24 + b'glTF\x02\x00\x00\x00\x0c\x00\x00\x00'

        with tempfile.NamedTemporaryFile(suffix='.b3dm', delete=False) as tmp:
            tmp.write(bad_data)
            tmp_path = tmp.name

        try:
            with self.assertRaises(B3dmConversionError):
                convert_file(tmp_path, os.path.join(OUTPUT_DIR, 'bad.glb'))
        finally:
            os.unlink(tmp_path)

    def test_truncated_file_raises_error(self):
        """B3dmConversionError raised for files smaller than header."""
        with tempfile.NamedTemporaryFile(suffix='.b3dm', delete=False) as tmp:
            tmp.write(b'b3dm\x01\x00')
            tmp_path = tmp.name

        try:
            with self.assertRaises(B3dmConversionError):
                convert_file(tmp_path, os.path.join(OUTPUT_DIR, 'truncated.glb'))
        finally:
            os.unlink(tmp_path)

    def test_no_glb_payload_raises_error(self):
        """B3dmConversionError raised when GLB offset exceeds file size."""
        # Feature table JSON length claims 9999 bytes but file is only 28 bytes
        header = struct.pack('<4s6I', b'b3dm', 1, 28, 9999, 0, 0, 0)

        with tempfile.NamedTemporaryFile(suffix='.b3dm', delete=False) as tmp:
            tmp.write(header)
            tmp_path = tmp.name

        try:
            with self.assertRaises(B3dmConversionError):
                convert_file(tmp_path, os.path.join(OUTPUT_DIR, 'empty.glb'))
        finally:
            os.unlink(tmp_path)


class TestConvertDirectory(unittest.TestCase):
    """Tests for convert_directory()."""

    def test_convert_real_directory(self):
        """Batch convert a subdirectory of real B3DM files."""
        subdir = os.path.join(B3DM_DIR, 'BlockBXAYX')
        if not os.path.isdir(subdir):
            self.skipTest(f"Test data not found: {subdir}")

        out_dir = os.path.join(OUTPUT_DIR, 'batch_BlockBXAYX')
        converted = convert_directory(subdir, out_dir)

        self.assertGreater(len(converted), 0, "Should convert at least one file")

        # Verify all outputs are valid GLB
        for _src, glb_path in converted:
            with open(glb_path, 'rb') as f:
                magic = f.read(4)
            self.assertEqual(magic, b'glTF', f"Invalid GLB: {glb_path}")

    def test_convert_nested_directory(self):
        """Batch convert preserves nested subdirectory structure."""
        if not os.path.isdir(B3DM_DIR):
            self.skipTest(f"Test data not found: {B3DM_DIR}")

        out_dir = os.path.join(OUTPUT_DIR, 'batch_nested')
        converted = convert_directory(B3DM_DIR, out_dir)

        self.assertGreater(len(converted), 0)

        # Check that subdirectory structure is preserved
        for _src, glb_path in converted:
            self.assertTrue(glb_path.startswith(out_dir))
            self.assertTrue(os.path.isfile(glb_path))

    def test_missing_directory_raises_error(self):
        """FileNotFoundError raised for non-existent input directory."""
        with self.assertRaises(FileNotFoundError):
            convert_directory('/nonexistent/dir', os.path.join(OUTPUT_DIR, 'nowhere'))

    def test_empty_directory(self):
        """Empty directory returns empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = convert_directory(tmpdir, os.path.join(OUTPUT_DIR, 'empty_dir'))
            self.assertEqual(result, [])

    def test_directory_with_mixed_files(self):
        """Only .b3dm files are converted, other files are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a valid b3dm
            b3dm_data = _make_fake_b3dm()
            with open(os.path.join(tmpdir, 'test.b3dm'), 'wb') as f:
                f.write(b3dm_data)

            # Create a non-b3dm file
            with open(os.path.join(tmpdir, 'readme.txt'), 'w') as f:
                f.write('not a b3dm')

            out_dir = os.path.join(OUTPUT_DIR, 'mixed')
            converted = convert_directory(tmpdir, out_dir)

            self.assertEqual(len(converted), 1)
            self.assertTrue(converted[0][1].endswith('.glb'))


if __name__ == '__main__':
    unittest.main()
