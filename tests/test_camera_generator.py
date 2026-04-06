"""Tests for the camera_generator module."""

import json
import math
import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

# Add script directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'script'))

from camera_generator import (
    generate_cameras_ini,
    compute_track_bounds_ac,
    _cumulative_lengths,
    _local_curvature,
    _load_centerline_ac,
)


class TestCumulativeLengths(unittest.TestCase):
    """Tests for _cumulative_lengths helper."""

    def test_straight_line(self):
        pts = [(0, 0, 0), (3, 0, 4), (6, 0, 8)]
        cum = _cumulative_lengths(pts)
        self.assertEqual(len(cum), 3)
        self.assertAlmostEqual(cum[0], 0.0)
        self.assertAlmostEqual(cum[1], 5.0)
        self.assertAlmostEqual(cum[2], 10.0)

    def test_single_point(self):
        cum = _cumulative_lengths([(1, 2, 3)])
        self.assertEqual(cum, [0.0])

    def test_xz_plane_only(self):
        """Y component should not affect arc length (uses XZ plane)."""
        pts = [(0, 0, 0), (10, 99, 0)]
        cum = _cumulative_lengths(pts)
        self.assertAlmostEqual(cum[1], 10.0)


class TestLocalCurvature(unittest.TestCase):
    """Tests for _local_curvature helper."""

    def test_straight_line_zero_curvature(self):
        pts = [(i, 0, 0) for i in range(20)]
        curv = _local_curvature(pts, 10, window=6)
        self.assertAlmostEqual(curv, 0.0, places=5)

    def test_curved_path_nonzero(self):
        """A circular arc should have nonzero curvature."""
        n = 100
        radius = 50.0
        pts = [
            (radius * math.cos(2 * math.pi * i / n),
             0,
             radius * math.sin(2 * math.pi * i / n))
            for i in range(n)
        ]
        curv = _local_curvature(pts, 25, window=10)
        self.assertGreater(curv, 0.0)

    def test_too_few_points(self):
        curv = _local_curvature([(0, 0, 0), (1, 0, 0)], 0)
        self.assertEqual(curv, 0.0)


class TestGenerateCamerasIni(unittest.TestCase):
    """Tests for generate_cameras_ini with mocked coordinate conversion."""

    def _make_temp_files(self, centerline_pts, geo_meta):
        """Create temporary centerline.json and geo_metadata.json."""
        tmpdir = tempfile.mkdtemp()
        cl_path = os.path.join(tmpdir, "centerline.json")
        gm_path = os.path.join(tmpdir, "geo_metadata.json")
        with open(cl_path, "w") as f:
            json.dump({"centerline": centerline_pts}, f)
        with open(gm_path, "w") as f:
            json.dump(geo_meta, f)
        return tmpdir, cl_path, gm_path

    def test_zero_cameras(self):
        result = generate_cameras_ini("dummy", "dummy", "dummy", num_cameras=0)
        self.assertIn("CAMERA_COUNT=0", result)

    @patch("camera_generator._load_centerline_ac")
    def test_basic_output_format(self, mock_load):
        """Output should have correct INI structure."""
        # Mock 20 AC-space points along a straight line
        mock_load.return_value = [(float(i * 10), 0.0, 0.0) for i in range(20)]
        result = generate_cameras_ini("cl.json", "gm.json", "tiles", num_cameras=3)

        self.assertIn("[HEADER]", result)
        self.assertIn("CAMERA_COUNT=3", result)
        self.assertIn("[CAMERA_0]", result)
        self.assertIn("[CAMERA_1]", result)
        self.assertIn("[CAMERA_2]", result)
        self.assertIn("POSITION=", result)
        self.assertIn("FORWARD=", result)
        self.assertIn("MIN_FOV=", result)
        self.assertIn("MAX_FOV=", result)

    @patch("camera_generator._load_centerline_ac")
    def test_custom_fov(self, mock_load):
        mock_load.return_value = [(float(i), 0.0, float(i)) for i in range(20)]
        result = generate_cameras_ini(
            "cl.json", "gm.json", "tiles",
            num_cameras=2,
            fov_range=(15.0, 45.0),
        )
        self.assertIn("MIN_FOV=15", result)
        self.assertIn("MAX_FOV=45", result)

    @patch("camera_generator._load_centerline_ac")
    def test_insufficient_points(self, mock_load):
        mock_load.return_value = [(0, 0, 0)]
        result = generate_cameras_ini("cl.json", "gm.json", "tiles", num_cameras=5)
        self.assertIn("CAMERA_COUNT=0", result)

    @patch("camera_generator._load_centerline_ac")
    def test_none_centerline(self, mock_load):
        mock_load.return_value = None
        result = generate_cameras_ini("cl.json", "gm.json", "tiles")
        self.assertIn("CAMERA_COUNT=0", result)

    @patch("camera_generator._load_centerline_ac")
    def test_cameras_have_coverage(self, mock_load):
        """Each camera should have in_point and out_point."""
        mock_load.return_value = [(float(i * 5), 0.0, float(i * 3)) for i in range(50)]
        result = generate_cameras_ini("cl.json", "gm.json", "tiles", num_cameras=4)
        self.assertIn("IN_POINT=0.0", result)
        self.assertIn("IN_POINT=0.25", result)
        self.assertIn("IN_POINT=0.5", result)
        self.assertIn("IN_POINT=0.75", result)


class TestComputeTrackBoundsAc(unittest.TestCase):

    @patch("camera_generator._load_centerline_ac")
    def test_bounds(self, mock_load):
        mock_load.return_value = [(-10, 0, -20), (30, 0, 40), (5, 0, 10)]
        bounds = compute_track_bounds_ac("cl.json", "gm.json", "tiles")
        self.assertIsNotNone(bounds)
        min_x, min_z, max_x, max_z = bounds
        self.assertAlmostEqual(min_x, -10)
        self.assertAlmostEqual(max_x, 30)
        self.assertAlmostEqual(min_z, -20)
        self.assertAlmostEqual(max_z, 40)

    @patch("camera_generator._load_centerline_ac")
    def test_bounds_none(self, mock_load):
        mock_load.return_value = None
        bounds = compute_track_bounds_ac("cl.json", "gm.json", "tiles")
        self.assertIsNone(bounds)


if __name__ == "__main__":
    unittest.main()
