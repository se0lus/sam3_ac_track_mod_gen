"""Tests for script/surface_extraction.py -- pure Python, no Blender needed."""

import json
import math
import os
import sys
import unittest

# Add script directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "script"))

from surface_extraction import (
    MATERIAL_PREFIXES,
    extract_polygon_xz,
    generate_collision_name,
    generate_sampling_grid,
    load_clip_polygons,
    triangulate_points,
    _point_in_polygon_2d,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "test_surface_extractor")


class TestCollisionNaming(unittest.TestCase):
    """Tests for generate_collision_name() -- TODO-6 naming convention."""

    def test_road_names(self):
        self.assertEqual(generate_collision_name("road", 0), "1ROAD_0")
        self.assertEqual(generate_collision_name("road", 5), "1ROAD_5")

    def test_grass_names(self):
        self.assertEqual(generate_collision_name("grass", 0), "1GRASS_0")
        self.assertEqual(generate_collision_name("grass", 12), "1GRASS_12")

    def test_kerb_names(self):
        self.assertEqual(generate_collision_name("kerb", 0), "1KERB_0")

    def test_sand_names(self):
        self.assertEqual(generate_collision_name("sand", 0), "1SAND_0")

    def test_wall_names(self):
        self.assertEqual(generate_collision_name("wall", 3), "1WALL_3")

    def test_case_insensitive(self):
        self.assertEqual(generate_collision_name("ROAD", 0), "1ROAD_0")
        self.assertEqual(generate_collision_name("Road", 1), "1ROAD_1")
        self.assertEqual(generate_collision_name(" road ", 2), "1ROAD_2")

    def test_unknown_material_raises(self):
        with self.assertRaises(ValueError):
            generate_collision_name("asphalt", 0)

    def test_all_known_materials(self):
        for tag in MATERIAL_PREFIXES:
            name = generate_collision_name(tag, 0)
            self.assertTrue(name.startswith("1"))
            self.assertTrue(name.endswith("_0"))


class TestPointInPolygon(unittest.TestCase):
    """Tests for the internal ray-casting PIP check."""

    def _square(self):
        # Unit square at origin: (0,0) to (10,10)
        return [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]

    def test_inside(self):
        self.assertTrue(_point_in_polygon_2d(5.0, 5.0, self._square()))

    def test_outside(self):
        self.assertFalse(_point_in_polygon_2d(15.0, 5.0, self._square()))
        self.assertFalse(_point_in_polygon_2d(-1.0, 5.0, self._square()))

    def test_near_edge(self):
        # Points very close to the edge may be borderline; we just verify no crash
        _point_in_polygon_2d(0.0, 5.0, self._square())

    def test_triangle(self):
        tri = [(0.0, 0.0), (10.0, 0.0), (5.0, 10.0)]
        self.assertTrue(_point_in_polygon_2d(5.0, 3.0, tri))
        self.assertFalse(_point_in_polygon_2d(0.0, 10.0, tri))


class TestSamplingGrid(unittest.TestCase):
    """Tests for generate_sampling_grid()."""

    def _square(self):
        return [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]

    def test_basic_grid(self):
        points, boundary_idx = generate_sampling_grid(self._square(), density=2.0)
        self.assertGreater(len(points), 4)
        # Boundary indices should match polygon vertex count
        self.assertEqual(len(boundary_idx), 4)
        # All boundary points should be from the original polygon
        for i in boundary_idx:
            self.assertIn(points[i], [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])

    def test_all_interior_points_inside_polygon(self):
        polygon = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
        points, boundary_idx = generate_sampling_grid(polygon, density=1.0)
        boundary_set = set(boundary_idx)
        for i, (px, pz) in enumerate(points):
            if i in boundary_set:
                continue
            self.assertTrue(
                _point_in_polygon_2d(px, pz, polygon),
                f"Interior point ({px}, {pz}) at index {i} is outside polygon",
            )

    def test_triangle_grid(self):
        tri = [(0.0, 0.0), (20.0, 0.0), (10.0, 20.0)]
        points, boundary_idx = generate_sampling_grid(tri, density=2.0)
        self.assertGreater(len(points), 3)
        self.assertEqual(len(boundary_idx), 3)

    def test_density_affects_count(self):
        sq = self._square()
        coarse_pts, _ = generate_sampling_grid(sq, density=5.0)
        fine_pts, _ = generate_sampling_grid(sq, density=1.0)
        self.assertGreater(len(fine_pts), len(coarse_pts))

    def test_invalid_density_raises(self):
        with self.assertRaises(ValueError):
            generate_sampling_grid(self._square(), density=0)
        with self.assertRaises(ValueError):
            generate_sampling_grid(self._square(), density=-1.0)

    def test_too_few_vertices_raises(self):
        with self.assertRaises(ValueError):
            generate_sampling_grid([(0.0, 0.0), (1.0, 1.0)], density=1.0)

    def test_no_boundary(self):
        points, boundary_idx = generate_sampling_grid(self._square(), density=2.0, include_boundary=False)
        self.assertEqual(len(boundary_idx), 0)
        self.assertGreater(len(points), 0)

    def test_output_to_file(self):
        """Write grid output to output/ for visual inspection."""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        polygon = [(0.0, 0.0), (30.0, 0.0), (30.0, 20.0), (15.0, 25.0), (0.0, 20.0)]
        points, boundary_idx = generate_sampling_grid(polygon, density=2.0)
        out = {
            "polygon": polygon,
            "points": points,
            "boundary_indices": boundary_idx,
            "total_points": len(points),
            "boundary_count": len(boundary_idx),
        }
        out_path = os.path.join(OUTPUT_DIR, "grid_output.json")
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        self.assertTrue(os.path.isfile(out_path))


class TestTriangulation(unittest.TestCase):
    """Tests for triangulate_points()."""

    def test_simple_square(self):
        pts = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (10.0, 0.0, 10.0), (0.0, 0.0, 10.0)]
        faces = triangulate_points(pts)
        self.assertGreater(len(faces), 0)
        # All face indices should be valid
        for f in faces:
            self.assertEqual(len(f), 3)
            for idx in f:
                self.assertGreaterEqual(idx, 0)
                self.assertLess(idx, len(pts))

    def test_many_points(self):
        # Generate a grid of points on the XZ plane
        pts = []
        for x in range(5):
            for z in range(5):
                pts.append((float(x), 0.0, float(z)))
        faces = triangulate_points(pts)
        self.assertGreater(len(faces), 0)
        # With 25 points and Delaunay, we expect a good number of triangles
        # For a planar grid, roughly 2*(n-1)^2 triangles
        self.assertGreaterEqual(len(faces), 10)

    def test_too_few_points(self):
        self.assertEqual(triangulate_points([]), [])
        self.assertEqual(triangulate_points([(0, 0, 0), (1, 0, 0)]), [])

    def test_collinear_points(self):
        pts = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
        # Collinear points -- Delaunay may or may not produce faces, but should not crash
        faces = triangulate_points(pts)
        # Just check it doesn't raise
        self.assertIsInstance(faces, list)

    def test_triangulation_output_to_file(self):
        """Write triangulation output for inspection."""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        pts = []
        for x in range(5):
            for z in range(5):
                pts.append((float(x) * 2, 0.0, float(z) * 2))
        faces = triangulate_points(pts)
        out = {
            "vertices": pts,
            "faces": faces,
            "vertex_count": len(pts),
            "face_count": len(faces),
        }
        out_path = os.path.join(OUTPUT_DIR, "triangulation_output.json")
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        self.assertTrue(os.path.isfile(out_path))


class TestClipLoading(unittest.TestCase):
    """Tests for load_clip_polygons() and extract_polygon_xz()."""

    def setUp(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    def _write_clip(self, tag: str, include_polys: list, exclude_polys: list | None = None) -> str:
        clip = {
            "source_tag": tag,
            "polygons": {
                "include": include_polys,
                "exclude": exclude_polys or [],
            },
        }
        path = os.path.join(OUTPUT_DIR, f"{tag}_clip.json")
        with open(path, "w") as f:
            json.dump(clip, f)
        return path

    def test_load_road_clip(self):
        poly = {
            "points_xyz": [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 0.0, 10.0], [0.0, 0.0, 10.0]],
            "tag": "road",
        }
        path = self._write_clip("road", [poly])
        data = load_clip_polygons(path)
        self.assertEqual(data["tag"], "road")
        self.assertEqual(len(data["include"]), 1)
        self.assertEqual(len(data["exclude"]), 0)

    def test_extract_polygon_xz(self):
        poly = {
            "points_xyz": [[1.0, 5.0, 2.0], [3.0, 5.0, 4.0], [5.0, 5.0, 6.0]],
        }
        xz = extract_polygon_xz(poly)
        # X and Z should be extracted (index 0 and 2)
        self.assertEqual(xz, [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)])

    def test_empty_clip(self):
        path = self._write_clip("sand", [])
        data = load_clip_polygons(path)
        self.assertEqual(data["tag"], "sand")
        self.assertEqual(len(data["include"]), 0)


class TestEndToEnd(unittest.TestCase):
    """End-to-end test of the pure-Python pipeline (no raycasting)."""

    def test_grid_to_triangulation(self):
        """Generate grid -> assign flat Y -> triangulate."""
        polygon_xz = [(0.0, 0.0), (20.0, 0.0), (20.0, 15.0), (0.0, 15.0)]
        grid_pts, boundary_idx = generate_sampling_grid(polygon_xz, density=2.0)
        self.assertGreater(len(grid_pts), 4)

        # Assign Y=0 (flat surface -- raycasting would set real heights)
        pts_3d = [(x, 0.0, z) for x, z in grid_pts]
        faces = triangulate_points(pts_3d, boundary_idx)
        self.assertGreater(len(faces), 0)

        # Write for inspection
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out = {
            "polygon_xz": polygon_xz,
            "grid_count": len(grid_pts),
            "boundary_count": len(boundary_idx),
            "faces_count": len(faces),
        }
        path = os.path.join(OUTPUT_DIR, "e2e_output.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        self.assertTrue(os.path.isfile(path))


if __name__ == "__main__":
    unittest.main()
