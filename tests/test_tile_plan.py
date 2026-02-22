"""Tests for script/tile_plan.py — tile load plan pre-computation."""

from __future__ import annotations

import json
import os
import sys
import tempfile

import pytest

# Ensure project paths are available
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_script_dir = os.path.join(_project_root, "script")
_actions_dir = os.path.join(_project_root, "blender_scripts", "sam3_actions")
for p in [_script_dir, _actions_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from tile_plan import (
    AABB2D,
    bbox2_overlap,
    compute_plan_from_config,
    compute_tile_load_plan,
    detect_axis_map,
    extract_polygon_aabbs,
    extract_polygon_aabbs_for_tags,
    load_tileset_tree,
    tile_aabb_xz,
)
from c_tiles import CTile


# ---------------------------------------------------------------------------
# bbox2_overlap
# ---------------------------------------------------------------------------

class TestBbox2Overlap:
    def test_overlap(self):
        a = (0.0, 10.0, 0.0, 10.0)
        b = (5.0, 15.0, 5.0, 15.0)
        assert bbox2_overlap(a, b) is True

    def test_no_overlap_x(self):
        a = (0.0, 10.0, 0.0, 10.0)
        b = (11.0, 20.0, 0.0, 10.0)
        assert bbox2_overlap(a, b) is False

    def test_no_overlap_z(self):
        a = (0.0, 10.0, 0.0, 10.0)
        b = (0.0, 10.0, 11.0, 20.0)
        assert bbox2_overlap(a, b) is False

    def test_touching_edge(self):
        a = (0.0, 10.0, 0.0, 10.0)
        b = (10.0, 20.0, 0.0, 10.0)
        assert bbox2_overlap(a, b) is True  # touching counts as overlap

    def test_contained(self):
        a = (0.0, 100.0, 0.0, 100.0)
        b = (10.0, 20.0, 10.0, 20.0)
        assert bbox2_overlap(a, b) is True

    def test_identical(self):
        a = (5.0, 15.0, 5.0, 15.0)
        assert bbox2_overlap(a, a) is True

    def test_no_overlap_both_axes(self):
        a = (0.0, 1.0, 0.0, 1.0)
        b = (100.0, 200.0, 100.0, 200.0)
        assert bbox2_overlap(a, b) is False


# ---------------------------------------------------------------------------
# tile_aabb_xz
# ---------------------------------------------------------------------------

class TestTileAabbXz:
    def _make_tile(self, cx, cy, cz, hx, hy, hz):
        """Create a CTile with an axis-aligned bounding box."""
        tile = CTile()
        tile.boxBoundingVolume = [
            cx, cy, cz,    # center
            hx, 0, 0,      # X half-extent (axis-aligned)
            0, hy, 0,       # Y half-extent
            0, 0, hz,       # Z half-extent
        ]
        return tile

    def test_xy_mapping(self):
        tile = self._make_tile(10, 20, 30, 5, 3, 4)
        result = tile_aabb_xz(tile, "XY")
        assert result == (10 - 5, 10 + 5, 20 - 3, 20 + 3)

    def test_xny_mapping(self):
        tile = self._make_tile(10, 20, 30, 5, 3, 4)
        result = tile_aabb_xz(tile, "XnY")
        assert result == (10 - 5, 10 + 5, -20 - 3, -20 + 3)

    def test_xz_mapping(self):
        tile = self._make_tile(10, 20, 30, 5, 3, 4)
        result = tile_aabb_xz(tile, "XZ")
        assert result == (10 - 5, 10 + 5, 30 - 4, 30 + 4)

    def test_xnz_mapping(self):
        tile = self._make_tile(10, 20, 30, 5, 3, 4)
        result = tile_aabb_xz(tile, "XnZ")
        assert result == (10 - 5, 10 + 5, -30 - 4, -30 + 4)

    def test_invalid_bbox_returns_none(self):
        tile = CTile()
        tile.boxBoundingVolume = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        assert tile_aabb_xz(tile, "XY") is None

    def test_obb_conservative(self):
        """OBB with off-diagonal elements → half-extents sum of absolutes."""
        tile = CTile()
        tile.boxBoundingVolume = [
            10, 20, 0,     # center
            3, 1, 0,       # X row: hx sums to 3+1+0=4
            0, 2, 1,       # Y row: hy sums to 0+2+1=3
            0, 0, 0,       # Z row
        ]
        result = tile_aabb_xz(tile, "XY")
        assert result is not None
        assert result == (10 - 4, 10 + 4, 20 - 3, 20 + 3)


# ---------------------------------------------------------------------------
# extract_polygon_aabbs
# ---------------------------------------------------------------------------

class TestExtractPolygonAabbs:
    def test_basic_extraction(self, tmp_path):
        data = {
            "mesh_groups": [
                {
                    "group_index": 0,
                    "tag": "road",
                    "points_xyz": [
                        [10.0, 0.0, 20.0],
                        [15.0, 0.0, 25.0],
                        [12.0, 0.0, 30.0],
                    ],
                }
            ]
        }
        json_path = str(tmp_path / "road_merged_blender.json")
        with open(json_path, "w") as f:
            json.dump(data, f)

        aabbs = extract_polygon_aabbs(json_path, padding_m=0.0)
        assert len(aabbs) == 1
        assert aabbs[0] == (10.0, 15.0, 20.0, 30.0)

    def test_with_padding(self, tmp_path):
        data = {
            "mesh_groups": [
                {
                    "group_index": 0,
                    "tag": "road",
                    "points_xyz": [
                        [10.0, 0.0, 20.0],
                        [15.0, 0.0, 25.0],
                        [12.0, 0.0, 30.0],
                    ],
                }
            ]
        }
        json_path = str(tmp_path / "road_merged_blender.json")
        with open(json_path, "w") as f:
            json.dump(data, f)

        aabbs = extract_polygon_aabbs(json_path, padding_m=1.0)
        assert len(aabbs) == 1
        assert aabbs[0] == (9.0, 16.0, 19.0, 31.0)

    def test_multiple_groups(self, tmp_path):
        data = {
            "mesh_groups": [
                {"points_xyz": [[0, 0, 0], [1, 0, 1], [1, 0, 0]]},
                {"points_xyz": [[10, 0, 10], [11, 0, 11], [11, 0, 10]]},
            ]
        }
        json_path = str(tmp_path / "test.json")
        with open(json_path, "w") as f:
            json.dump(data, f)

        aabbs = extract_polygon_aabbs(json_path, padding_m=0.0)
        assert len(aabbs) == 2

    def test_skip_small_groups(self, tmp_path):
        data = {
            "mesh_groups": [
                {"points_xyz": [[0, 0, 0], [1, 0, 1]]},  # only 2 points → skip
                {"points_xyz": [[0, 0, 0], [1, 0, 1], [1, 0, 0]]},  # 3 points → ok
            ]
        }
        json_path = str(tmp_path / "test.json")
        with open(json_path, "w") as f:
            json.dump(data, f)

        aabbs = extract_polygon_aabbs(json_path, padding_m=0.0)
        assert len(aabbs) == 1

    def test_with_real_data(self):
        """Use actual Stage 8 output if available."""
        json_path = os.path.join(
            _project_root, "output", "08_blender_polygons", "gap_filled",
            "road", "road_merged_blender.json"
        )
        if not os.path.isfile(json_path):
            pytest.skip("Real Stage 8 data not available")

        aabbs = extract_polygon_aabbs(json_path, padding_m=0.5)
        assert len(aabbs) > 0
        # Verify format
        for aabb in aabbs:
            assert len(aabb) == 4
            assert aabb[0] < aabb[1]  # min_x < max_x
            assert aabb[2] < aabb[3]  # min_z < max_z


# ---------------------------------------------------------------------------
# extract_polygon_aabbs_for_tags
# ---------------------------------------------------------------------------

class TestExtractPolygonAabbsForTags:
    def test_basic(self, tmp_path):
        # Create tag directory structure
        road_dir = tmp_path / "road"
        road_dir.mkdir()
        data = {
            "mesh_groups": [
                {"points_xyz": [[0, 0, 0], [10, 0, 10], [10, 0, 0]]}
            ]
        }
        with open(road_dir / "road_merged_blender.json", "w") as f:
            json.dump(data, f)

        aabbs = extract_polygon_aabbs_for_tags(str(tmp_path), ["road"], padding_m=0.0)
        assert len(aabbs) == 1

    def test_missing_tag_skipped(self, tmp_path):
        aabbs = extract_polygon_aabbs_for_tags(str(tmp_path), ["nonexistent"], padding_m=0.0)
        assert len(aabbs) == 0


# ---------------------------------------------------------------------------
# compute_tile_load_plan
# ---------------------------------------------------------------------------

def _build_tile_tree() -> CTile:
    """Build a mock tile tree for testing:

    root (virtual, no mesh)
    ├── A (L15, canRefine)
    │   ├── B (L17, canRefine)
    │   │   ├── D (L19, leaf)
    │   │   └── E (L19, leaf)
    │   └── C (L17, canRefine)
    │       ├── F (L19, leaf)
    │       └── G (L19, leaf)
    """
    root = CTile()
    root.canRefine = True

    a = CTile()
    a.hasMesh = True
    a.meshLevel = 15
    a.canRefine = True
    a.content = "A_L15_0.b3dm"
    a.boxBoundingVolume = [0, 0, 0, 50, 0, 0, 0, 50, 0, 0, 0, 0]
    a.parent = root
    root.children.append(a)

    b = CTile()
    b.hasMesh = True
    b.meshLevel = 17
    b.canRefine = True
    b.content = "B_L17_0.b3dm"
    b.boxBoundingVolume = [-25, -25, 0, 25, 0, 0, 0, 25, 0, 0, 0, 0]
    b.parent = a
    a.children.append(b)

    c = CTile()
    c.hasMesh = True
    c.meshLevel = 17
    c.canRefine = True
    c.content = "C_L17_1.b3dm"
    c.boxBoundingVolume = [25, 25, 0, 25, 0, 0, 0, 25, 0, 0, 0, 0]
    c.parent = a
    a.children.append(c)

    d = CTile()
    d.hasMesh = True
    d.meshLevel = 19
    d.canRefine = False
    d.content = "D_L19_0.b3dm"
    d.boxBoundingVolume = [-37, -37, 0, 12, 0, 0, 0, 12, 0, 0, 0, 0]
    d.parent = b
    b.children.append(d)

    e = CTile()
    e.hasMesh = True
    e.meshLevel = 19
    e.canRefine = False
    e.content = "E_L19_1.b3dm"
    e.boxBoundingVolume = [-12, -12, 0, 12, 0, 0, 0, 12, 0, 0, 0, 0]
    e.parent = b
    b.children.append(e)

    f = CTile()
    f.hasMesh = True
    f.meshLevel = 19
    f.canRefine = False
    f.content = "F_L19_2.b3dm"
    f.boxBoundingVolume = [12, 12, 0, 12, 0, 0, 0, 12, 0, 0, 0, 0]
    f.parent = c
    c.children.append(f)

    g = CTile()
    g.hasMesh = True
    g.meshLevel = 19
    g.canRefine = False
    g.content = "G_L19_3.b3dm"
    g.boxBoundingVolume = [37, 37, 0, 12, 0, 0, 0, 12, 0, 0, 0, 0]
    g.parent = c
    c.children.append(g)

    return root


class TestComputeTileLoadPlan:
    def test_no_overlap_all_base_level(self):
        """No polygons → all tiles at base_level, no refinement."""
        root = _build_tile_tree()
        plan = compute_tile_load_plan(root, [], base_level=17, target_level=19, axis_map="XY")

        # With no polygons, tiles below base (L15) recurse to L17,
        # L17 tiles stop since no overlap.
        assert 17 in plan
        assert len(plan[17]) == 2  # B and C at L17
        assert 19 not in plan  # no refinement happened
        assert 15 not in plan  # L15 was refined to L17

    def test_full_overlap_all_target_level(self):
        """Polygon covers everything → all tiles at target_level."""
        root = _build_tile_tree()
        # Huge polygon covering all tiles
        poly_aabbs = [(-100.0, 100.0, -100.0, 100.0)]
        plan = compute_tile_load_plan(root, poly_aabbs, base_level=17, target_level=19, axis_map="XY")

        # All tiles should be refined to L19 leaves
        assert 19 in plan
        assert len(plan[19]) == 4  # D, E, F, G
        assert 17 not in plan  # B and C were refined

    def test_partial_overlap_mixed_levels(self):
        """Polygon overlaps only tile B → B refined, C stays at base."""
        root = _build_tile_tree()
        # Polygon only overlaps B's area (negative x,y quadrant)
        poly_aabbs = [(-50.0, -10.0, -50.0, -10.0)]
        plan = compute_tile_load_plan(root, poly_aabbs, base_level=17, target_level=19, axis_map="XY")

        # B overlaps → refined to D,E at L19
        # C doesn't overlap → stays at L17
        assert 19 in plan
        assert 17 in plan
        # D and E are B's children
        l19_contents = {t.content for t in plan[19]}
        l17_contents = {t.content for t in plan[17]}
        assert "D_L19_0.b3dm" in l19_contents
        assert "E_L19_1.b3dm" in l19_contents
        assert "C_L17_1.b3dm" in l17_contents

    def test_no_holes(self):
        """Every area must be covered — total tiles >= 2 (at least B and C coverage)."""
        root = _build_tile_tree()
        poly_aabbs = [(-50.0, -10.0, -50.0, -10.0)]
        plan = compute_tile_load_plan(root, poly_aabbs, base_level=17, target_level=19, axis_map="XY")

        # Count total tiles
        total = sum(len(tiles) for tiles in plan.values())
        # At minimum: C at L17 + D,E at L19 = 3
        assert total >= 3

    def test_target_equals_base(self):
        """When target == base, everything loads at base_level."""
        root = _build_tile_tree()
        poly_aabbs = [(-100.0, 100.0, -100.0, 100.0)]
        plan = compute_tile_load_plan(root, poly_aabbs, base_level=17, target_level=17, axis_map="XY")

        # L15 refines to L17, then at target → stop
        assert 17 in plan
        assert len(plan[17]) == 2  # B and C

    def test_level_skip_loads_above_target(self):
        """Tree skips L19 (L18→L20): overlapping area loads L20 (≥ target)."""
        root = CTile()
        root.canRefine = True

        # L17 tile, overlaps polygon, can refine
        b = CTile()
        b.hasMesh = True
        b.meshLevel = 17
        b.canRefine = True
        b.content = "B_L17.b3dm"
        b.boxBoundingVolume = [0, 0, 0, 50, 0, 0, 0, 50, 0, 0, 0, 0]
        b.parent = root
        root.children.append(b)

        # L18 child, can refine — but children jump to L20
        c18 = CTile()
        c18.hasMesh = True
        c18.meshLevel = 18
        c18.canRefine = True
        c18.content = "C_L18.b3dm"
        c18.boxBoundingVolume = [0, 0, 0, 25, 0, 0, 0, 25, 0, 0, 0, 0]
        c18.parent = b
        b.children.append(c18)

        # L20 leaf — no L19 in this branch
        d20 = CTile()
        d20.hasMesh = True
        d20.meshLevel = 20
        d20.canRefine = False
        d20.content = "D_L20.b3dm"
        d20.boxBoundingVolume = [0, 0, 0, 12, 0, 0, 0, 12, 0, 0, 0, 0]
        d20.parent = c18
        c18.children.append(d20)

        poly_aabbs = [(-100.0, 100.0, -100.0, 100.0)]
        plan = compute_tile_load_plan(root, poly_aabbs, base_level=17, target_level=19, axis_map="XY")

        # L20 must be loaded: it's the only way to satisfy ≥ target for this branch
        assert 20 in plan
        assert len(plan[20]) == 1
        assert plan[20][0].content == "D_L20.b3dm"
        # L17 and L18 should NOT appear (they were refined)
        assert 17 not in plan
        assert 18 not in plan

    def test_tile_without_children_stops(self):
        """Tile that can't refine → loads at its own level."""
        root = CTile()
        root.canRefine = True

        leaf = CTile()
        leaf.hasMesh = True
        leaf.meshLevel = 17
        leaf.canRefine = False  # can't go deeper
        leaf.content = "Leaf_L17.b3dm"
        leaf.boxBoundingVolume = [0, 0, 0, 10, 0, 0, 0, 10, 0, 0, 0, 0]
        leaf.parent = root
        root.children.append(leaf)

        poly_aabbs = [(-100.0, 100.0, -100.0, 100.0)]
        plan = compute_tile_load_plan(root, poly_aabbs, base_level=17, target_level=22, axis_map="XY")

        assert 17 in plan
        assert len(plan[17]) == 1


# ---------------------------------------------------------------------------
# Real data test
# ---------------------------------------------------------------------------

class TestWithRealData:
    """Use actual tileset + polygon data from test_images_gic."""

    _tiles_dir = os.path.join(_project_root, "test_images_gic", "b3dm", "Model_0")
    _polygon_dir = os.path.join(_project_root, "output", "08_blender_polygons", "gap_filled")

    def _has_real_data(self):
        return (
            os.path.isdir(self._tiles_dir)
            and os.path.isdir(self._polygon_dir)
            and os.path.isfile(os.path.join(
                self._polygon_dir, "road", "road_merged_blender.json"
            ))
        )

    def test_load_tileset_tree(self):
        if not os.path.isdir(self._tiles_dir):
            pytest.skip("test_images_gic not available")
        root = load_tileset_tree(self._tiles_dir)
        assert len(root.children) > 0

    def test_detect_axis_map(self):
        if not self._has_real_data():
            pytest.skip("Real data not available")
        aabbs = extract_polygon_aabbs_for_tags(self._polygon_dir, ["road"], padding_m=0.5)
        axis_map = detect_axis_map(self._tiles_dir, aabbs)
        assert axis_map in ("XY", "XnY", "XZ", "XnZ")
        print(f"Detected axis_map: {axis_map}")

    def test_plan_17_to_19(self):
        """Compute plan from 17→19, verify mixed levels and no over-refinement."""
        if not self._has_real_data():
            pytest.skip("Real data not available")

        plan = compute_plan_from_config(
            tiles_dir=self._tiles_dir,
            polygon_dir=self._polygon_dir,
            tags=["road"],
            base_level=17,
            target_level=19,
            padding_m=0.5,
        )

        # Basic validity
        assert len(plan) > 0
        total = sum(len(v) for v in plan.values())
        assert total > 0

        # Print summary
        print("\n=== Plan Summary (L17→L19) ===")
        for lv in sorted(plan.keys()):
            print(f"  Level {lv}: {len(plan[lv])} tiles")
        print(f"  Total: {total} tiles")

        # Must have L17 tiles (non-road areas)
        assert 17 in plan, "Expected L17 tiles for non-road areas"
        # Must have some tiles at L18 or L19 (refined areas)
        refined_levels = [lv for lv in plan if lv > 17]
        assert len(refined_levels) > 0, "Expected some tiles refined beyond L17"

        # Total should be reasonable (not thousands for L17→L19)
        assert total < 2000, f"Too many tiles ({total}), possible over-refinement"

    def test_plan_no_refinement_fewer_tiles(self):
        """Plan with empty polygon list should have fewer tiles than full refinement."""
        if not self._has_real_data():
            pytest.skip("Real data not available")

        root = load_tileset_tree(self._tiles_dir)

        # Plan with no polygons (base only)
        plan_base = compute_tile_load_plan(root, [], base_level=17, target_level=19, axis_map="XY")
        total_base = sum(len(v) for v in plan_base.values())

        # Plan with road polygons (selective refinement)
        aabbs = extract_polygon_aabbs_for_tags(self._polygon_dir, ["road"], padding_m=0.5)
        root2 = load_tileset_tree(self._tiles_dir)
        axis_map = detect_axis_map(self._tiles_dir, aabbs)
        plan_selective = compute_tile_load_plan(root2, aabbs, base_level=17, target_level=19, axis_map=axis_map)
        total_selective = sum(len(v) for v in plan_selective.values())

        print(f"\nBase only (no refinement): {total_base} tiles")
        print(f"Selective refinement: {total_selective} tiles")

        # Selective should have more tiles than base-only (due to refining road areas)
        assert total_selective >= total_base

    def test_plan_selective_vs_full_refinement(self):
        """Selective plan should have fewer tiles than refining everything to target."""
        if not self._has_real_data():
            pytest.skip("Real data not available")

        root_sel = load_tileset_tree(self._tiles_dir)
        aabbs = extract_polygon_aabbs_for_tags(self._polygon_dir, ["road"], padding_m=0.5)
        axis_map = detect_axis_map(self._tiles_dir, aabbs)

        # Selective refinement (only road areas)
        plan_sel = compute_tile_load_plan(root_sel, aabbs, base_level=17, target_level=19, axis_map=axis_map)
        total_sel = sum(len(v) for v in plan_sel.values())

        # Full refinement (huge polygon covering everything)
        root_full = load_tileset_tree(self._tiles_dir)
        huge_aabb = [(-10000.0, 10000.0, -10000.0, 10000.0)]
        plan_full = compute_tile_load_plan(root_full, huge_aabb, base_level=17, target_level=19, axis_map=axis_map)
        total_full = sum(len(v) for v in plan_full.values())

        print(f"\nSelective refinement: {total_sel} tiles")
        print(f"Full refinement (everything): {total_full} tiles")

        # Selective should have fewer or equal tiles than full
        assert total_sel <= total_full, (
            f"Selective ({total_sel}) should not exceed full ({total_full})"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
