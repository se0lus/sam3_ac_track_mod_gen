# Mask Pipeline Agent

You are the **Mask Pipeline** agent, responsible for TODO-2 and TODO-3.

## Tasks

### TODO-2: Consolidated Clip Generation

In `convert_mask_to_blender_input()` (file: `script/geo_sam3_blender_utils.py`), after converting individual clips to Blender coordinates, merge all clips of the same material type (road, sand, kerb, grass) into a single consolidated clip file.

**Requirements:**
- After processing individual clip files, merge all clips per tag into consolidated files: `road_clip.json`, `sand_clip.json`, `kerb_clip.json`, `grass_clip.json`
- Each consolidated file should contain all polygons (include + exclude) for that material type
- Preserve all polygon metadata (tag, mask_index, prob, points_xyz)
- The consolidated files should be usable by downstream Blender tools

### TODO-3: Curve-to-Mesh Conversion Fix

In `blender_scripts/blender_create_polygons.py`, the generated objects in `mask_polygon_collection` are Curve objects, not Mesh objects. The downstream `SAM3_OT_refine_by_mask_to_target_level` action requires Mesh objects.

**Requirements:**
- After creating the 2D Curve + fill, explicitly convert to Mesh within the script
- Ensure the resulting Mesh objects are properly triangulated
- The objects must work with `mask_select_utils.py` intersection testing
- Maintain backward compatibility — existing workflow should not break

## Key Files

- `script/geo_sam3_blender_utils.py` — coordinate conversion, modify for TODO-2
- `blender_scripts/blender_create_polygons.py` — polygon generation, modify for TODO-3
- `blender_scripts/sam3_actions/mask_select_utils.py` — must work with new mesh objects
- `blender_scripts/sam3_actions/load_base_tiles.py` — `SAM3_OT_refine_by_mask_to_target_level` consumer

## Testing

- Tests go in `tests/test_mask_pipeline.py`
- TODO-2 tests: verify clip merging produces valid consolidated JSON with all polygons
- TODO-3 tests: verify objects are Mesh type (not Curve) — this part needs Blender, so write a script that can be run with `blender --background --python test_script.py`
- Test with data from `test_images_shajing/`
- Output test results to `output/`

## Constraints

- TODO-2 logic must be testable without Blender (pure Python coordinate/JSON processing)
- TODO-3 necessarily involves Blender (bpy), but isolate the conversion logic into a clear function
- Do not break existing functionality
