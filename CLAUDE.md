# SAM3 Track Segmentation - Assetto Corsa Track Mod Generator

## Project Overview

Automated pipeline: Drone-captured 2D images / 3D models → Blender processing → playable Assetto Corsa track mod.

## Key Directories

- `script/` — Core Python pipeline (sam3_track_gen.py is the main entry)
- `blender_scripts/` — Blender-side scripts and actions (bpy API)
- `blender_scripts/sam3_actions/` — Blender right-click menu actions
- `model/` — SAM3 model weights
- `test_images_shajing/` — Test dataset (Shajing track, b3dm + GeoTIFF)
- `output/` — All test intermediates and results go here
- `tests/` — Unit and module tests

## Tech Stack

- Python 3.10+, Blender 3.0+ (bpy), SAM3 (Meta)
- rasterio, Pillow, OpenCV, numpy, pyproj
- 3D Tiles (b3dm/glb), tileset.json
- Gemini API (model: gemini-2.0-flash, key in agent configs)

## Development Rules

1. **Plan First**: Every module must have a coding plan reviewed by the architect agent before implementation begins.
2. **Testability**: All modules must be independently testable. Blender-coupled code must be isolated so most logic can be tested without Blender.
3. **Test Location**: Tests go in `tests/` directory, mirroring the source structure.
4. **Output Location**: All test intermediates and results go to `output/` directory.
5. **Blender Scripts**: New Blender actions go in `blender_scripts/sam3_actions/` following existing patterns in `__init__.py`.
6. **Config**: Blender config lives in `blender_scripts/config.py`. Do not hardcode paths elsewhere.
7. **Naming**: Collision objects follow Assetto Corsa naming: `1WALL_N`, `1ROAD_N`, `1SAND_N`, `1KERB_N`, `1GRASS_N`.
8. **Integration**: Only integrate (TODO-9) after all module tests pass.

## Workflow Pipeline

```
GeoTIFF → SAM3 segmentation → clips → blender JSON → Blender polygons → 3D surface extraction → collision meshes → game objects → Assetto Corsa mod
```

## Reference Code

- `old_blender_scripts_example.py` — Contains reference implementations for b3dm→glb conversion, texture unpacking, material conversion.
- `PROJECT.md` — Detailed documentation of existing modules and data formats.
- `Agents需求.md` — Full requirements specification.
