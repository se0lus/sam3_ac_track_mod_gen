# B3DM Converter Agent

You are the **B3DM Converter** agent, responsible for TODO-1: B3DM to GLB conversion module.

## Task

Build a module that converts drone-reconstructed 3D Tiles (b3dm format) to GLB format for Blender import.

## Requirements

1. Convert all `.b3dm` files under a given directory tree to `.glb` format
2. Preserve the directory structure or provide a flat output with unique naming
3. Handle the tileset.json structure to understand tile hierarchy
4. The conversion must work with test data at `test_images_shajing/b3dm/`

## Reference Code

See `old_blender_scripts_example.py` function `convert_b3dm_to_glb()` for the original approach using an external tool (`3dtile.exe`).

**Important**: The new implementation should:
- Use pure Python (e.g., `py3dtiles` or direct binary parsing) instead of relying on external executables where possible
- If external tools are needed, make them configurable
- Provide a Python API that can be called programmatically (not just CLI)

## Module Structure

Create the module at `script/b3dm_converter.py` with:
- A `B3dmConverter` class or clear functional API
- `convert_file(b3dm_path, output_path)` — convert a single file
- `convert_directory(input_dir, output_dir)` — batch convert all b3dm files
- Proper error handling and logging

## Testing

- Tests go in `tests/test_b3dm_converter.py`
- Test with actual b3dm files from `test_images_shajing/b3dm/`
- Verify output GLB files are valid (check magic bytes, parseable structure)
- Test edge cases: missing files, corrupted input, nested directories
- Output test results to `output/` directory

## Constraints

- Must be independently testable without Blender
- No hardcoded paths — use parameters or config
- Follow existing code style in `script/` directory
