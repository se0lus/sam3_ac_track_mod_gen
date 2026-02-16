# Texture Tools Agent

You are the **Texture Tools** agent, responsible for TODO-8.

## Task

Create Blender tools for unpacking and converting textures to PNG format, accessible via the right-click menu.

## Requirements

### Texture Unpacking
- Unpack all embedded textures from the current Blender scene
- Handle textures referenced by materials in imported GLB/glTF models

### Format Conversion
- Convert all JPG/JPEG textures to PNG format
- Update material references to point to the new PNG files
- Handle file path management properly

### Material Conversion
- Convert all materials to Principled BSDF shader nodes
- Preserve texture assignments during conversion

### PIL/Pillow Compatibility
- Blender's bundled Python may not have PIL/Pillow installed
- Strategy 1: Use Blender's built-in image API (`bpy.data.images`) for format conversion where possible
- Strategy 2: If PIL is needed, provide a fallback that installs it into Blender's Python (`blender_python -m pip install Pillow`)
- Strategy 3: Use subprocess to call external Python with PIL for conversion tasks

## Reference Code

See `old_blender_scripts_example.py` for reference implementations:
- `unpack_textures()` — Original texture unpacking logic
- `convert_jpg_png()` — JPG to PNG conversion (uses PIL)
- `convert_all_materials_to_bsdf()` — Material conversion to Principled BSDF

## Module Structure

```
blender_scripts/sam3_actions/
  texture_tools.py    — Blender action: unpack, convert, and manage textures
```

The tool should register three right-click menu actions:
1. **Unpack All Textures** — Unpack embedded textures
2. **Convert Textures to PNG** — Convert JPG→PNG and update references
3. **Convert Materials to BSDF** — Convert all materials to Principled BSDF

## Key Files

- `blender_scripts/sam3_actions/__init__.py` — Register new actions here
- `blender_scripts/blender_helpers.py` — Right-click menu framework
- `old_blender_scripts_example.py` — Reference implementations

## Testing

- Tests go in `tests/test_texture_tools.py`
- Test file format detection logic (pure Python, no Blender)
- Test path manipulation and file naming logic (pure Python)
- Blender integration tests via `blender --background --python`
- Verify textures are correctly unpacked and converted
- Test PIL availability detection and fallback mechanisms
- Output test results to `output/`

## Constraints

- Must work within Blender's Python environment
- Handle the case where PIL is not available gracefully
- Follow existing action registration patterns in `sam3_actions/__init__.py`
- Use `blender_helpers.py` framework for right-click menu integration
- Do not break existing actions
