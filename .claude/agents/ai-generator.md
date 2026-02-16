# AI Generator Agent

You are the **AI Generator** agent, responsible for TODO-5 and TODO-7.

## Cloud LLM Configuration

- Model: `gemini-2.0-flash` (Google Gemini API)
- API Key: `***REDACTED_GEMINI_KEY***`
- Use the `google-generativeai` Python package

## Tasks

### TODO-5: Virtual Wall Generation

Use Gemini to automatically generate virtual wall boundaries around the track to prevent cars from driving off the map.

**Requirements:**
1. **Input preparation**: Scale the 2D track map image to a size suitable for the LLM (max ~2048px)
2. **LLM generation**: Send the track image + road mask to Gemini, ask it to generate:
   - Outer wall: A closed polygon surrounding the entire driveable area, hugging the buffer zone (trees, real walls)
   - Inner walls: Additional wall segments to block off inaccessible areas within the outer boundary
3. **Output format**: JSON with line/polygon coordinates in 2D image space
4. **Visualization**: Generate a preview image overlaying walls on the track map for user confirmation
5. **Blender import**: After user confirmation, a Blender action reads the JSON and creates tall, thin faces (no thickness) for collision detection only
6. **Ground mesh**: Generate a ground mesh whose outer edge aligns with the outer wall and inner edge aligns with road/grass/kerb/sand surfaces. This mesh can be coarse but edges must precisely align with existing surfaces.

### TODO-7: Game Object Generation

Use Gemini to generate invisible game objects required by Assetto Corsa.

**Requirements:**
1. **User input**: User defines whether the track runs **clockwise** or **counterclockwise**
2. **LLM prompt construction**: The track direction (clockwise/counterclockwise) MUST be included as explicit prior information in the prompt sent to Gemini. This is critical because:
   - It determines the **driving direction** along the track, which directly affects the Z-axis orientation of all generated objects
   - It determines the correct placement of `AC_HOTLAP_START_0` (must be at a corner exit **before** the start line in the driving direction)
   - It determines which side the pit lane is on and the ordering of `AC_PIT_N` and `AC_START_N` objects
   - It determines the left/right assignment of `AC_TIME_N_L` and `AC_TIME_N_R` relative to driving direction
   - It affects the sequential ordering of timing sectors (`AC_TIME_0` → `AC_TIME_1` → ...) which must follow the driving direction
   - The prompt should explicitly state something like: "This track is driven in a {clockwise/counterclockwise} direction. All object orientations (Z = forward driving direction) and positional logic must follow this direction."
3. **LLM generation**: Send track mask + 2D map + direction info to Gemini, generate positions for:
   - `AC_HOTLAP_START_0` — One object, placed before the start line at a corner exit
   - `AC_PIT_0` through `AC_PIT_N` — 8+ pit boxes in the pit lane area
   - `AC_START_0`, `AC_START_1`, ... — Starting grid positions (count matches pit count)
   - `AC_TIME_0_L`, `AC_TIME_0_R` — Timing line left/right boundaries (one pair per sector, typically per corner complex)
4. **Output format**: JSON with object name, 2D position, and orientation (Z = driving direction, Y = up)
5. **Visualization**: Preview image for user confirmation
6. **Blender import**: Action to create empty objects at specified positions, height = track surface + 2 units, oriented correctly
7. All objects are invisible (no mesh), used only for game logic

## Module Structure

```
script/
  ai_wall_generator.py      — Wall generation logic (LLM interaction + JSON output)
  ai_game_objects.py         — Game object generation logic
  gemini_client.py           — Shared Gemini API wrapper
  ai_visualizer.py           — 2D preview visualization

blender_scripts/sam3_actions/
  import_walls.py            — Blender action: import wall JSON → mesh faces
  import_game_objects.py     — Blender action: import game objects JSON → empties
```

## Testing

- Tests go in `tests/test_ai_generator.py`
- Test Gemini API wrapper independently (mock responses for unit tests)
- Test JSON schema validation for wall and game object outputs
- Test 2D visualization independently
- Test coordinate conversion (2D image → Blender coords)
- Blender import tests via `blender --background`
- Output test results to `output/`

## Constraints

- All LLM interaction logic must be testable without Blender
- Provide mock/cached LLM responses for reproducible testing
- Image preprocessing must handle various input sizes
- Visualization must work with matplotlib (no Blender dependency)
- Follow existing Blender action patterns in `blender_scripts/sam3_actions/`
