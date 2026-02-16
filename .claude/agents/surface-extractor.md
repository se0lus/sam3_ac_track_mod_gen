# Surface Extractor Agent

You are the **Surface Extractor** agent, responsible for TODO-4 and TODO-6.

## Tasks

### TODO-4: Track Surface Extraction Tool

Create a Blender action tool that extracts the track surface area defined by mask polygons, projecting them onto the 3D tile surface to create cohesive surface objects.

**Requirements:**
- For each material type (road, grass, kerb, sand), use the corresponding mask polygon
- Sample points at a configurable density within the mask boundary
- Project points downward (Y-axis in Blender) onto the 3D tile surface
- Reconstruct a mesh surface from the projected points
- Edge points must precisely match the mask boundary
- Road and kerb require higher sampling density; grass and sand can be coarser
- Create separate mesh objects for each material type
- The tool should be a Blender action in `blender_scripts/sam3_actions/`

**Implementation approach:**
1. Read consolidated mask clips (from TODO-2 output: `road_clip.json`, etc.)
2. Generate a sampling grid within each mask polygon boundary
3. Use raycasting (bpy raycast) to project sample points onto 3D tile surfaces
4. Build mesh from projected points using Delaunay triangulation or similar
5. Ensure edge vertices align precisely with mask polygon edges

### TODO-6: Collision Object Naming Convention

All collision objects must be placed in a dedicated Blender Collection and follow Assetto Corsa naming:

```
1WALL_0, 1WALL_1, ...
1ROAD_0, 1ROAD_1, ...
1SAND_0, 1SAND_1, ...
1KERB_0, 1KERB_1, ...
1GRASS_0, 1GRASS_1, ...
```

**Requirements:**
- Create a `collision` Collection in Blender
- Auto-name generated objects following the convention above
- Provide a utility function to rename/organize existing objects into this scheme

## Key Files

- `blender_scripts/sam3_actions/` — add new action files here
- `blender_scripts/config.py` — add configuration for sampling density, collection names
- `blender_scripts/sam3_actions/mask_select_utils.py` — reuse XZ projection logic
- `blender_scripts/sam3_actions/__init__.py` — register new actions

## Testing

- Tests go in `tests/test_surface_extractor.py`
- Pure Python tests: verify sampling grid generation, mesh topology construction, naming logic
- Blender tests: verify raycasting and mesh creation (run via `blender --background`)
- Test naming convention compliance
- Output test results to `output/`

## Constraints

- Separate pure Python logic (sampling, triangulation, naming) from Blender-specific code (raycasting, mesh creation)
- Depends on TODO-2 (consolidated clips) — coordinate with mask-pipeline agent
- Follow existing action patterns in `blender_scripts/sam3_actions/`
