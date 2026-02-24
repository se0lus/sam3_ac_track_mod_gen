# SAM3 AC Track Mod Generator

Automatically generate playable [Assetto Corsa](https://www.assettocorsa.net/) track mods from drone aerial imagery and 3D photogrammetry data.

**Drone 2D Ortho / 3D Tiles &rarr; SAM3 Semantic Segmentation &rarr; Blender 3D Processing &rarr; Assetto Corsa Track Mod**

## Features

- **11-Stage Automated Pipeline** &mdash; from raw GeoTIFF + 3D Tiles to a ready-to-play AC track mod
- **SAM3 Semantic Segmentation** &mdash; 8-class segmentation (road, grass, sand, kerb, trees, building, water, concrete) with fallback prompts
- **AI + Procedural Hybrid** &mdash; Gemini VLM for high-level decisions (object placement, track descriptions), algorithms for geometric precision (wall generation, centerline extraction)
- **Web Dashboard** &mdash; Flask-based control panel with real-time SSE logs, progress tracking, and configuration editor
- **6 Interactive Editors** &mdash; layout, surface mask, wall, game object, centerline editors with Leaflet map integration
- **Human-in-the-Loop** &mdash; every critical stage has a manual editing step; edits are transparently respected by downstream stages via junction links
- **Multi-Layout Support** &mdash; generate multiple track layouts (clockwise, counterclockwise, etc.) from a single dataset
- **Blender Automation** &mdash; headless Blender pipeline for polygon generation, surface extraction, tile refinement, and FBX/KN5 export

## Pipeline Overview

```
GeoTIFF + 3D Tiles (b3dm)
    |
    v
 S1   B3DM -> GLB               3D Tiles format conversion
 S2   Full-map SAM3              8-class semantic segmentation
 S2a  Layout Editor ........     (manual, web editor)
 S3   Full-map Clipping          ~40m x 40m tile splitting
 S4   Per-tile SAM3              Fine-grained segmentation
 S5   Segment Merging            Merge + Blender coord transform
 S5a  Surface Editor ........    (manual, web editor)
 S6   Wall Generation            Flood-fill from road mask
 S6a  Wall Editor ...........    (manual, web editor)
 S7   Game Object Generation     VLM + procedural hybrid
 S7a  Object Editor .........    (manual, web editor)
 S8   Blender Polygons           JSON -> Mesh + gap filling
 S9   Blender Automation         Tiles + surfaces + textures
 S9a  Manual Blender ........    (manual, Blender editing)
 S10  Model Export               FBX split + KN5 conversion
 S11  Track Packaging            AC folder + UI + metadata
    |
    v
  Assetto Corsa Track Mod
```

Dotted stages are optional manual editing steps.

## Requirements

| Component | Version |
|-----------|---------|
| Python | 3.12+ |
| CUDA | 12.6+ |
| Blender | 5.0+ |
| OS | Windows 10/11 |
| GPU | NVIDIA with CUDA support |

### Python Dependencies

Core libraries (see [`requirements.txt`](requirements.txt) for full list):

- **Deep Learning:** PyTorch (with CUDA), SAM3
- **Geospatial:** rasterio, GDAL
- **Image Processing:** OpenCV, Pillow, scikit-image, NumPy, SciPy
- **AI/LLM:** google-genai (Gemini API)
- **Web:** Flask, Leaflet.js
- **3D Export:** Blender bpy, ksEditorAT (auto-downloaded)

## Installation

### Quick Setup (Windows)

```bash
# Clone the repository
git clone https://github.com/se0lus/sam3_ac_track_mod_gen.git
cd sam3_ac_track_mod_gen

# One-click setup
setup_env.bat
```

### Manual Setup

```bash
# 1. Create conda environment
conda create -n sam3 python=3.12
conda activate sam3

# 2. Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 3. Install project dependencies
pip install -r requirements.txt

# 4. Install SAM3 model package
pip install -e ./sam3/

# 5. Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
```

### Additional Setup

1. **SAM3 Model Weights** &mdash; place `sam3.pt` in `model/sam3.pt`
2. **Blender** &mdash; install [Blender 5.0+](https://www.blender.org/download/) and set the path in config
3. **Gemini API Key** &mdash; copy `.env.example` to `.env` and fill in your key:
   ```bash
   cp .env.example .env
   # Edit .env: GEMINI_API_KEY=your_key_here
   ```

## Usage

### Input Data

You need two types of input data:

- **GeoTIFF orthophoto** &mdash; aerial/drone 2D orthomosaic image with georeference
- **3D Tiles** &mdash; OGC 3D Tiles (b3dm format) from photogrammetry reconstruction, with `tileset.json`

### Run the Full Pipeline

```bash
python script/sam3_track_gen.py \
    --geotiff path/to/orthophoto.tif \
    --tiles-dir path/to/3dtiles/ \
    --output-dir output
```

### Run Individual Stages

Each stage can be run independently:

```bash
# Stage 1: Convert B3DM to GLB
python script/stages/s01_b3dm_convert.py \
    --tiles-dir path/to/3dtiles --output-dir output

# Stage 2: Full-map SAM3 segmentation
python script/stages/s02_mask_full_map.py \
    --geotiff path/to/orthophoto.tif --output-dir output

# Stages 3-11 follow the same pattern
```

### Web Dashboard

Launch the web dashboard for pipeline management and interactive editing:

```bash
python script/webTools/run_webtools.py
# Opens browser automatically -> Dashboard + 6 interactive editors
```

The dashboard provides:
- Pipeline stage execution (single or batch)
- Real-time log streaming via SSE
- Progress bars with ETA
- Output file browser with preview
- Configuration editor

### Interactive Editors

| Editor | Purpose |
|--------|---------|
| **Layout Editor** | Create/manage multiple track layout boundaries |
| **Surface Editor** | Fine-tune per-tag surface masks (road, grass, sand, kerb) |
| **Wall Editor** | Drag-edit wall control points on the map |
| **Object Editor** | Position and orient game objects on the map |
| **Game Object Editor** | Advanced game object management |
| **Centerline Editor** | Draw and adjust track centerline |

### Blender Interactive Mode

For manual 3D editing:

1. Open Blender &rarr; Text Editor &rarr; load `blender_scripts/blender_helpers.py` &rarr; Run
2. Right-click &rarr; **SAM3 Quick Tools** submenu
3. Load Base Tiles &rarr; Refine by Mask &rarr; Extract Surface &rarr; Import Walls/Objects

## Configuration

All configuration is centralized in [`script/pipeline_config.py`](script/pipeline_config.py).

Key configuration areas:

| Category | Fields | Description |
|----------|--------|-------------|
| **Input/Output** | `geotiff_path`, `tiles_dir`, `output_dir` | Data paths |
| **AI/LLM** | `gemini_api_key`, `gemini_model` | Gemini API settings |
| **Track Metadata** | `track_direction`, `track_description` | Track properties |
| **SAM3** | `sam3_prompts` | 8-class segmentation prompts and thresholds |
| **Blender** | `blender_exe`, `base_level`, `target_fine_level` | Blender and LOD settings |
| **Export** | `s10_max_vertices`, `s10_fbx_scale` | FBX/KN5 export parameters |
| **Packaging** | `s11_track_name`, `s11_track_author` | AC track metadata |

## Output

The final output is a complete Assetto Corsa track folder:

```
{track_name}/
├── models_{layout}.ini          (one per layout)
├── {shared}.kn5                 (terrain, collision, environment)
├── go_{layout}.kn5              (game objects per layout)
├── {layout}/
│   ├── map.png                  (minimap)
│   └── data/
│       ├── map.ini
│       └── cameras.ini
└── ui/{layout}/
    ├── ui_track.json            (metadata)
    ├── preview.png
    └── outline.png
```

Copy this folder to `assettocorsa/content/tracks/` to play.

## Tech Stack

| Area | Technology |
|------|-----------|
| AI Segmentation | SAM3 (Segment Anything Model 3, Meta) |
| AI LLM | Google Gemini 2.5 Pro |
| Image Inpainting | Google Gemini 2.5 Flash Image |
| Image Processing | rasterio, Pillow, OpenCV, NumPy, scikit-image |
| Geospatial | WGS84, ECEF, ENU coordinate transforms |
| 3D Engine | Blender 5.0+ (bpy API, headless automation) |
| 3D Data | B3DM / GLB, OGC 3D Tiles 1.0 |
| AC Export | FBX &rarr; KN5 (via ksEditorAT) |
| Web Frontend | Flask, Leaflet.js, Server-Sent Events |
| Runtime | Windows, Python 3.12, CUDA 12.6+, PyTorch |

## Project Structure

```
sam3_track_seg/
├── script/
│   ├── pipeline_config.py          # Centralized configuration
│   ├── sam3_track_gen.py           # Main CLI entry point
│   ├── stages/                     # Pipeline stages (S01-S11)
│   └── webTools/                   # Dashboard + 6 web editors
├── blender_scripts/
│   ├── blender_automate.py         # Headless Blender orchestration
│   ├── blender_helpers.py          # Right-click menu actions
│   └── sam3_actions/               # Blender action plugins
├── sam3/                           # SAM3 model (git submodule)
├── model/                          # Model weights (sam3.pt)
├── ac_toolbox/                     # AC tools and resources
├── tests/                          # Unit tests
├── output/                         # Pipeline output (auto-created)
├── requirements.txt
├── setup_env.bat                   # One-click Windows setup
├── .env.example                    # Environment variable template
├── CLAUDE.md                       # Development rules
└── PROJECT.md                      # Detailed project documentation
```

## Documentation

- **[CLAUDE.md](CLAUDE.md)** &mdash; Development rules and conventions
- **[PROJECT.md](PROJECT.md)** &mdash; Detailed technical documentation (stages, data formats, configuration reference, in Chinese)

## License

All rights reserved.
