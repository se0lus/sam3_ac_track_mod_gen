"""
blender_scripts 全局配置文件。

Thin shim that provides module-level constants for Blender action modules.
Defaults are sourced from ``script/pipeline_config.py`` when available;
values can be overridden at runtime (e.g. by ``blender_automate.py``)
before the action modules are imported.
"""

from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# Try to import unified config for defaults
# ---------------------------------------------------------------------------
_script_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "script")
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

try:
    from pipeline_config import PipelineConfig as _PC
    _defaults = _PC()
except Exception:
    _defaults = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# 导入路径 / 环境变量
# ---------------------------------------------------------------------------
SAM3_IMPORT_ROOT_OVERRIDE: str | None = os.path.dirname(os.path.abspath(__file__))
SAM3_IMPORT_ROOT_ENVVAR: str = "SAM3_BLENDER_SCRIPTS_DIR"

# ---------------------------------------------------------------------------
# Blender 可执行文件路径
# ---------------------------------------------------------------------------
BLENDER_EXE: str = _defaults.blender_exe if _defaults else r"E:\SteamLibrary\steamapps\common\Blender\blender.exe"

# ---------------------------------------------------------------------------
# 瓦片数据路径 (overridden at runtime by blender_automate.py)
# ---------------------------------------------------------------------------
BASE_TILES_DIR: str = ""
GLB_DIR: str = ""

# ---------------------------------------------------------------------------
# 瓦片级别
# ---------------------------------------------------------------------------
BASE_LEVEL: int = _defaults.base_level if _defaults else 17
TARGET_FINE_LEVEL: int = _defaults.target_fine_level if _defaults else 22

# ---------------------------------------------------------------------------
# Blender Collection 名称
# ---------------------------------------------------------------------------
ROOT_CURVE_COLLECTION_NAME: str = _defaults.root_curve_collection_name if _defaults else "mask_curve2D_collection"
ROOT_POLYGON_COLLECTION_NAME: str = _defaults.root_polygon_collection_name if _defaults else "mask_polygon_collection"

# ---------------------------------------------------------------------------
# Surface extraction / collision mesh
# ---------------------------------------------------------------------------
SURFACE_SAMPLING_DENSITY_ROAD: float = _defaults.surface_density_road if _defaults else 0.5
SURFACE_SAMPLING_DENSITY_GRASS: float = _defaults.surface_density_grass if _defaults else 2.0
SURFACE_SAMPLING_DENSITY_KERB: float = _defaults.surface_density_kerb if _defaults else 0.5
SURFACE_SAMPLING_DENSITY_SAND: float = _defaults.surface_density_sand if _defaults else 2.0
SURFACE_SAMPLING_DENSITY_DEFAULT: float = _defaults.surface_density_default if _defaults else 1.0

COLLISION_COLLECTION_NAME: str = _defaults.collision_collection_name if _defaults else "collision"

# ---------------------------------------------------------------------------
# Consolidated clips directory (overridden at runtime by blender_automate.py)
# ---------------------------------------------------------------------------
CONSOLIDATED_CLIPS_DIR: str = ""

# Clean up module namespace
del _defaults
