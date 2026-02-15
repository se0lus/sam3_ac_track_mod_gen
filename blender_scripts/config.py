"""
blender_scripts 全局配置文件。

所有可配置的路径、级别、集合名称等参数集中于此，
方便在不同环境下统一修改。
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 导入路径 / 环境变量
# ---------------------------------------------------------------------------
# blender_helpers.py 会将该路径插入 sys.path，确保 sam3_actions 包可被正常导入。
# 如果项目目录变更，只需修改此处即可。
SAM3_IMPORT_ROOT_OVERRIDE: str | None = r"E:\sam3_track_seg\blender_scripts"

# 也可通过环境变量指定（优先级次于上面的硬编码路径）。
SAM3_IMPORT_ROOT_ENVVAR: str = "SAM3_BLENDER_SCRIPTS_DIR"

# ---------------------------------------------------------------------------
# 瓦片数据路径
# ---------------------------------------------------------------------------
BASE_TILES_DIR: str = r"E:\sam3_track_seg\test_images_shajing\b3dm"
GLB_DIR: str = r"E:\sam3_track_seg\test_images_shajing\glb"

# ---------------------------------------------------------------------------
# 瓦片级别
# ---------------------------------------------------------------------------
BASE_LEVEL: int = 17
TARGET_FINE_LEVEL: int = 22

# ---------------------------------------------------------------------------
# Blender Collection 名称
# ---------------------------------------------------------------------------
ROOT_CURVE_COLLECTION_NAME: str = "mask_curve2D_collection"
ROOT_POLYGON_COLLECTION_NAME: str = "mask_polygon_collection"

