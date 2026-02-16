# SAM3 Track Segmentation Project

## 项目概述

本项目利用 **SAM3（Segment Anything Model 3）** 对赛道航拍/卫星影像（GeoTIFF）进行语义分割，自动识别赛道上的不同地物类型（road、grass、sand、kerb），并将分割结果转换为 **Blender 三维坐标系**中的多边形网格，叠加到倾斜摄影 3D Tiles 模型上，实现从"二维影像分割"到"三维场景标注"的完整工作流。

此外，项目还包含一个独立的 **赛道 Session 分析器**（Track Session Analyzer），用于在地图上可视化 GPS 遥测数据并分析圈速。

---

## 整体工作流

```
GeoTIFF 影像
    │
    ▼
┌─────────────────────────────┐
│  1. mask_full_map()         │  对全幅地图做初步 SAM3 分割，生成全图 mask
│  2. clip_full_map()         │  将 mask 区域智能切分为若干 clip 小图
│  3. generate_mask_on_clips()│  对每个 clip 逐标签做精细 SAM3 分割
└─────────────────────────────┘
    │  输出: clips/{tag}/clip_N_masks.json （像素/地理坐标多边形）
    ▼
┌─────────────────────────────┐
│  4. convert_mask_to_blender │  读取 3D Tiles tileset.json 中的坐标原点，
│     _input()                │  将地理坐标多边形转换为 Blender 本地坐标
└─────────────────────────────┘
    │  输出: blender_clips/{tag}/clip_N_blender.json
    ▼
┌─────────────────────────────┐
│  5. blender_create_polygons │  在 Blender 中批量读取 *_blender.json，
│     .py                     │  生成 2D Curve + Mesh，保存为 .blend 文件
└─────────────────────────────┘
    │  输出: polygons.blend
    ▼
┌─────────────────────────────┐
│  6. Blender 交互式操作       │  在 Blender 中加载 3D Tiles (glb)，
│     (blender_helpers +      │  将 mask polygon 叠加到三维模型上，
│      sam3_actions/*)        │  通过右键菜单进行交互式选择/refine
└─────────────────────────────┘
```

---

## 目录结构

```
sam3_track_seg/
│
├── sam3/                          # SAM3 模型源码（Meta 官方，子模块/第三方）
│
├── model/                         # SAM3 模型权重与 tokenizer 配置
│   ├── config.json
│   ├── tokenizer.json
│   └── ...
│
├── script/                        # 核心处理脚本（Python 命令行）
│   ├── sam3_track_gen.py          # 主流水线入口
│   ├── geo_tiff_image.py          # GeoTIFF 影像读取/缩放/坐标转换
│   ├── geo_sam3_image.py          # SAM3 分割 + mask → polygon 转换
│   ├── geo_sam3_utils.py          # clip box 智能生成算法 v1
│   ├── geo_sam3_utils2.py         # clip box 智能生成算法 v2（改进版）
│   ├── geo_sam3_blender_utils.py  # 地理坐标 → Blender 坐标转换
│   └── track_session_anaylzer/    # 赛道 Session 分析器（独立 Web 工具）
│       ├── run_analyzer.py
│       ├── index.html
│       ├── app.js
│       └── style.css
│
├── blender_scripts/               # Blender 端脚本
│   ├── config.py                  # 全局配置（路径、级别、集合名称）
│   ├── blender_create_polygons.py # 批处理：JSON → Blender Curve/Mesh
│   ├── blender_helpers.py         # 右键菜单框架（Add-on / Script 两用）
│   └── sam3_actions/              # 右键菜单 Action 插件包
│       ├── __init__.py            # ActionSpec 定义
│       ├── c_tiles.py             # CTile 类：3D Tiles tileset 树管理
│       ├── load_base_tiles.py     # Action: 加载 glb 瓦片到 Blender
│       ├── clear_scene.py         # Action: 清除场景（保留 mask 对象）
│       ├── mask_select_utils.py   # 工具: mask XZ 投影相交测试
│       ├── selected_obj_by_mask_polygons.py  # Action: 用 mask 框选对象
│       └── print_selected_names.py           # Action: 打印选中对象名
│
└── test_images_shajing/           # 测试数据集（沙井赛道）
    ├── b3dm/                      # 3D Tiles (b3dm + tileset.json)
    ├── result.tif                 # 2D 赛道全图（很巨大，使用之前需要先降采样）
```

---

## 各模块详解

### 1. `script/` — 核心处理流水线

#### `sam3_track_gen.py` — 主入口

整个流水线的编排脚本，串联以下步骤：

| 函数 | 功能 |
|------|------|
| `mask_full_map()` | 对 GeoTIFF 全图做 SAM3 分割，输出全局 mask 图像 |
| `clip_full_map()` | 根据 mask 区域智能切分为不重叠/低重叠的 clip 小图 |
| `generate_mask_on_clips()` | 对每个 clip 按标签（road/grass/sand/kerb）分别做精细分割 |
| `convert_mask_to_blender_input()` | 批量将 mask JSON 转为 Blender 坐标格式 |

#### `geo_tiff_image.py` — GeoTIFF 影像管理

`GeoTiffImage` 类，封装 rasterio 读取 GeoTIFF，提供：
- 影像缩放（按最大尺寸 / 按 GSD 地面分辨率）
- 像素坐标 ↔ 地理坐标双向转换
- 裁剪窗口读取（clip）

#### `geo_sam3_image.py` — SAM3 分割与多边形提取

`GeoSam3Image` 类，将 SAM3 分割结果管理起来：
- `convert_mask_to_polygon()`: 二值 mask → 多边形（include 外轮廓 + exclude 内洞），支持 Douglas-Peucker 简化
- `save_masks_to_json_file()`: 导出归一化坐标、像素坐标、地理坐标三套多边形到 JSON
- 可视化辅助函数

#### `geo_sam3_utils.py` / `geo_sam3_utils2.py` — 智能 Clip 生成

核心算法：在全图 mask 上用贪心覆盖策略生成最少数量的 clip 框，使得：
- 所有 mask 像素都被至少一个 clip 覆盖
- clip 之间允许可控重叠
- 避免 clip 边界切穿 mask 核心区域
- v2 版本在重叠处理和评分策略上做了改进

#### `geo_sam3_blender_utils.py` — 地理 → Blender 坐标转换

读取 3D Tiles 的 `tileset.json` 获取坐标原点信息，将 WGS84 地理坐标经过以下变换链映射到 Blender 本地坐标：

```
WGS84 (lon, lat, alt)
  → ECEF (X, Y, Z)
    → ENU (East, North, Up) 相对于 tileset 原点
      → Blender 坐标 (X, Y, Z)
```

输出 `*_blender.json`，包含 `points_xyz` 字段（Blender 坐标系下的多边形顶点）。

---

### 2. `blender_scripts/` — Blender 端

#### `config.py` — 全局配置

集中管理所有可配置参数，方便切换数据集/环境：

| 配置项 | 说明 |
|--------|------|
| `SAM3_IMPORT_ROOT_OVERRIDE` | Blender Python 导入根路径 |
| `BASE_TILES_DIR` | 3D Tiles (b3dm) 目录 |
| `GLB_DIR` | glb 瓦片目录 |
| `BASE_LEVEL` / `TARGET_FINE_LEVEL` | 瓦片加载/精细化级别 |
| `ROOT_CURVE_COLLECTION_NAME` | mask 曲线集合名称 |
| `ROOT_POLYGON_COLLECTION_NAME` | mask 多边形集合名称 |

#### `blender_create_polygons.py` — 批处理 Polygon 生成

以 Blender 后台批处理模式运行，读取所有 `*_blender.json`，在 Blender 中生成：
- **2D Curve 对象**（用于调试/可视化）：按 tag → clip → mask_index → include/exclude 分组
- **Mesh 对象**（最终多边形面）：include 区域扣除 exclude 洞，经 2D Curve 填充 → 转 Mesh → 三角化

运行方式：
```powershell
blender.exe --background --python blender_create_polygons.py -- --input <blender_clips_dir> --output <output.blend>
```

#### `blender_helpers.py` — 右键菜单框架

提供 Blender Add-on / Script 两种使用方式，自动发现 `sam3_actions/` 包下的所有 Action 模块，并注册到 View3D / Outliner 右键菜单中。支持：
- 幂等注册（重复运行不会累积菜单项）
- 多来源选择兼容（View3D selected_objects + Outliner selected_ids）
- 模块热重载

#### `sam3_actions/` — 交互式操作

| 模块 | 菜单项 | 功能 |
|------|--------|------|
| `load_base_tiles.py` | Load Base Tiles | 从 tileset.json 加载指定级别的 glb 瓦片到 Blender，非阻塞 modal 执行 |
| `load_base_tiles.py` | Refine Selected Tiles | 对选中的瓦片加载更精细的下一级子瓦片 |
| `load_base_tiles.py` | Refine by Mask to Target Level | 根据 mask 多边形自动框选瓦片并逐步 refine 到目标级别 |
| `clear_scene.py` | Clear Scene (Keep mask_ & script) | 清除场景中除 mask 集合和 script 对象外的所有对象及孤儿数据块 |
| `selected_obj_by_mask_polygons.py` | Select Objects By Mask XZ Polygons | 以 mask 多边形在 XZ 平面的投影框选场景中的瓦片对象 |
| `print_selected_names.py` | Print Selected Object Names | 调试用，打印选中对象名称 |

##### `c_tiles.py` — CTile 瓦片树

`CTile` 类递归解析 3D Tiles 的 `tileset.json`，构建瓦片树结构，支持：
- 按名称查找瓦片 (`find`)
- 获取所有子瓦片 (`allChildren`)
- 判断是否可 refine (`canRefine`, `canRefineBy`)
- LOD 级别管理 (`meshLevel`)

##### `mask_select_utils.py` — Mask 相交测试工具

提供高效的 XZ 平面相交测试，用于判断场景对象的 bounding box 是否与 mask 多边形在俯视投影下重叠。使用三角化 + AABB 预筛选 + 精确几何测试的三级加速策略。

---

### 3. `script/track_session_anaylzer/` — 赛道 Session 分析器

独立的 Web 工具，用于可视化赛道 GPS 遥测数据（RaceChrono CSV 格式）。

| 文件 | 功能 |
|------|------|
| `run_analyzer.py` | Python HTTP 服务器启动器，自动打开浏览器 |
| `index.html` | 页面结构 |
| `app.js` | 核心逻辑：Leaflet 地图、CSV 解析、圈速检测、轨迹可视化、偏移控制 |
| `style.css` | 样式 |

功能特点：
- 支持本地瓦片地图
- 多数据集叠加对比
- 每个数据集可独立偏移（东/北方向，1m/0.1m 步长）
- 自动圈速检测与统计

---

## 关键数据流格式

### Mask JSON (`clip_N_masks.json`)

```json
{
  "masks": [
    {
      "tag": "road",
      "mask_index": 0,
      "prob": 0.95,
      "polygons": {
        "include": [
          { "points_normalized": [[x,y], ...], "points_pixel": [...], "points_geo": [[lon,lat], ...] }
        ],
        "exclude": [...]
      }
    }
  ]
}
```

### Blender JSON (`clip_N_blender.json`)

```json
{
  "polygons": {
    "include": [
      { "tag": "road", "mask_index": 0, "prob": 0.95, "points_xyz": [[x,y,z], ...] }
    ],
    "exclude": [...]
  }
}
```

### 3D Tiles (`tileset.json`)

标准 3D Tiles 1.0 格式，包含瓦片树结构、bounding volume、LOD 级别（L13-L23）。

---

## 技术栈

| 领域 | 技术 |
|------|------|
| AI 分割模型 | SAM3 (Segment Anything Model 3, Meta) |
| 影像处理 | rasterio, PIL/Pillow, OpenCV, numpy |
| 地理坐标 | pyproj / 手动 WGS84↔ECEF↔ENU 转换 |
| 三维可视化 | Blender 3.0+ (bpy API) |
| 三维数据 | 3D Tiles (b3dm/glb), tileset.json |
| Web 可视化 | Leaflet.js, HTML/CSS/JS |
| 运行环境 | Windows, Python 3.10+, Blender Python |

---

## 快速开始

### 1. 运行分割流水线

```python
# 在 script/ 目录下
from sam3_track_gen import mask_full_map, clip_full_map, generate_mask_on_clips, convert_mask_to_blender_input

# Step 1: 全图 mask
mask_full_map(source_tiff, output_dir)

# Step 2: 生成 clips
clip_full_map(source_tiff, mask_path, clips_dir)

# Step 3: 逐标签精细分割
generate_mask_on_clips(clips_dir, tags=["road", "grass", "sand", "kerb"])

# Step 4: 转换为 Blender 坐标
convert_mask_to_blender_input(clips_dir, tileset_json, blender_clips_dir)
```

### 2. 生成 Blender Polygon

```powershell
blender.exe --background --python blender_scripts\blender_create_polygons.py -- `
  --input E:\sam3_track_seg\test_images_shajing\blender_clips `
  --output E:\sam3_track_seg\output\polygons.blend
```

### 3. 在 Blender 中交互式操作

1. 打开 Blender，在 Text Editor 中打开 `blender_helpers.py` 并运行
2. 右键菜单出现 "SAM3 Quick Tools" 子菜单
3. 使用 "Load Base Tiles" 加载底图瓦片
4. 选中 mask polygon 对象后，使用 "Refine by Mask to Target Level" 自动精细化

### 4. 赛道 Session 分析器

```powershell
python script\track_session_anaylzer\run_analyzer.py
# 自动打开浏览器，拖入 RaceChrono CSV 文件即可可视化
```

---

## 配置说明

所有 Blender 端的可配置参数集中在 `blender_scripts/config.py` 中。切换数据集时只需修改此文件：

```python
# 瓦片数据路径
BASE_TILES_DIR = r"E:\sam3_track_seg\test_images_shajing\b3dm"
GLB_DIR = r"E:\sam3_track_seg\test_images_shajing\glb"

# 瓦片级别
BASE_LEVEL = 17           # 初始加载级别
TARGET_FINE_LEVEL = 22    # 自动 refine 目标级别
```

---