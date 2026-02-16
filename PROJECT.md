# SAM3 赛道分割项目 — 详细技术文档

## 项目概述

本项目利用 **SAM3（Segment Anything Model 3）** 对赛道航拍/卫星影像（GeoTIFF）进行语义分割，自动识别赛道上的不同地物类型（road、grass、sand、kerb），并将分割结果转换为 **Blender 三维坐标系**中的多边形网格，叠加到倾斜摄影 3D Tiles 模型上，最终自动生成可玩的 **Assetto Corsa 赛道 Mod**。

此外，项目还包含一个独立的 **赛道 Session 分析器**（Track Session Analyzer），用于在地图上可视化 GPS 遥测数据并分析圈速。

---

## 整体工作流

```
GeoTIFF 影像 + 3D Tiles (b3dm)
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  阶段 1: B3DM → GLB 转换                                           │
│  将无人机原始重建的 3D Tiles (b3dm) 转换为 Blender 可导入的 GLB 格式     │
├─────────────────────────────────────────────────────────────────────┤
│  阶段 2: mask_full_map()                                            │
│  对全幅 GeoTIFF 地图做初步 SAM3 分割，生成全图 mask                     │
├─────────────────────────────────────────────────────────────────────┤
│  阶段 3: clip_full_map()                                            │
│  根据 mask 区域智能切分为若干 clip 小图（目标 40m x 40m 瓦片）          │
├─────────────────────────────────────────────────────────────────────┤
│  阶段 4: generate_mask_on_clips()                                   │
│  对每个 clip 逐标签做精细 SAM3 分割 (road/grass/sand/kerb)             │
├─────────────────────────────────────────────────────────────────────┤
│  阶段 5: convert_mask_to_blender_input()                            │
│  读取 3D Tiles tileset.json 坐标原点，将地理坐标多边形转为 Blender 坐标  │
│  同时按类型（road/grass/sand/kerb）合并生成整合 clip 文件               │
├─────────────────────────────────────────────────────────────────────┤
│  阶段 6: blender_create_polygons                                    │
│  在 Blender 中批量读取 *_blender.json，生成 2D Curve + Mesh            │
│  注意：生成的对象已自动转换为 Mesh（非 Curve），以确保后续操作可执行      │
├─────────────────────────────────────────────────────────────────────┤
│  阶段 7: AI 围墙生成                                                │
│  利用 Gemini 大模型分析赛道全图，自动生成虚拟围墙边界 JSON               │
├─────────────────────────────────────────────────────────────────────┤
│  阶段 8: AI 游戏对象生成                                             │
│  利用 Gemini 大模型生成起点/计时点/维修区等游戏功能对象                  │
├─────────────────────────────────────────────────────────────────────┤
│  阶段 9: Blender 无头自动化集成                                       │
│  加载瓦片 → 按 mask 精炼到高级别 → 提取赛道表面 → 导入围墙/对象 → 保存   │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼
  output/09_blender_automate/final_track.blend  →  Assetto Corsa Mod
```

---

## 配置系统

所有配置集中在 `script/pipeline_config.py` 中的 `PipelineConfig` 数据类：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `geotiff_path` | `""` | GeoTIFF 影像路径 |
| `tiles_dir` | `""` | 3D Tiles (b3dm) 目录 |
| `output_dir` | `sam3_track_seg/output` | 输出根目录 |
| `blender_exe` | `E:\SteamLibrary\...\blender.exe` | Blender 可执行文件路径 |
| `gemini_api_key` | 内置 key | Gemini API 密钥 |
| `gemini_model` | `gemini-2.0-flash` | Gemini 模型名称 |
| `track_direction` | `clockwise` | 赛道行驶方向（clockwise/counterclockwise） |
| `track_description` | `""` | 可选的赛道描述（用于 AI 提示词） |
| `base_level` | `17` | 瓦片初始加载级别 |
| `target_fine_level` | `22` | 自动精炼目标级别 |
| `root_curve_collection_name` | `mask_curve2D_collection` | Blender mask 曲线集合 |
| `root_polygon_collection_name` | `mask_polygon_collection` | Blender mask 多边形集合 |
| `collision_collection_name` | `collision` | Blender 碰撞对象集合 |
| `surface_density_road` | `0.5` | 路面采样密度（米） |
| `surface_density_grass` | `2.0` | 草地采样密度（米） |
| `surface_density_kerb` | `0.5` | 路缘采样密度（米） |
| `surface_density_sand` | `2.0` | 砂石采样密度（米） |

### SAM3 分割提示词配置

```python
sam3_prompts = [
    {"tag": "road",  "prompt": "race track surface",  "threshold": 0.4},
    {"tag": "grass", "prompt": "grass",                "threshold": 0.4},
    {"tag": "sand",  "prompt": "sand surface",         "threshold": 0.4},
    {"tag": "kerb",  "prompt": "race track curb",      "threshold": 0.2},
]
```

### 输出目录结构

`PipelineConfig.resolve()` 自动为每个阶段派生输出子目录：

```
output/
├── 01_b3dm_convert/       → glb/ (转换后的 GLB 文件)
├── 02_mask_full_map/      → result_modelscale.png, merged_mask.png, result_masks.json, ...
├── 03_clip_full_map/      → clip_0.tif, clip_1.tif, ..., clip_boxes_visualization.png
├── 04_mask_on_clips/      → road/, grass/, kerb/, sand/ (各标签子目录)
├── 05_convert_to_blender/ → blender JSON 文件 + 按标签合并的 {tag}_clip.json
├── 06_blender_polygons/   → polygons.blend
├── 07_ai_walls/           → walls.json, walls_preview.png
├── 08_ai_game_objects/    → game_objects.json, game_objects_preview.png
└── 09_blender_automate/   → final_track.blend
```

---

## 各模块详解

### 核心流水线 (`script/`)

#### `sam3_track_gen.py` — 主入口

流水线编排脚本，导入所有阶段模块，串联执行。支持：
- 运行全部 9 个阶段
- 通过 `--stage` 参数选择性运行指定阶段
- 每个阶段通过 `stages/sNN_*.py:run(config)` 统一接口调用

#### `pipeline_config.py` — 统一配置

`PipelineConfig` 数据类是所有配置的唯一来源。核心方法：
- `stage_dir(stage_name)` — 返回 `output/NN_stage_name/` 路径
- `resolve()` — 从基础设置派生所有中间路径，调用一次后即可使用

#### `stages/` — 阶段模块

每个阶段文件遵循统一模式：

```python
def run(config: PipelineConfig) -> None:
    """执行本阶段。"""
    ...

if __name__ == "__main__":
    # 独立运行器，便于单独测试
    ...
```

##### 阶段 1: B3DM → GLB 转换 (`s01_b3dm_convert.py`)

将 3D Tiles 的 B3DM 格式文件批量转换为 Blender 可导入的 GLB 格式。B3DM 文件头包含 JSON 头和二进制 glTF 数据，提取后写入 GLB 文件。

- 输入: `test_images_shajing/b3dm/` (含 tileset.json + *.b3dm)
- 输出: `output/01_b3dm_convert/glb/` (保持目录结构的 *.glb 文件)

##### 阶段 2: 全图 SAM3 分割 (`s02_mask_full_map.py`)

对 GeoTIFF 全图执行 SAM3 分割，使用文本提示词 "race track surface" 识别赛道区域。

- 输入: GeoTIFF 影像
- 输出: `output/02_mask_full_map/`
  - `result_modelscale.png` — 模型缩放后的影像
  - `result_mask0_prob(...).png` — 分割 mask 概率图
  - `result_masks.json` — mask 多边形数据
  - `merged_mask.png` — 合并后的二值 mask
  - `results_visualization.png` — 分割可视化

##### 阶段 3: 全图裁剪 (`s03_clip_full_map.py`)

根据阶段 2 的 mask 结果，智能将全图裁剪为 ~40m x 40m 的瓦片。使用贪心覆盖算法，确保所有 mask 像素被覆盖，clip 间允许可控重叠（10%）。

- 输入: GeoTIFF + `output/02_mask_full_map/merged_mask.png`
- 输出: `output/03_clip_full_map/`
  - `clip_0.tif`, `clip_1.tif`, ... — 裁剪后的 GeoTIFF 瓦片
  - `clip_boxes_visualization.png` — 裁剪框可视化

##### 阶段 4: 逐瓦片精细分割 (`s04_mask_on_clips.py`)

对每个 clip 按 4 种标签（road/grass/sand/kerb）分别执行精细 SAM3 分割。

- 输入: `output/03_clip_full_map/clip_*.tif`
- 输出: `output/04_mask_on_clips/`
  - `road/`, `grass/`, `sand/`, `kerb/` 子目录
  - 每个子目录含 `clip_N_masks.json`（包含归一化/像素/地理坐标三套多边形）

##### 阶段 5: Blender 坐标转换 (`s05_convert_to_blender.py`)

读取 3D Tiles 的 `tileset.json` 获取坐标原点信息，将地理坐标多边形经过变换链映射到 Blender 本地坐标：

```
WGS84 (lon, lat, alt) → ECEF (X, Y, Z) → ENU (East, North, Up) → Blender (X, Y, Z)
```

同时按标签类型合并所有 clip，生成整合文件（如 `road_clip.json`），便于后续 Blender 批处理。

- 输入: `output/04_mask_on_clips/` + `tileset.json`
- 输出: `output/05_convert_to_blender/`
  - `{tag}/clip_N_blender.json` — 单个 clip 的 Blender 坐标
  - `{tag}_clip.json` — 按标签合并的整合文件

##### 阶段 6: Blender 多边形生成 (`s06_blender_polygons.py`)

以 Blender 后台模式运行，读取所有 `*_blender.json`，在 Blender 中生成：
- **2D Curve 对象**（调试/可视化用）：按 tag → clip → mask_index → include/exclude 分组
- **Mesh 对象**（最终多边形面）：include 区域扣除 exclude 洞，经 2D Curve 填充 → 转 Mesh → 三角化

生成的 Mesh 对象可直接用于后续的 mask 投影和表面提取。

- 输入: `output/05_convert_to_blender/`
- 输出: `output/06_blender_polygons/polygons.blend`

##### 阶段 7: AI 围墙生成 (`s07_ai_walls.py`)

利用 Gemini 大模型分析赛道全图（模型缩放版），自动生成虚拟围墙边界。围墙设计原则：
- **外围墙**: 围绕整个赛道可行驶区域形成闭合多边形，贴着缓冲区外围（树木/现实围墙处）
- **内围墙**: 避免赛车开进不必要区域
- **地面网格**: 外边缘与外层围墙对齐，内边缘与 road/grass/kerb/sand 碰撞表面对齐，形成托底大表面

围墙在 Blender 中表现为高且无厚度的面，仅用于碰撞检测。

- 输入: `output/02_mask_full_map/result_modelscale.png` + GeoTIFF
- 输出: `output/07_ai_walls/`
  - `walls.json` — 围墙线段数据
  - `walls_preview.png` — 2D 预览图

##### 阶段 8: AI 游戏对象生成 (`s08_ai_game_objects.py`)

利用 Gemini 大模型生成 Assetto Corsa 所需的虚拟游戏功能对象（不可见，无网格）。

对象规范：
- Z 轴为行驶方向，Y 轴朝上
- 高度为赛道表面之上 2 单位
- 支持顺时针/逆时针行驶方向配置

所需对象：

| 对象名 | 数量 | 放置规则 |
|--------|------|----------|
| `AC_HOTLAP_START_0` | 1 | 起点线前一个弯道的出弯处 |
| `AC_PIT_0` ~ `AC_PIT_N` | 8+ | 维修区，沿维修区通道等间距排列 |
| `AC_START_0` ~ `AC_START_N` | 与 PIT 匹配 | 静止起步发车格，在起点线后排列 |
| `AC_TIME_N_L` / `AC_TIME_N_R` | 成对 | 每个组合弯一个计时路段的左右边界 |

- 输入: `output/02_mask_full_map/result_modelscale.png` + GeoTIFF
- 输出: `output/08_ai_game_objects/`
  - `game_objects.json` — 游戏对象数据
  - `game_objects_preview.png` — 2D 预览图

##### 阶段 9: Blender 无头自动化 (`s09_blender_automate.py`)

以 Blender 后台模式运行完整的自动化集成流程：
1. 加载基础 3D 瓦片（base_level 级别）
2. 按 road mask 自动精炼到 target_fine_level
3. 提取赛道表面（按 road/grass/sand/kerb 分别设定采样密度）
4. 导入围墙对象（从阶段 7 的 JSON）
5. 导入游戏对象（从阶段 8 的 JSON）
6. 纹理处理（解包、转换为 PNG、材质转 BSDF）
7. 保存最终 .blend 文件

- 输入: 阶段 1/5/6/7/8 的输出
- 输出: `output/09_blender_automate/final_track.blend`

---

### 工具模块 (`script/`)

#### `geo_tiff_image.py` — GeoTIFF 影像管理

`GeoTiffImage` 类，封装 rasterio 读取 GeoTIFF，提供：
- 影像缩放（按最大尺寸 / 按 GSD 地面分辨率）
- 像素坐标 ↔ 地理坐标双向转换
- 裁剪窗口读取（clip）

#### `geo_sam3_image.py` — SAM3 分割与多边形提取

`GeoSam3Image` 类，SAM3 分割结果管理：
- `convert_mask_to_polygon()` — 二值 mask → 多边形（include 外轮廓 + exclude 内洞），支持 Douglas-Peucker 简化
- `save_masks_to_json_file()` — 导出归一化坐标、像素坐标、地理坐标三套多边形到 JSON
- `save(output_dir=...)` — 支持指定独立输出目录（不污染源目录）

#### `geo_sam3_utils2.py` — 智能 Clip 生成

核心算法：在全图 mask 上用贪心覆盖策略生成最少数量的 clip 框，确保：
- 所有 mask 像素都被至少一个 clip 覆盖
- clip 之间允许可控重叠
- 避免 clip 边界切穿 mask 核心区域

#### `geo_sam3_blender_utils.py` — 地理 → Blender 坐标转换

读取 3D Tiles 的 `tileset.json` 获取坐标原点，将 WGS84 地理坐标经过 ECEF → ENU 变换映射到 Blender 本地坐标。支持：
- 单点/批量坐标转换
- `consolidate_clips_by_tag()` — 按标签类型合并所有 clip 到整合文件

#### `b3dm_converter.py` — B3DM/GLB 格式转换

解析 B3DM 文件头，提取内嵌的 glTF 二进制数据，写入标准 GLB 文件。支持批量目录转换。

#### `gemini_client.py` — Gemini API 客户端

封装 Google Gemini API 调用，支持图像 + 文本多模态输入，JSON 格式化输出。

#### `ai_wall_generator.py` — AI 围墙生成逻辑

基于赛道全图缩放版 + mask 信息，构建提示词让 Gemini 生成围墙边界线段。

#### `ai_game_objects.py` — AI 游戏对象生成逻辑

基于赛道全图 + mask + 行驶方向，构建提示词让 Gemini 生成游戏功能对象的位置和朝向。

#### `surface_extraction.py` — 赛道表面提取

按 mask 多边形在 Y 方向上投影到 3D 瓦片表面，重建出赛道碰撞表面。边缘精确匹配 mask 边界。

---

### Blender 端 (`blender_scripts/`)

#### `config.py` — Blender 配置

从 `pipeline_config.py` 导入默认值的薄层模块。运行时由 `blender_automate.py` 通过命令行参数覆盖。保持与 Blender Actions 的兼容接口。

#### `blender_create_polygons.py` — 批处理 Polygon 生成

以 Blender 后台批处理模式运行，读取 `*_blender.json`：
- 按 tag → clip → mask_index → include/exclude 分组生成 2D Curve
- 自动转换为 Mesh 并三角化
- 保存为 .blend 文件

#### `blender_automate.py` — 无头自动化

串联所有 Blender 操作的无头脚本，接收命令行参数覆盖 config.py 的默认值。执行流程：加载瓦片 → 精炼 → 表面提取 → 围墙导入 → 游戏对象导入 → 纹理处理 → 保存。

#### `blender_helpers.py` — 右键菜单框架

提供 Blender Add-on / Script 两种使用方式，自动发现 `sam3_actions/` 包下的 Action 模块，注册到 View3D / Outliner 右键菜单。支持幂等注册和模块热重载。

#### `sam3_actions/` — 交互式操作

| 模块 | 菜单项 | 功能 |
|------|--------|------|
| `load_base_tiles.py` | Load Base Tiles | 从 tileset.json 加载指定级别的 glb 瓦片 |
| `load_base_tiles.py` | Refine Selected Tiles | 加载更精细的下一级子瓦片 |
| `load_base_tiles.py` | Refine by Mask to Target Level | 按 mask 多边形自动框选并逐步精炼到目标级别 |
| `surface_extractor.py` | Extract Surface | 按 mask 投影提取赛道表面网格 |
| `import_walls.py` | Import Walls | 从 walls.json 导入围墙对象 |
| `import_game_objects.py` | Import Game Objects | 从 game_objects.json 导入游戏功能对象 |
| `texture_tools.py` | Unpack & Convert Textures | 解包纹理、转 PNG、转 BSDF 材质 |
| `clear_scene.py` | Clear Scene | 清除场景（保留 mask 集合和 script 对象） |

##### `c_tiles.py` — CTile 瓦片树

`CTile` 类递归解析 3D Tiles 的 `tileset.json`，构建瓦片树结构。支持按名称查找、获取子瓦片、LOD 级别管理。

##### `mask_select_utils.py` — Mask 相交测试

XZ 平面高效相交测试：三角化 + AABB 预筛选 + 精确几何测试的三级加速策略。

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

### 围墙 JSON (`walls.json`)

```json
{
  "walls": [
    {
      "type": "outer",
      "points": [[x1, y1], [x2, y2], ...]
    }
  ]
}
```

### 游戏对象 JSON (`game_objects.json`)

```json
{
  "objects": [
    {
      "name": "AC_HOTLAP_START_0",
      "position": [x, y, z],
      "rotation": [rx, ry, rz]
    }
  ]
}
```

### 3D Tiles (`tileset.json`)

标准 3D Tiles 1.0 格式，包含瓦片树结构、bounding volume、LOD 级别（L13-L23）。

---

## 技术栈

| 领域 | 技术 |
|------|------|
| AI 分割模型 | SAM3 (Segment Anything Model 3, Meta) |
| AI 大模型 | Google Gemini (gemini-2.0-flash) |
| 影像处理 | rasterio, PIL/Pillow, OpenCV, numpy |
| 地理坐标 | pyproj / 手动 WGS84 ↔ ECEF ↔ ENU 转换 |
| 三维可视化 | Blender 3.0+ (bpy API) |
| 三维数据 | 3D Tiles (b3dm/glb), tileset.json |
| Web 可视化 | Leaflet.js, HTML/CSS/JS |
| 运行环境 | Windows, Python 3.10+, Blender Python |

---

## 快速开始

### 1. 运行完整流水线

```bash
python script/sam3_track_gen.py \
    --geotiff test_images_shajing/result.tif \
    --tiles-dir test_images_shajing/b3dm \
    --output-dir output \
    --blender-exe "E:\SteamLibrary\steamapps\common\Blender\blender.exe"
```

### 2. 单独运行各阶段

```bash
# 阶段1: B3DM → GLB 转换
python script/stages/s01_b3dm_convert.py \
    --tiles-dir test_images_shajing/b3dm --output-dir output

# 阶段2: 全图 SAM3 分割
python script/stages/s02_mask_full_map.py \
    --geotiff test_images_shajing/result.tif --output-dir output

# 阶段3: 裁剪全图为瓦片
python script/stages/s03_clip_full_map.py \
    --geotiff test_images_shajing/result.tif --output-dir output

# 阶段4: 逐瓦片精细分割
python script/stages/s04_mask_on_clips.py \
    --geotiff test_images_shajing/result.tif --output-dir output

# 阶段5: 转换为 Blender 坐标
python script/stages/s05_convert_to_blender.py \
    --geotiff test_images_shajing/result.tif \
    --tiles-dir test_images_shajing/b3dm --output-dir output

# 阶段6: Blender 多边形生成
python script/stages/s06_blender_polygons.py --output-dir output

# 阶段7: AI 围墙生成
python script/stages/s07_ai_walls.py \
    --geotiff test_images_shajing/result.tif --output-dir output

# 阶段8: AI 游戏对象生成
python script/stages/s08_ai_game_objects.py \
    --geotiff test_images_shajing/result.tif --output-dir output

# 阶段9: Blender 自动化集成
python script/stages/s09_blender_automate.py \
    --tiles-dir test_images_shajing/b3dm --output-dir output
```

### 3. 在 Blender 中交互式操作

1. 打开 Blender，在 Text Editor 中打开 `blender_helpers.py` 并运行
2. 右键菜单出现 "SAM3 Quick Tools" 子菜单
3. 使用 "Load Base Tiles" 加载底图瓦片
4. 选中 mask polygon 对象后，使用 "Refine by Mask to Target Level" 自动精细化
5. 使用 "Extract Surface" 提取赛道表面
6. 使用 "Import Walls" / "Import Game Objects" 导入 AI 生成的数据
7. 使用 "Unpack & Convert Textures" 处理纹理

### 4. 赛道 Session 分析器

```bash
python script/track_session_anaylzer/run_analyzer.py
# 自动打开浏览器，拖入 RaceChrono CSV 文件即可可视化
```

功能特点：支持本地瓦片地图、多数据集叠加对比、独立偏移控制、自动圈速检测。

---

## 设计原则

1. **统一配置**: 所有配置集中在 `pipeline_config.py`，`blender_scripts/config.py` 仅作为薄层导入
2. **阶段独立**: 每个阶段一个文件，可通过 `__main__` 独立运行
3. **输出隔离**: 每个阶段有独立的输出子目录（`output/NN_stage_name/`），不污染源数据
4. **阶段间通信**: 下游阶段从上游阶段的输出目录读取文件，不依赖内存状态
5. **可测试性**: Blender 耦合代码与纯 Python 逻辑分离，支持脱离 Blender 测试
