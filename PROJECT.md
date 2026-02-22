# SAM3 赛道分割项目 — 详细技术文档

## 项目概述

本项目利用 **SAM3（Segment Anything Model 3）** 对赛道航拍/卫星影像（GeoTIFF）进行语义分割，自动识别赛道上的不同地物类型（road、grass、sand、kerb、trees、building、water、concrete），并将分割结果转换为 **Blender 三维坐标系**中的多边形网格，叠加到倾斜摄影 3D Tiles 模型上，最终自动生成可玩的 **Assetto Corsa 赛道 Mod**。

项目还包含一套完整的 **Web 交互编辑器**（基于 Leaflet.js），可在地图上手动精调 AI 生成的中线、围墙、游戏对象等数据。

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
│  对全幅 GeoTIFF 地图做 SAM3 分割（8 类标签），生成全图 mask              │
│  同时生成 modelscale (~1008px) 和 vlmscale (~3072px) 两种分辨率影像     │
│  自动检测并修复航拍影像中心孔洞（Gemini 图像修复）                       │
├─────────────────────────────────────────────────────────────────────┤
│  阶段 2a: 赛道布局管理（可选，手动）                                    │
│  通过 Web 编辑器创建/管理多赛道布局，生成 per-layout 二值 mask           │
├─────────────────────────────────────────────────────────────────────┤
│  阶段 3: clip_full_map()                                            │
│  根据 mask 区域智能切分为若干 clip 小图（目标 40m x 40m 瓦片）          │
├─────────────────────────────────────────────────────────────────────┤
│  阶段 4: generate_mask_on_clips()                                   │
│  对每个 clip 逐标签做精细 SAM3 分割                                    │
│  road 支持回退提示词 (asphalt road / concrete road) 确保 100% 覆盖     │
├─────────────────────────────────────────────────────────────────────┤
│  阶段 5: merge_segments()                                            │
│  多瓦片分割结果合并 + 优先级合成 + 地理坐标 → Blender 坐标转换          │
│  消除瓦片重叠、tag 间间隙，生成统一的 per-tag blender JSON              │
├─────────────────────────────────────────────────────────────────────┤
│  阶段 6: 程序化围墙生成                                               │
│  SAM3 洪水填充算法：从 road → 穿越 driveable 区域 → 到达障碍物边界       │
│  无 LLM 依赖，纯 mask 几何运算                                        │
├─────────────────────────────────────────────────────────────────────┤
│  阶段 7: 混合游戏对象生成                                              │
│  VLM (Gemini 2.5 Pro) 生成布局对象 (hotlap/pit/start)                 │
│  程序化中线分析生成计时点 (timing)                                      │
│  snap-to-road 后处理确保 100% 验证通过                                 │
├─────────────────────────────────────────────────────────────────────┤
│  阶段 7a: 手动游戏对象编辑（可选）                                      │
│  通过 Web 编辑器精调 AI 生成的游戏对象位置和朝向                        │
├─────────────────────────────────────────────────────────────────────┤
│  阶段 8: blender_create_polygons                                    │
│  在 Blender 中批量读取 *_blender.json，生成 2D Curve + Mesh            │
│  注意：生成的对象已自动转换为 Mesh（非 Curve），以确保后续操作可执行      │
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
| `blender_exe` | `C:\Program Files\...\blender.exe` | Blender 可执行文件路径 |
| `gemini_api_key` | 内置 key | Gemini API 密钥 |
| `gemini_model` | `gemini-2.5-pro` | Gemini VLM 模型名称 |
| `inpaint_center_holes` | `True` | 是否修复航拍影像中心孔洞 |
| `inpaint_model` | `gemini-3-pro-image-preview` | 图像修复模型 |
| `inpaint_min_hole_ratio` | `0.001` | 孔洞面积最小阈值（0.1%） |
| `vlm_max_size` | `3072` | VLM 输入图像最大维度（Gemini 支持 3072） |
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
| `surface_density_default` | `1.0` | 默认采样密度（米） |

### SAM3 分割提示词配置

8 类语义标签，road 支持回退提示词机制确保 100% 覆盖率：

```python
sam3_prompts = [
    {"tag": "road",     "prompt": "race track surface",     "threshold": 0.25,
     "fallback_prompts": ["asphalt road", "concrete road"]},
    {"tag": "grass",    "prompt": "grass",                   "threshold": 0.4},
    {"tag": "sand",     "prompt": "sand surface",            "threshold": 0.4},
    {"tag": "kerb",     "prompt": "race track curb",         "threshold": 0.2},
    {"tag": "trees",    "prompt": "forest",                  "threshold": 0.3},
    {"tag": "building", "prompt": "building structure",      "threshold": 0.4},
    {"tag": "water",    "prompt": "water pond",              "threshold": 0.4},
    {"tag": "concrete", "prompt": "concrete paved ground",   "threshold": 0.4},
]

# 全图分割使用的标签（阶段 2）
sam3_fullmap_tags = ["road", "trees", "grass", "kerb", "sand", "building", "water", "concrete"]
```

**回退机制**：当主提示词在某个 clip 上完全无法检测到任何有效 mask 时，系统按顺序尝试 `fallback_prompts` 列表中的替代提示词。实测 road 标签在沙井赛道 43 个 clip 中，主提示词覆盖 36 个，回退提示词 "asphalt road" 恢复剩余 7 个（置信度 0.85-0.93），达到 100% 覆盖。

### 输出目录结构

`PipelineConfig.resolve()` 自动为每个阶段派生输出子目录：

```
output/
├── 01_b3dm_convert/       → glb/ (转换后的 GLB 文件)
├── 02_mask_full_map/      → result_modelscale.png, result_vlmscale.png,
│                            merged_mask.png, {tag}_mask.png, result_masks.json, ...
├── 02a_track_layouts/     → layouts.json, per-layout 二值 mask（可选）
├── 03_clip_full_map/      → clip_0.tif, clip_1.tif, ..., clip_boxes_visualization.png
├── 04_mask_on_clips/      → road/, grass/, kerb/, sand/, trees/, building/, water/, concrete/
├── 05_merge_segments/    → 合并分割结果 + 按标签合并的 blender JSON
├── 06_ai_walls/           → walls.json, walls_preview.png
├── 07_ai_game_objects/    → geo_metadata.json, {LayoutName}/game_objects.json,
│                            {LayoutName}/centerline.json, {LayoutName}/game_objects_preview.png
├── 07a_manual_game_objects/ → game_objects.json（手动精调，可选）
├── 08_blender_polygons/   → polygons.blend
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

对 GeoTIFF 全图执行 SAM3 分割，使用 8 类文本提示词识别不同地物类型。同时生成两种分辨率的影像用于后续阶段：

- **modelscale** (~1008px): 用于 mask 坐标系和下游处理
- **vlmscale** (~3072px): 高分辨率图像供 Gemini VLM 分析使用

两种图像均经过中心孔洞检测和 Gemini 图像修复。

- 输入: GeoTIFF 影像
- 输出: `output/02_mask_full_map/`
  - `result_modelscale.png` — 模型缩放后的影像 (~1008px)
  - `result_vlmscale.png` — VLM 高分辨率影像 (~3072px)
  - `merged_mask.png` — 合并后的 road 二值 mask
  - `{tag}_mask.png` — 每类标签的二值 mask（trees_mask.png, grass_mask.png 等）
  - `result_masks.json` — mask 多边形数据
  - `results_visualization.png` — 分割可视化

##### 阶段 2a: 赛道布局管理 (`s02a_track_layouts.py`) — 可选

手动阶段，通过 Web 编辑器（Layout Editor）创建和管理多赛道布局。每个布局对应一套独立的游戏对象配置。不参与自动流水线。

- 输出: `output/02a_track_layouts/layouts.json`

##### 阶段 3: 全图裁剪 (`s03_clip_full_map.py`)

根据阶段 2 的 mask 结果，智能将全图裁剪为 ~40m x 40m 的瓦片。使用贪心覆盖算法，确保所有 mask 像素被覆盖，clip 间允许可控重叠（10%）。

- 输入: GeoTIFF + `output/02_mask_full_map/merged_mask.png`
- 输出: `output/03_clip_full_map/`
  - `clip_0.tif`, `clip_1.tif`, ... — 裁剪后的 GeoTIFF 瓦片
  - `clip_boxes_visualization.png` — 裁剪框可视化

##### 阶段 4: 逐瓦片精细分割 (`s04_mask_on_clips.py`)

对每个 clip 按 8 种标签分别执行精细 SAM3 分割。

**回退提示词机制**：当主提示词在某个 clip 上未产生任何有效 mask 时，按顺序尝试 `fallback_prompts` 列表中的替代提示词，直到检测成功。例如 road 标签的 "race track surface" 无法检测时，自动回退到 "asphalt road" 重试。

- 输入: `output/03_clip_full_map/clip_*.tif`
- 输出: `output/04_mask_on_clips/`
  - `road/`, `grass/`, `sand/`, `kerb/`, `trees/`, `building/`, `water/`, `concrete/` 子目录
  - 每个子目录含 `clip_N_masks.json`（包含归一化/像素/地理坐标三套多边形）

##### 阶段 5: 合并分割 (`s05_merge_segments.py`)

将多瓦片 SAM3 分割结果合并为统一的 per-tag mask，并转换为 Blender 坐标：

1. 将各 clip 的分割多边形光栅化到共享画布，消除重叠
2. 优先级合成（sand < grass < road2 < road < kerb）消除 tag 间间隙
3. 读取 `tileset.json` 将地理坐标映射到 Blender 本地坐标

- 输入: `output/04_mask_on_clips/` + `tileset.json`
- 输出: `output/05_merge_segments/`
  - `{tag}/{tag}_merged_blender.json` — 合并后的 Blender 坐标
  - `merge_preview/` — 合成预览图

##### 阶段 6: 程序化围墙生成 (`s06_ai_walls.py`)

**纯程序化实现，无 LLM 依赖。** 使用 SAM3 mask 进行洪水填充算法：

1. 从 road mask 出发，穿越可行驶区域（grass = 缓冲区，包含在外墙内）
2. 遇到障碍物（trees、building、water）停止扩展
3. 生成外围墙轮廓 + 障碍物围墙

设计原则：
- **草地可行驶**：外墙包含草地缓冲区
- **树木不可行驶**：外墙在树木边缘停止
- **赛道邻近约束**（80px）：防止围墙包含远离赛道的建筑物

- 输入: `output/02_mask_full_map/` (各标签 mask) + GeoTIFF
- 输出: `output/06_ai_walls/`
  - `walls.json` — 围墙线段数据
  - `walls_preview.png` — 2D 预览图

##### 阶段 7: 混合游戏对象生成 (`s07_ai_game_objects.py`)

采用 **VLM + 程序化** 混合策略生成 Assetto Corsa 游戏功能对象：

**VLM 部分**（Gemini 2.5 Pro）：
- 输入 vlmscale 高分辨率图像 (~3072px) 供 VLM 分析，但在提示词中使用 modelscale 坐标尺寸，使 VLM 直接返回 modelscale 坐标
- 分类型调用 VLM：先 hotlap_start → pit → start → timing
- 包含 mask 验证 + 重试机制

**程序化部分**：
- `road_centerline.py` 提取赛道中线 → 弯道检测 → 生成计时点对 (AC_TIME_N_L/R)

**后处理**：
- `snap-to-road`：将验证未通过的对象吸附到最近的合法位置（路面/维修区）
- 最终验证通过率目标 100%

支持多布局（每个布局独立的游戏对象 JSON）。

- 输入: `output/02_mask_full_map/result_vlmscale.png` + road mask + GeoTIFF
- 输出: `output/07_ai_game_objects/`
  - `geo_metadata.json` — 图像-地理坐标映射
  - `{LayoutName}/game_objects.json` — 游戏对象数据
  - `{LayoutName}/centerline.json` — 赛道中线数据
  - `{LayoutName}/game_objects_preview.png` — 2D 预览图

##### 阶段 7a: 手动游戏对象编辑 (`s07a_manual_game_objects.py`) — 可选

初始化手动编辑目录，从阶段 7 AI 结果复制初始数据。通过 Web 编辑器（Objects Editor / Game Objects Editor）精调后保存。保留已有手动编辑结果，仅覆盖不存在的布局。

- 输出: `output/07a_manual_game_objects/game_objects.json`

##### 阶段 8: Blender 多边形生成 (`s08_blender_polygons.py`)

以 Blender 后台模式运行，读取所有 `*_blender.json`，在 Blender 中生成：
- **2D Curve 对象**（调试/可视化用）：按 tag → clip → mask_index → include/exclude 分组
- **Mesh 对象**（最终多边形面）：include 区域扣除 exclude 洞，经 2D Curve 填充 → 转 Mesh → 三角化

生成的 Mesh 对象可直接用于后续的 mask 投影和表面提取。

- 输入: `output/05_merge_segments/`
- 输出: `output/08_blender_polygons/polygons.blend`

##### 阶段 9: Blender 无头自动化 (`s09_blender_automate.py`)

以 Blender 后台模式运行完整的自动化集成流程：
1. 加载基础 3D 瓦片（base_level 级别）
2. 按 road mask 自动精炼到 target_fine_level
3. 提取赛道表面（按 road/grass/sand/kerb 分别设定采样密度）
4. 导入围墙对象（从阶段 6 的 JSON）
5. 导入游戏对象（从阶段 7/7a 的 JSON）
6. 纹理处理（解包、转换为 PNG、材质转 BSDF）
7. 保存最终 .blend 文件

- 输入: 阶段 1/5/6/7/8 的输出（围墙 6、对象 7、多边形 8）
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

封装 Google Gemini API 调用（使用 `google-genai` 新版 SDK），支持图像 + 文本多模态输入，JSON 格式化输出。

#### `ai_wall_generator.py` — 程序化围墙生成

**无 LLM 依赖。** 基于 SAM3 mask 的洪水填充算法生成虚拟围墙：
- 从 road mask 出发扩展到所有可行驶区域（grass 是缓冲区）
- 遇到障碍物（trees/building/water）停止
- 输出外围墙轮廓 + 障碍物独立围墙

#### `ai_game_objects.py` — 混合游戏对象生成

采用 VLM + 程序化混合策略：
- **VLM**（Gemini 2.5 Pro）分类型生成布局依赖对象（hotlap/pit/start）
- **程序化**（road_centerline.py）生成中线计时点
- **后处理**：snap-to-road 吸附 + mask 验证确保 100% 通过率
- 支持 vlmscale 高分辨率输入 + modelscale 坐标系输出

#### `road_centerline.py` — 赛道中线提取

从 road mask 提取赛道中线的纯程序化模块：
- 骨架化（scikit-image skeletonize）
- 中线点排序与平滑
- 弯道曲率分析与弯道检测
- 生成 AC_TIME_N_L / AC_TIME_N_R 计时点对

#### `image_inpainter.py` — 航拍影像修复

检测并修复航拍影像中的黑色缺失区域：
- 洪水填充算法区分内部孔洞和影像边界
- 使用 Gemini 图像修复模型（gemini-3-pro-image-preview）填补孔洞
- 面积阈值过滤小孔洞，避免过度修复

#### `ai_visualizer.py` — AI 结果可视化

Matplotlib 无头后端可视化：围墙预览、游戏对象预览等 2D 标注图。

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

### Web 编辑器套件 (`script/track_session_anaylzer/`)

基于 Leaflet.js 的交互式 Web 编辑器套件，通过本地 HTTP 服务器提供。启动方式：

```bash
python script/track_session_anaylzer/run_analyzer.py
```

#### Session Analyzer (`index.html` / `app.js`)

GPS 遥测数据可视化分析器：
- 支持 RaceChrono CSV 文件拖入
- 本地瓦片地图显示
- 多数据集叠加对比
- 独立偏移控制
- 自动圈速检测

#### Centerline Editor (`centerline_editor.html/js/css`)

赛道中线交互编辑器：
- 在 Leaflet 地图上绘制/调整赛道中线
- 读取和保存 centerline.json

#### Layout Editor (`layout_editor.html/js/css`)

赛道布局编辑器：
- 创建/管理多赛道布局
- 生成 per-layout 二值 mask
- 验证布局合理性

#### Wall Editor (`wall_editor.html/js/css`)

围墙交互编辑器：
- 加载 walls.json 在地图上显示
- 拖拽编辑围墙控制点
- 保存修改后的围墙数据

#### Objects Editor (`objects_editor.html/js/css`)

游戏对象编辑器：
- 在地图上显示/编辑游戏对象位置
- 调整朝向（orientation_z）
- 支持各类对象类型

#### Game Objects Editor (`gameobjects_editor.html/js`)

游戏对象高级编辑器：
- 功能更丰富的游戏对象编辑界面
- 支持完整的对象管理操作

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
  "layout_name": "LayoutCW",
  "track_direction": "clockwise",
  "objects": [
    {
      "name": "AC_HOTLAP_START_0",
      "position": [x, y],
      "orientation_z": [dx, dy],
      "type": "hotlap_start"
    }
  ],
  "_validation": {
    "hotlap": {"total": 1, "passed": 1, "rule": "on_road"},
    "pit": {"total": 8, "passed": 8, "rule": "near_road+not_on_road+not_invalid"},
    "start": {"total": 8, "passed": 8, "rule": "on_road"},
    "timing_0": {"total": 1, "passed": 1, "rule": "near_road"}
  }
}
```

### 地理元数据 (`geo_metadata.json`)

```json
{
  "image_width": 1008,
  "image_height": 998,
  "bounds": {
    "north": 22.713015,
    "south": 22.710245,
    "east": 113.866563,
    "west": 113.863549
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
| AI 大模型 | Google Gemini (VLM: gemini-2.5-pro, 图像修复: gemini-3-pro-image-preview) |
| 影像处理 | rasterio, PIL/Pillow, OpenCV, numpy, scikit-image |
| 地理坐标 | pyproj / 手动 WGS84 ↔ ECEF ↔ ENU 转换 |
| 三维可视化 | Blender 5.0+ (bpy API) |
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
    --output-dir output
```

### 2. 单独运行各阶段

```bash
# 阶段1: B3DM → GLB 转换
python script/stages/s01_b3dm_convert.py \
    --tiles-dir test_images_shajing/b3dm --output-dir output

# 阶段2: 全图 SAM3 分割 + VLM 图像生成
python script/stages/s02_mask_full_map.py \
    --geotiff test_images_shajing/result.tif --output-dir output

# 阶段3: 裁剪全图为瓦片
python script/stages/s03_clip_full_map.py \
    --geotiff test_images_shajing/result.tif --output-dir output

# 阶段4: 逐瓦片精细分割（含回退提示词）
python script/stages/s04_mask_on_clips.py \
    --geotiff test_images_shajing/result.tif --output-dir output

# 阶段5: 合并分割
python script/stages/s05_merge_segments.py \
    --geotiff test_images_shajing/result.tif \
    --tiles-dir test_images_shajing/b3dm --output-dir output

# 阶段6: 程序化围墙生成
python script/stages/s06_ai_walls.py \
    --geotiff test_images_shajing/result.tif --output-dir output

# 阶段7: AI 游戏对象生成
python script/stages/s07_ai_game_objects.py \
    --geotiff test_images_shajing/result.tif --output-dir output

# 阶段8: Blender 多边形生成
python script/stages/s08_blender_polygons.py --output-dir output

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

### 4. Web 编辑器套件

```bash
python script/track_session_anaylzer/run_analyzer.py
# 自动打开浏览器，访问各编辑器：
# - Session Analyzer: GPS 遥测可视化
# - Centerline Editor: 赛道中线编辑
# - Layout Editor: 赛道布局管理
# - Wall Editor: 围墙交互编辑
# - Objects Editor / Game Objects Editor: 游戏对象编辑
```

---

## 设计原则

1. **统一配置**: 所有配置集中在 `pipeline_config.py`，`blender_scripts/config.py` 仅作为薄层导入
2. **阶段独立**: 每个阶段一个文件，可通过 `__main__` 独立运行
3. **输出隔离**: 每个阶段有独立的输出子目录（`output/NN_stage_name/`），不污染源数据
4. **阶段间通信**: 下游阶段从上游阶段的输出目录读取文件，不依赖内存状态
5. **可测试性**: Blender 耦合代码与纯 Python 逻辑分离，支持脱离 Blender 测试
6. **物理空间优先**: 算法参数用物理单位（米、度）定义，运行时按 GeoTIFF 分辨率转换为像素值
7. **AI + 程序化混合**: VLM 处理高层语义决策（布局、位置），程序化算法处理几何精确任务（围墙、计时点），后处理确保质量
8. **手动可介入**: 关键阶段提供 Web 编辑器用于 AI 输出的手动精调（布局、围墙、游戏对象）
