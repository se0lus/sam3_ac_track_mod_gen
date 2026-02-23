# SAM3 赛道分割 — Assetto Corsa 赛道 Mod 自动生成器

> 无人机 2D 航拍 / 3D 倾斜摄影 → SAM3 语义分割 → Blender 三维处理 → Assetto Corsa 赛道 Mod

全自动 11 阶段流水线，将无人机影像（GeoTIFF）和 3D Tiles 转化为可游玩的 Assetto Corsa 赛道 Mod。
内置 Web Dashboard 管理流水线，配备 6 个交互式编辑器，支持在每个关键阶段手动精调 AI 输出。

## 目录

- [流水线总览](#流水线总览)
- [快速开始](#快速开始)
- [阶段详解](#阶段详解)
  - [阶段 1: B3DM → GLB 转换](#阶段-1-b3dm--glb-转换)
  - [阶段 2: 全图 SAM3 分割](#阶段-2-全图-sam3-分割)
  - [阶段 2a: 赛道布局管理（手动）](#阶段-2a-赛道布局管理手动)
  - [阶段 3: 全图裁剪](#阶段-3-全图裁剪)
  - [阶段 4: 逐瓦片精细分割](#阶段-4-逐瓦片精细分割)
  - [阶段 5: 分割合并](#阶段-5-分割合并)
  - [阶段 5a: 手动表面编辑（手动）](#阶段-5a-手动表面编辑手动)
  - [阶段 6: 程序化围墙生成](#阶段-6-程序化围墙生成)
  - [阶段 6a: 手动围墙编辑（手动）](#阶段-6a-手动围墙编辑手动)
  - [阶段 7: 混合游戏对象生成](#阶段-7-混合游戏对象生成)
  - [阶段 7a: 手动游戏对象编辑（手动）](#阶段-7a-手动游戏对象编辑手动)
  - [阶段 8: Blender 多边形生成](#阶段-8-blender-多边形生成)
  - [阶段 9: Blender 无头自动化](#阶段-9-blender-无头自动化)
  - [阶段 9a: 手动 Blender 编辑（手动）](#阶段-9a-手动-blender-编辑手动)
  - [阶段 10: 模型导出（FBX / KN5）](#阶段-10-模型导出fbx--kn5)
  - [阶段 11: 赛道打包](#阶段-11-赛道打包)
- [Junction 链接系统](#junction-链接系统)
- [Web Dashboard 与编辑器](#web-dashboard-与编辑器)
- [Blender 脚本](#blender-脚本)
- [核心工具模块](#核心工具模块)
- [配置参考](#配置参考)
- [数据格式](#数据格式)
- [项目结构](#项目结构)
- [技术栈](#技术栈)

---

## 流水线总览

```
GeoTIFF 影像 + 3D Tiles (b3dm)
    │
    ▼
 ┌──────────────────────────────────────────────────────────────┐
 │  S1   B3DM → GLB               3D Tiles 格式转换             │
 │  S2   全图 SAM3 分割            8 类语义分割                  │
 │  S2a  布局编辑器 ···········    （手动，Web 编辑器）           │
 │  S3   全图裁剪                  40m × 40m 瓦片切分            │
 │  S4   逐瓦片 SAM3               精细分割每个瓦片              │
 │  S5   分割合并                  合并 + Blender 坐标转换       │
 │  S5a  表面编辑器 ···········    （手动，Web 编辑器）           │
 │  S6   围墙生成                  从 road mask 洪水填充         │
 │  S6a  围墙编辑器 ···········    （手动，Web 编辑器）           │
 │  S7   游戏对象生成              VLM + 程序化混合              │
 │  S7a  对象编辑器 ···········    （手动，Web 编辑器）           │
 │  S8   Blender 多边形            JSON → Mesh + 间隙填充       │
 │  S9   Blender 自动化            瓦片 + 表面 + 纹理           │
 │  S9a  手动 Blender ·········    （手动，Blender 编辑）        │
 │  S10  模型导出                  FBX 拆分 + KN5 转换          │
 │  S11  赛道打包                  AC 文件夹 + UI + 元数据       │
 └──────────────────────────────────────────────────────────────┘
    │
    ▼
  output/11_track_packaging/{track_name}/  →  Assetto Corsa Mod
```

虚线（·····）表示可选的手动编辑阶段，通过 [Junction 链接系统](#junction-链接系统) 回馈到流水线。

---

## 快速开始

### 环境要求

- Python 3.12+, CUDA 12.6
- Blender 5.0+
- SAM3 模型权重 (`sam3_model/sam3.pt`)

### 环境部署

```bash
# 一键部署（Windows）
setup_env.bat

# 手动部署
conda create -n sam3 python=3.12
conda activate sam3
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pip install -e ./sam3/
```

### 运行完整流水线

```bash
python script/sam3_track_gen.py \
    --geotiff test_images_shajing/result.tif \
    --tiles-dir test_images_shajing/b3dm \
    --output-dir output
```

### 单独运行各阶段

```bash
# 阶段 1: B3DM → GLB
python script/stages/s01_b3dm_convert.py \
    --tiles-dir test_images_shajing/b3dm --output-dir output

# 阶段 2: 全图 SAM3 分割
python script/stages/s02_mask_full_map.py \
    --geotiff test_images_shajing/result.tif --output-dir output

# 阶段 3–11: 类似模式（各阶段模块内有具体参数说明）
```

### 启动 Web Dashboard

```bash
python script/webTools/run_webtools.py
# 自动打开浏览器 → Dashboard + 6 个交互式编辑器
```

### Blender 交互模式

1. 打开 Blender → Text Editor → 加载 `blender_helpers.py` → 运行
2. 右键 → **SAM3 Quick Tools** 子菜单
3. Load Base Tiles → Refine by Mask → Extract Surface → Import Walls/Objects

---

## 阶段详解

每个阶段遵循统一接口：

```python
def run(config: PipelineConfig) -> None: ...

if __name__ == "__main__":
    # 独立运行器，便于单独测试
```

### 阶段 1: B3DM → GLB 转换

**文件:** `script/stages/s01_b3dm_convert.py`

批量将 3D Tiles 的 B3DM 文件转换为 Blender 可导入的 GLB 格式。解析 B3DM 文件头，提取内嵌的 glTF 二进制数据。

| | |
|---|---|
| **输入** | `tiles_dir/` (b3dm + tileset.json) |
| **输出** | `output/01_b3dm_convert/glb/`（保持原始目录结构） |

### 阶段 2: 全图 SAM3 分割

**文件:** `script/stages/s02_mask_full_map.py`

对全幅 GeoTIFF 影像执行 SAM3 分割，使用 8 类文本提示词识别不同地物。同时生成两种分辨率影像供下游使用：

- **modelscale**（~1008px）— mask 坐标系，用于所有下游处理
- **vlmscale**（~3072px）— 高分辨率影像，供 Gemini VLM 分析

包含自动检测航拍影像中心孔洞并调用 Gemini 图像修复。

| | |
|---|---|
| **输入** | GeoTIFF 影像 |
| **输出** | `output/02_mask_full_map/` |

输出文件：
- `result_modelscale.png`, `result_vlmscale.png` — 缩放后的影像
- `{tag}_mask.png` — 每类标签的二值 mask（road, grass, sand, kerb, trees, building, water, concrete）
- `merged_mask.png` — 合并后的 road 二值 mask
- `result_masks.json` — mask 多边形数据
- `results_visualization.png` — 分割可视化叠加图

### 阶段 2a: 赛道布局管理（手动）

**文件:** `script/stages/s02a_track_layouts.py`

可选手动阶段。从阶段 2 输出初始化布局编辑器目录。用户通过 Web 布局编辑器创建/管理多赛道布局（顺时针、逆时针等）。每个布局在下游生成独立的游戏对象配置。

| | |
|---|---|
| **输出** | `output/02a_track_layouts/`（layouts.json + 每布局二值 mask） |

### 阶段 3: 全图裁剪

**文件:** `script/stages/s03_clip_full_map.py`

使用贪心覆盖算法将全图智能切分为约 40m × 40m 的瓦片。确保所有 mask 像素被覆盖，瓦片间允许可控重叠（~10%）。

从 `02_result` junction 读取 — 自动尊重手动布局编辑。

| | |
|---|---|
| **输入** | GeoTIFF + `02_result/merged_mask.png` |
| **输出** | `output/03_clip_full_map/`（clip_0.tif, clip_1.tif, ..., clip_boxes_visualization.png） |

### 阶段 4: 逐瓦片精细分割

**文件:** `script/stages/s04_mask_on_clips.py`

对每个 clip 瓦片按 8 种标签分别执行精细 SAM3 分割。支持**回退提示词机制**：当主提示词在某个 clip 上未产生任何有效 mask 时，按顺序尝试替代提示词（如 "race track surface" → "asphalt road" → "concrete road"）。

| | |
|---|---|
| **输入** | `output/03_clip_full_map/clip_*.tif` |
| **输出** | `output/04_mask_on_clips/{tag}/clip_N_masks.json`（归一化/像素/地理坐标三套多边形） |

### 阶段 5: 分割合并

**文件:** `script/stages/s05_merge_segments.py`

将多瓦片 SAM3 分割结果合并为统一的 per-tag mask，并转换为 Blender 坐标：

1. 将各 clip 的分割多边形光栅化到共享画布，消除重叠
2. 优先级合成（sand < grass < road2 < road < kerb）消除 tag 间间隙
3. Road 边缘间隙闭合（可配置：`s5_road_gap_close_m`）
4. 窄 kerb 吸收到 road
5. 地理坐标 → Blender 坐标转换（通过 `tileset.json`）

| | |
|---|---|
| **输入** | `output/04_mask_on_clips/` + `tileset.json` |
| **输出** | `output/05_merge_segments/{tag}/{tag}_merged_blender.json` + merge_preview/ |

### 阶段 5a: 手动表面编辑（手动）

**文件:** `script/stages/s05a_manual_surface_masks.py`

可选。从阶段 5 输出初始化手动编辑目录。用户通过 Web 表面编辑器精调 per-tag 表面 mask（road, grass, sand, kerb, road2）。可编辑的表面以瓦片方式渲染，便于高效编辑。

| | |
|---|---|
| **输出** | `output/05a_manual_surface_masks/`（保留已有编辑） |

### 阶段 6: 程序化围墙生成

**文件:** `script/stages/s06_ai_walls.py`

**无 LLM 依赖。** 纯 mask 几何运算的洪水填充算法：

1. 从 road mask 出发 → 穿越可行驶区域（grass 是缓冲区，包含在外墙内）
2. 遇到障碍物（trees, building, water）停止扩展
3. 生成外围墙轮廓 + 障碍物围墙
4. 赛道邻近约束（80px）— 防止围墙包含远离赛道的建筑物

| | |
|---|---|
| **输入** | `02_result/`（per-tag masks）+ GeoTIFF |
| **输出** | `output/06_ai_walls/`（walls.json, walls_preview.png, geo_metadata.json） |

### 阶段 6a: 手动围墙编辑（手动）

**文件:** `script/stages/s06a_manual_walls.py`

可选。复制阶段 6 结果供手动精调，通过 Web 围墙编辑器编辑。已有手动编辑不会被覆盖。

| | |
|---|---|
| **输出** | `output/06a_manual_walls/` |

### 阶段 7: 混合游戏对象生成

**文件:** `script/stages/s07_ai_game_objects.py`

采用 **VLM + 程序化** 混合策略生成 Assetto Corsa 游戏功能对象：

**VLM 部分（Gemini 2.5 Pro）：**
- 输入 vlmscale 高分辨率图像（~3072px）分析对象放置位置
- 分类型顺序调用：hotlap_start → pit → start → timing
- mask 验证 + 重试机制

**程序化部分：**
- `road_centerline.py` — 骨架化 → 中线提取 → 曲率分析 → AC_TIME_N_L/R 计时点对

**后处理：**
- snap-to-road：将验证未通过的对象吸附到最近的合法位置
- 目标 100% 验证通过率

支持多布局模式 — 每个布局获得独立的游戏对象和中线数据。

| | |
|---|---|
| **输入** | `02_result/result_vlmscale.png` + road mask + GeoTIFF |
| **输出** | `output/07_ai_game_objects/`（geo_metadata.json, {Layout}/game_objects.json, {Layout}/centerline.json） |

### 阶段 7a: 手动游戏对象编辑（手动）

**文件:** `script/stages/s07a_manual_game_objects.py`

可选。复制阶段 7 AI 结果供手动精调，通过 Web 对象编辑器 / 游戏对象编辑器编辑。保留已有编辑；仅复制不存在的布局。

| | |
|---|---|
| **输出** | `output/07a_manual_game_objects/` |

### 阶段 8: Blender 多边形生成

**文件:** `script/stages/s08_blender_polygons.py`

以 Blender 无头模式运行。两阶段处理：

**第一阶段（纯 Python，无需 Blender）：**
- 从 JSON 重新光栅化合并后的 mask
- 应用围墙约束
- 形态学间隙填充（`s8_gap_fill_enabled`）
- 剩余空隙用默认标签填充（`s8_gap_fill_default_tag`，默认 road2）
- 从填充后的光栅重新提取轮廓

**第二阶段（Blender 无头）：**
- 生成 2D Curve 对象（可选，仅调试用：`s8_generate_curves`）
- 转换为 Mesh + 三角化
- 组织到 `mask_polygon_{tag}` 集合

从 `05_result` 和 `06_result` junction 读取 — 尊重手动编辑。

| | |
|---|---|
| **输入** | `05_result/` 合并后的 JSON，`06_result/` 围墙 |
| **输出** | `output/08_blender_polygons/polygons.blend` |

### 阶段 9: Blender 无头自动化

**文件:** `script/stages/s09_blender_automate.py`

Blender 无头模式的主编排脚本：

1. 加载基础 3D 瓦片（base_level 级别）
2. 按 road mask 自动精炼到 target_fine_level
3. 提取赛道表面（per-tag 采样密度：road 0.1m, grass 2.0m, kerb 0.1m, sand 2.0m）
4. 从 `06_result/` 导入围墙
5. 从 `07_result/` 导入游戏对象
6. 纹理处理（解包、转 PNG、材质转 BSDF）
7. 保存最终 .blend 文件

| | |
|---|---|
| **输入** | 阶段 1, 5, 6, 7, 8 的输出（通过 junction） |
| **输出** | `output/09_blender_automate/final_track.blend` + `texture/` |

### 阶段 9a: 手动 Blender 编辑（手动）

**文件:** `script/stages/s09a_manual_blender.py`

可选。复制 `final_track.blend` + 纹理供 Blender 手动编辑。已有编辑不会被覆盖。

| | |
|---|---|
| **输出** | `output/09a_manual_blender/` |

### 阶段 10: 模型导出（FBX / KN5）

**文件:** `script/stages/s10_model_export.py`

将最终 .blend 导出为 Assetto Corsa 可用的 FBX 文件，并可选转换为 KN5：

1. 清理 — 移除 mask 集合和非游戏数据
2. 拆分超大 mesh（顶点上限：21K/mesh，批次上限：100MB/FBX）
3. 按 AC 命名规范重命名碰撞对象
4. 从 `L{N}` 集合自动检测瓦片级别
5. 按批次导出 FBX（不嵌入纹理）
6. FBX → KN5 转换（通过 `ksEditorAT.exe`，自动从 GitHub 下载）
7. 材质生成（ksAmbient, ksDiffuse, ksEmissive）

从 `09_result` junction 读取。

| | |
|---|---|
| **输入** | `09_result/final_track.blend` |
| **输出** | `output/10_model_export/`（*.fbx, *.kn5） |

### 阶段 11: 赛道打包

**文件:** `script/stages/s11_track_packaging.py`

组装最终的 Assetto Corsa 赛道文件夹，完整支持多布局：

```
{track_name}/
├── models_{layout}.ini          （每布局一个）
├── {shared}.kn5                 （地形、碰撞、环境 — 共享）
├── go_{layout}.kn5              （游戏对象 — 每布局独立）
├── {layout}/
│   ├── map.png                  （TrackMapGenerator 生成的小地图）
│   └── data/
│       ├── map.ini
│       └── cameras.ini
└── ui/{layout}/
    ├── ui_track.json            （元数据：名称、国家、作者、年份、标签）
    ├── preview.png
    └── outline.png
```

功能特性：
- LLM 生成赛道描述（Gemini，`s11_llm_description` 启用时）
- LLM 生成预览图（`s11_llm_preview` 启用时）
- TrackMapGenerator 集成生成小地图
- 可配置维修区数量、赛道元数据、布局显示名

| | |
|---|---|
| **输入** | 阶段 10 的 KN5 文件 + 布局/对象元数据 |
| **输出** | `output/11_track_packaging/{track_name}/` |

---

## Junction 链接系统

手动编辑阶段通过**目录 junction**（Windows `mklink /J`）或符号链接（POSIX）透明覆盖对应的自动阶段，所有下游消费者自动使用手动编辑结果。

| Junction | 自动阶段 | 手动阶段 | 下游消费者 |
|----------|---------|---------|-----------|
| `02_result` | `02_mask_full_map` | `02a_track_layouts` | S3, S6, S7 |
| `05_result` | `05_merge_segments` | `05a_manual_surface_masks` | S8, S9 |
| `06_result` | `06_ai_walls` | `06a_manual_walls` | S8, S9 |
| `07_result` | `07_ai_game_objects` | `07a_manual_game_objects` | S9 |
| `09_result` | `09_blender_automate` | `09a_manual_blender` | S10 |

当手动阶段目录存在时，junction 指向手动目录；否则指向自动阶段。下游阶段始终从 `_result` junction 读取，因此手动编辑被透明尊重。

由 `PipelineConfig.setup_result_junctions()` 管理。

---

## Web Dashboard 与编辑器

**服务器:** `script/webTools/run_webtools.py`（基于 Flask）

### Dashboard

`dashboard.html/js/css` — 流水线管理界面：
- 执行流水线阶段（单个或批量）
- 通过 SSE 实时日志流
- 带 ETA 的进度条
- 输出文件浏览器（支持预览）
- 流水线配置编辑器

### 交互式编辑器

| 编辑器 | 文件 | 用途 |
|--------|------|------|
| **布局编辑器** | `layout_editor.html/js/css` | 创建/管理多赛道布局边界 |
| **表面编辑器** | `surface_editor.html/js/css` | 精调 per-tag 表面 mask（road/grass/sand/kerb/road2） |
| **围墙编辑器** | `wall_editor.html/js/css` | 在地图上拖拽编辑围墙控制点 |
| **对象编辑器** | `objects_editor.html/js/css` | 在地图上定位/调整游戏对象朝向 |
| **游戏对象编辑器** | `gameobjects_editor.html/js` | 高级游戏对象管理 |
| **中线编辑器** | `centerline_editor.html/js/css` | 绘制/调整赛道中线 |

所有编辑器共享：
- Leaflet.js 地图，统一**右键长按拖动**（一致的交互方式）
- 深色主题（`#0b0f17` / `#1a1a2e`）
- `style.css` 共享 CSS 变量
- 自定义 toggle switch 和 pill selector（禁止浏览器原生控件）
- 本地离线地图瓦片代理

### 主要 API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/pipeline/run` | POST | 执行阶段 |
| `/api/pipeline/status` | GET | 阶段执行状态 |
| `/api/pipeline/config` | GET/POST | 流水线配置读写 |
| `/api/sse/pipeline` | SSE | 实时日志流 |
| `/api/walls` | GET/POST | 围墙数据 |
| `/api/game_objects` | GET/POST | 游戏对象数据 |
| `/api/track_layouts` | GET/POST | 布局数据 |
| `/api/centerline` | GET/POST | 中线数据 |
| `/api/surface_mask/{tag}` | GET/POST | per-tag 表面 mask |
| `/api/centerline/regenerate` | POST | 重新计算中线 |
| `/api/vlm_objects/regenerate` | POST | 重新生成 VLM 对象 |
| `/tiles/{z}/{x}/{y}.png` | GET | 离线地图瓦片代理 |

---

## Blender 脚本

**目录:** `blender_scripts/`

### 核心模块

| 文件 | 用途 |
|------|------|
| `config.py` | 配置薄层（从 `pipeline_config.py` 导入，运行时通过命令行参数覆盖） |
| `blender_automate.py` | 阶段 9 无头编排脚本 |
| `blender_create_polygons.py` | 阶段 8 JSON → Blender 曲线/网格 |
| `blender_export.py` | 阶段 10 FBX/KN5 导出、mesh 拆分、材质生成 |
| `blender_helpers.py` | 工具库 + 右键菜单 Action 自动发现注册 |

### 右键菜单 Action（`sam3_actions/`）

由 `blender_helpers.py` 自动发现并注册到 View3D / Outliner 右键菜单。

| 模块 | 菜单项 | 功能 |
|------|--------|------|
| `load_base_tiles.py` | Load Base Tiles | 按指定 LOD 级别导入 GLB 瓦片 |
| `load_base_tiles.py` | Refine Selected Tiles | 为选中瓦片加载更精细的子瓦片 |
| `load_base_tiles.py` | Refine by Mask to Target Level | 按 mask 多边形自动精炼到目标级别 |
| `terrain_mesh_extractor.py` | Terrain Mesh Extract | 将表面 mask 投影到 3D 地形 |
| `surface_extractor.py` | Extract Surface | 从 mask 生成 AC 碰撞网格 |
| `boolean_mesh_generator.py` | Boolean Mesh Generate | 布尔运算处理复杂表面切割 |
| `import_walls.py` | Import Walls | 导入 walls.json 到场景 |
| `import_game_objects.py` | Import Game Objects | 导入 game_objects.json 到场景 |
| `texture_tools.py` | Unpack & Convert Textures | 解包纹理、转 PNG、设置 BSDF 材质 |
| `clear_scene.py` | Clear Scene | 重置场景（保留 mask 集合） |

辅助模块：`c_tiles.py`（瓦片树管理）、`mask_select_utils.py`（XZ 平面相交测试）。

---

## 核心工具模块

**目录:** `script/`

### 地理空间与影像

| 模块 | 用途 |
|------|------|
| `geo_tiff_image.py` | GeoTIFF 读取、缩放、像素 ↔ 地理坐标双向转换 |
| `geo_sam3_image.py` | SAM3 分割 + mask → 多边形提取（Douglas-Peucker 简化） |
| `geo_sam3_utils.py` / `geo_sam3_utils2.py` | 坐标变换、智能 clip 生成 |
| `geo_sam3_blender_utils.py` | WGS84 → ECEF → ENU → Blender 本地坐标转换 |
| `image_inpainter.py` | 洪水填充孔洞检测 + Gemini 图像修复 |

### 分割与合并

| 模块 | 用途 |
|------|------|
| `mask_merger.py` | 多 clip mask 多边形光栅化 + 合并到共享画布 |
| `mask_gap_filler.py` | 可行驶区域内的形态学间隙填充 |
| `b3dm_converter.py` | B3DM 文件头解析 → GLB 提取（多线程） |

### 赛道生成

| 模块 | 用途 |
|------|------|
| `road_centerline.py` | 骨架化 → 中线提取 → 曲率分析 → 计时点生成 |
| `ai_wall_generator.py` | 基于 mask 的洪水填充围墙生成（无 LLM） |
| `ai_game_objects.py` | VLM + 程序化混合游戏对象生成 |
| `ai_visualizer.py` | Matplotlib 围墙/对象可视化 |
| `gemini_client.py` | Google Gemini API 封装（多模态、JSON 输出） |

### 表面与碰撞

| 模块 | 用途 |
|------|------|
| `surface_extraction.py` | 采样网格、Delaunay 三角化、AC 碰撞命名 |
| `tile_plan.py` | 瓦片 LOD 规划（渐进式精炼） |

---

## 配置参考

所有配置集中在 `script/pipeline_config.py` → `PipelineConfig` 数据类。

### 输入 / 输出

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `geotiff_path` | `""` | GeoTIFF 影像路径 |
| `tiles_dir` | `""` | 3D Tiles 目录（b3dm + tileset.json） |
| `output_dir` | `output/` | 输出根目录 |
| `blender_exe` | `C:\...\blender.exe` | Blender 可执行文件路径 |
| `max_workers` | `4` | 并行阶段线程池大小 |

### AI / LLM

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `gemini_api_key` | （内置） | Gemini API 密钥 |
| `gemini_model` | `gemini-2.5-pro` | VLM 模型（对象生成） |
| `inpaint_model` | `gemini-2.5-flash-image` | 图像修复模型 |
| `inpaint_center_holes` | `True` | 是否自动修复航拍中心孔洞 |
| `inpaint_min_hole_ratio` | `0.001` | 孔洞面积最小阈值（0.1%） |
| `vlm_max_size` | `3072` | VLM 输入图像最大维度 |

### 赛道元数据

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `track_direction` | `clockwise` | 行驶方向（clockwise/counterclockwise） |
| `track_description` | `""` | 可选赛道描述（用于 AI 提示词） |
| `base_level` | `17` | 瓦片初始加载 LOD 级别 |
| `target_fine_level` | `22` | 自动精炼目标 LOD 级别 |

### SAM3 分割提示词

8 类语义标签，可配置提示词和阈值。road 支持回退提示词机制：

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
```

### 表面采样密度

| 标签 | 密度 | 说明 |
|------|------|------|
| road | 0.1 m | 高精度赛道表面 |
| kerb | 0.1 m | 高精度路缘 |
| grass | 2.0 m | 粗略草地缓冲区 |
| sand | 2.0 m | 粗略砂石区域 |
| road2 | 2.0 m | 次级可行驶表面 |
| default | 1.0 m | 其他标签的默认值 |

### 各阶段专属配置

<details>
<summary>阶段 5（分割合并）</summary>

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `s5_road_gap_close_m` | `0.20` | Road 边缘小间隙闭合阈值（米） |
| `s5_kerb_narrow_max_width_m` | `0.30` | 窄 kerb 吸收最大宽度（米） |
| `s5_kerb_narrow_adjacency_m` | `0.20` | 窄 kerb 邻近 road 阈值（米） |

</details>

<details>
<summary>阶段 8（多边形生成）</summary>

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `s8_generate_curves` | `False` | 生成调试用 2D 曲线（较慢） |
| `s8_gap_fill_enabled` | `True` | 自动填充 mask 间隙 |
| `s8_gap_fill_threshold_m` | `0.20` | 小间隙阈值（米） |
| `s8_gap_fill_default_tag` | `road2` | 剩余空隙的默认填充标签 |

</details>

<details>
<summary>阶段 9（Blender 自动化）</summary>

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `s9_no_walls` | `False` | 跳过围墙导入 |
| `s9_no_game_objects` | `False` | 跳过游戏对象导入 |
| `s9_no_surfaces` | `False` | 跳过表面提取 |
| `s9_no_textures` | `False` | 跳过纹理处理 |
| `s9_no_background` | `False` | 跳过背景生成 |
| `s9_refine_tags` | `["road"]` | 用于瓦片精炼的标签 |
| `s9_tile_padding` | `0.0` | 多边形 AABB 周围填充（米） |
| `s9_mesh_simplify` | `False` | 地形网格后处理简化 |

</details>

<details>
<summary>阶段 10（模型导出）</summary>

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `s10_max_vertices` | `21000` | 每个 mesh 对象最大顶点数 |
| `s10_max_batch_mb` | `100` | 每个 FBX 文件最大体积（MB） |
| `s10_fbx_scale` | `0.01` | FBX 导出全局缩放 |
| `s10_ks_ambient` | `0.5` | ksAmbient 材质值 |
| `s10_ks_diffuse` | `0.1` | ksDiffuse 材质值 |
| `s10_ks_emissive` | `0.1` | ksEmissive 材质值 |
| `s10_kseditor_exe` | `""` | ksEditorAT 路径（为空则自动检测） |

</details>

<details>
<summary>阶段 11（赛道打包）</summary>

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `s11_track_name` | `""` | 赛道文件夹名（为空则从 geotiff 推导） |
| `s11_track_author` | `""` | 作者名 |
| `s11_track_country` | `""` | 国家 |
| `s11_track_city` | `""` | 城市 |
| `s11_track_tags` | `["circuit", "original"]` | 赛道标签 |
| `s11_track_year` | `0` | 年份（0 = 当前年份） |
| `s11_pitboxes` | `10` | 维修区数量 |
| `s11_llm_description` | `True` | 使用 LLM 生成赛道描述 |
| `s11_llm_preview` | `True` | 使用 LLM 生成预览图 |

</details>

### Blender 集合

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `root_curve_collection_name` | `mask_curve2D_collection` | 调试曲线集合 |
| `root_polygon_collection_name` | `mask_polygon_collection` | mask 多边形集合 |
| `collision_collection_name` | `collision` | 碰撞对象集合 |

碰撞子集合按 tag 分：`collision_road`, `collision_kerb`, `collision_grass`, `collision_sand`, `collision_road2`, `collision_walls`。映射在 `surface_extraction.COLLISION_COLLECTION_MAP`。

---

## 数据格式

### Mask JSON（`clip_N_masks.json`）

```json
{
  "masks": [{
    "tag": "road",
    "mask_index": 0,
    "prob": 0.95,
    "polygons": {
      "include": [{ "points_normalized": [[x,y]], "points_pixel": [...], "points_geo": [[lon,lat]] }],
      "exclude": [...]
    }
  }]
}
```

### Blender JSON（`{tag}_merged_blender.json`）

```json
{
  "polygons": {
    "include": [{ "tag": "road", "mask_index": 0, "prob": 0.95, "points_xyz": [[x,y,z]] }],
    "exclude": [...]
  }
}
```

### 围墙 JSON（`walls.json`）

```json
{
  "walls": [{ "type": "outer", "points": [[x1, y1], [x2, y2]] }]
}
```

### 游戏对象 JSON（`game_objects.json`）

```json
{
  "layout_name": "LayoutCW",
  "track_direction": "clockwise",
  "objects": [{
    "name": "AC_HOTLAP_START_0",
    "position": [x, y],
    "orientation_z": [dx, dy],
    "type": "hotlap_start"
  }],
  "_validation": {
    "hotlap": { "total": 1, "passed": 1, "rule": "on_road" },
    "pit":    { "total": 8, "passed": 8, "rule": "near_road+not_on_road+not_invalid" },
    "start":  { "total": 8, "passed": 8, "rule": "on_road" }
  }
}
```

### 地理元数据（`geo_metadata.json`）

```json
{
  "image_width": 1008,
  "image_height": 998,
  "bounds": { "north": 22.713, "south": 22.710, "east": 113.866, "west": 113.863 }
}
```

---

## 项目结构

```
sam3_track_seg/
├── script/
│   ├── pipeline_config.py            # 统一配置（唯一来源）
│   ├── sam3_track_gen.py             # 主入口 CLI
│   ├── stages/
│   │   ├── s01_b3dm_convert.py
│   │   ├── s02_mask_full_map.py
│   │   ├── s02a_track_layouts.py
│   │   ├── s03_clip_full_map.py
│   │   ├── s04_mask_on_clips.py
│   │   ├── s05_merge_segments.py
│   │   ├── s05a_manual_surface_masks.py
│   │   ├── s06_ai_walls.py
│   │   ├── s06a_manual_walls.py
│   │   ├── s07_ai_game_objects.py
│   │   ├── s07a_manual_game_objects.py
│   │   ├── s08_blender_polygons.py
│   │   ├── s09_blender_automate.py
│   │   ├── s09a_manual_blender.py
│   │   ├── s10_model_export.py
│   │   └── s11_track_packaging.py
│   ├── webTools/                     # Dashboard + 6 个 Web 编辑器
│   │   ├── run_webtools.py           # Flask 服务器
│   │   ├── dashboard.html/js/css
│   │   ├── layout_editor.html/js/css
│   │   ├── surface_editor.html/js/css
│   │   ├── wall_editor.html/js/css
│   │   ├── objects_editor.html/js/css
│   │   ├── gameobjects_editor.html/js
│   │   ├── centerline_editor.html/js/css
│   │   └── style.css                # 共享深色主题
│   ├── geo_tiff_image.py
│   ├── geo_sam3_image.py
│   ├── geo_sam3_blender_utils.py
│   ├── mask_merger.py
│   ├── mask_gap_filler.py
│   ├── road_centerline.py
│   ├── ai_wall_generator.py
│   ├── ai_game_objects.py
│   ├── surface_extraction.py
│   ├── gemini_client.py
│   └── ...
├── blender_scripts/
│   ├── config.py
│   ├── blender_automate.py
│   ├── blender_create_polygons.py
│   ├── blender_export.py
│   ├── blender_helpers.py
│   └── sam3_actions/                 # Blender 右键菜单插件
│       ├── load_base_tiles.py
│       ├── terrain_mesh_extractor.py
│       ├── surface_extractor.py
│       ├── boolean_mesh_generator.py
│       ├── import_walls.py
│       ├── import_game_objects.py
│       ├── texture_tools.py
│       └── ...
├── sam3/                             # SAM3 模型（git 子模块）
├── sam3_model/                       # 模型权重
├── ac_toolbox/                       # Assetto Corsa 工具和资源
├── tests/                            # 单元测试（镜像源代码结构）
├── test_images_shajing/              # 测试数据集（只读）
├── output/                           # 流水线输出（自动创建）
│   ├── 01_b3dm_convert/
│   ├── 02_mask_full_map/
│   ├── 02_result → (junction)
│   ├── 02a_track_layouts/
│   ├── 03_clip_full_map/
│   ├── 04_mask_on_clips/
│   ├── 05_merge_segments/
│   ├── 05_result → (junction)
│   ├── 05a_manual_surface_masks/
│   ├── 06_ai_walls/
│   ├── 06_result → (junction)
│   ├── 06a_manual_walls/
│   ├── 07_ai_game_objects/
│   ├── 07_result → (junction)
│   ├── 07a_manual_game_objects/
│   ├── 08_blender_polygons/
│   ├── 09_blender_automate/
│   ├── 09_result → (junction)
│   ├── 09a_manual_blender/
│   ├── 10_model_export/
│   └── 11_track_packaging/
├── requirements.txt
├── setup_env.bat
└── CLAUDE.md                         # 开发规则
```

---

## 技术栈

| 领域 | 技术 |
|------|------|
| AI 分割模型 | SAM3（Segment Anything Model 3, Meta） |
| AI 大模型 | Google Gemini 2.5 Pro（对象放置、赛道描述） |
| 图像修复 | Google Gemini 2.5 Flash Image |
| 影像处理 | rasterio, Pillow, OpenCV, NumPy, scikit-image |
| 地理坐标 | WGS84 ↔ ECEF ↔ ENU 坐标变换 |
| 三维引擎 | Blender 5.0+（bpy API, 无头自动化） |
| 三维数据 | B3DM / GLB, tileset.json（OGC 3D Tiles 1.0） |
| AC 导出 | FBX → KN5（通过 ksEditorAT） |
| Web 前端 | Flask, Leaflet.js, Server-Sent Events |
| 运行环境 | Windows, Python 3.12, CUDA 12.6, PyTorch |

---

## 设计原则

1. **统一配置** — 所有配置集中在 `pipeline_config.py`，`blender_scripts/config.py` 仅作为薄层导入。
2. **阶段独立** — 每个阶段一个文件，通过 `run(config)` + `__main__` 独立运行。
3. **输出隔离** — 每个阶段写入 `output/NN_stage_name/`，源数据只读。
4. **Junction 链接** — 手动编辑通过目录 junction 透明覆盖自动结果。
5. **阶段间通信** — 下游从上游输出目录读取文件，不依赖内存状态。
6. **可测试性** — Blender 耦合代码隔离，大部分逻辑无需 `bpy` 即可测试。
7. **物理空间优先** — 算法参数用物理单位（米、度），运行时按 GeoTIFF 分辨率转像素。
8. **AI + 程序化混合** — VLM 处理高层语义决策，算法处理几何精确任务，后处理确保质量。
9. **手动可介入** — 每个关键阶段配备 Web 编辑器，支持人工精调 AI 输出。
