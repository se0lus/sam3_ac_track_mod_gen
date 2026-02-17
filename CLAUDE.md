# SAM3 赛道分割 — Assetto Corsa 赛道 Mod 自动生成器

## 项目概述

自动化流水线：无人机采集的赛道 2D 航拍影像 / 3D 倾斜摄影模型 → SAM3 语义分割 → Blender 三维处理 → 可玩的 Assetto Corsa 赛道 Mod。

## 目录结构

```
sam3_track_seg/
├── script/                           # 核心 Python 流水线
│   ├── sam3_track_gen.py             # 主入口（CLI + 流水线编排）
│   ├── pipeline_config.py            # 统一配置（所有参数的唯一来源）
│   ├── stages/                       # 每个阶段一个文件，可独立运行
│   │   ├── s01_b3dm_convert.py       # 阶段1: B3DM → GLB 格式转换
│   │   ├── s02_mask_full_map.py      # 阶段2: 全图 SAM3 分割 + VLM 高分辨率图生成
│   │   ├── s02a_track_layouts.py     # 阶段2a: 赛道布局管理（手动，可选）
│   │   ├── s03_clip_full_map.py      # 阶段3: 全图裁剪为瓦片
│   │   ├── s04_mask_on_clips.py      # 阶段4: 逐瓦片精细分割（含回退提示词）
│   │   ├── s05_convert_to_blender.py # 阶段5: 地理坐标 → Blender 坐标
│   │   ├── s06_blender_polygons.py   # 阶段6: Blender 多边形生成
│   │   ├── s07_ai_walls.py           # 阶段7: AI 围墙生成（纯程序化，无 LLM）
│   │   ├── s08_ai_game_objects.py    # 阶段8: AI 游戏对象生成（VLM + 程序化混合）
│   │   ├── s08a_manual_game_objects.py # 阶段8a: 手动编辑游戏对象（可选）
│   │   └── s09_blender_automate.py   # 阶段9: Blender 自动化集成
│   ├── geo_tiff_image.py             # GeoTIFF 影像读取/缩放/坐标转换
│   ├── geo_sam3_image.py             # SAM3 分割 + mask → 多边形转换
│   ├── geo_sam3_utils2.py            # clip box 智能生成算法
│   ├── geo_sam3_blender_utils.py     # 地理坐标 → Blender 坐标转换
│   ├── b3dm_converter.py             # B3DM/GLB 格式转换器
│   ├── gemini_client.py              # Gemini API 客户端
│   ├── ai_wall_generator.py          # 围墙生成（SAM3 洪水填充，无 LLM）
│   ├── ai_game_objects.py            # 游戏对象生成（VLM + 程序化混合）
│   ├── ai_visualizer.py              # AI 结果可视化（matplotlib）
│   ├── road_centerline.py            # 赛道中线提取 + 弯道检测 + 计时点生成
│   ├── image_inpainter.py            # 航拍影像黑洞修复（Gemini 图像修复）
│   └── surface_extraction.py         # 赛道表面提取
├── blender_scripts/                  # Blender 端脚本（bpy API）
│   ├── config.py                     # Blender 配置（从 pipeline_config 导入）
│   ├── blender_create_polygons.py    # 批处理: JSON → Blender Curve/Mesh
│   ├── blender_automate.py           # 无头自动化（加载/精炼/提取/导入/保存）
│   ├── blender_helpers.py            # 右键菜单框架
│   └── sam3_actions/                 # 右键菜单 Action 插件包
│       ├── __init__.py
│       ├── c_tiles.py                # CTile 瓦片树管理
│       ├── load_base_tiles.py        # 加载/精炼 3D 瓦片
│       ├── surface_extractor.py      # 赛道表面提取 Action
│       ├── import_walls.py           # 导入围墙 Action
│       ├── import_game_objects.py    # 导入游戏对象 Action
│       ├── texture_tools.py          # 纹理解包/转换 Action
│       ├── clear_scene.py            # 清除场景
│       └── mask_select_utils.py      # Mask XZ 投影相交测试
├── script/track_session_anaylzer/    # Web 可视化 + 交互编辑器套件
│   ├── run_analyzer.py               # HTTP 服务器启动器
│   ├── index.html / app.js / style.css  # GPS 遥测数据可视化分析器
│   ├── centerline_editor.html/js/css    # 赛道中线交互编辑器
│   ├── layout_editor.html/js/css        # 赛道布局交互编辑器
│   ├── wall_editor.html/js/css          # 围墙交互编辑器
│   ├── objects_editor.html/js/css       # 游戏对象交互编辑器
│   ├── gameobjects_editor.html/js       # 游戏对象高级编辑器
│   └── map/                             # 本地离线瓦片地图
├── model/                            # SAM3 模型权重
├── sam3/                             # SAM3 模型源码
├── test_images_shajing/              # 测试数据集（沙井赛道）
│   ├── b3dm/                         # 3D Tiles 原始数据
│   └── result.tif                    # 2D 赛道全图 GeoTIFF
├── output/                           # 所有输出（按阶段子目录组织）
│   ├── 01_b3dm_convert/              # GLB 转换结果
│   ├── 02_mask_full_map/             # 全图分割 + modelscale/vlmscale 图像
│   ├── 02a_track_layouts/            # 赛道布局配置（可选）
│   ├── 03_clip_full_map/             # 裁剪瓦片
│   ├── 04_mask_on_clips/             # 逐瓦片分割结果（8 类标签）
│   ├── 05_convert_to_blender/        # Blender 坐标 JSON
│   ├── 06_blender_polygons/          # polygons.blend
│   ├── 07_ai_walls/                  # 围墙 JSON + 预览图
│   ├── 08_ai_game_objects/           # 游戏对象 JSON + 预览图（按布局子目录）
│   ├── 08a_manual_game_objects/      # 手动编辑的游戏对象（可选）
│   └── 09_blender_automate/          # 最终 final_track.blend
└── tests/                            # 单元和模块测试
```

## 技术栈

- Python 3.10+, Blender 5.0+ (bpy API), SAM3 (Meta Segment Anything 3)
- rasterio, Pillow, OpenCV, numpy, pyproj, scikit-image
- 3D Tiles (b3dm/glb), tileset.json
- Gemini API (VLM: gemini-2.5-pro, 图像修复: gemini-3-pro-image-preview)
- Web 编辑器: Leaflet.js, HTML/CSS/JS

## 开发规则

1. **统一配置**: 所有配置集中在 `script/pipeline_config.py`，不要在其他地方硬编码路径或参数。
2. **阶段独立**: 每个阶段 (`script/stages/s0N_*.py`) 必须可独立运行和测试。
3. **输出隔离**: 所有输出必须在 `output/` 下的阶段子目录中，绝不污染 `test_images_shajing/` 源数据目录。
4. **可测试性**: 所有模块必须可独立测试。耦合 Blender 的功能要隔离，使大部分代码可脱离 Blender 测试。
5. **测试位置**: 测试文件放在 `tests/` 目录，镜像源代码结构。
6. **Blender 脚本**: 新的 Blender Action 放在 `blender_scripts/sam3_actions/`，遵循 `__init__.py` 中的模式。
7. **碰撞命名**: 碰撞对象遵循 Assetto Corsa 命名规范: `1WALL_N`, `1ROAD_N`, `1SAND_N`, `1KERB_N`, `1GRASS_N`。
8. **先计划后编码**: 每个模块开始编码前需完成编码计划并经过 review。
9. **先测试后集成**: 所有模块测试通过后才可进行集成。
10. **物理空间优先**: 算法参数用物理单位（米、度）定义，运行时按 GeoTIFF 分辨率转换为像素值。

## 流水线概览

```
GeoTIFF 影像 + 3D Tiles(b3dm)
    │
    ├─[1]  B3DM → GLB 转换
    ├─[2]  全图 SAM3 分割（8 类标签）+ VLM 高分辨率图 + 中心孔洞修复
    ├─[2a] 赛道布局管理（可选，手动 Web 编辑器）
    ├─[3]  智能裁剪为瓦片 → clips
    ├─[4]  逐瓦片精细分割 (road/grass/sand/kerb/trees/building/water/concrete)
    │      └─ road 支持回退提示词："asphalt road" / "concrete road"
    ├─[5]  地理坐标 → Blender 坐标转换 + 按类型合并
    ├─[6]  Blender 批处理生成 2D Curve + Mesh
    ├─[7]  程序化围墙生成（SAM3 洪水填充，无 LLM 依赖）
    ├─[8]  混合游戏对象生成（VLM 布局对象 + 程序化计时点 + snap-to-road 后处理）
    ├─[8a] 手动游戏对象编辑（可选，Web 编辑器）
    └─[9]  Blender 无头自动化（加载瓦片 → 精炼 → 表面提取 → 导入围墙/对象 → 保存）
```

## 各阶段单独执行示例

以下示例基于测试数据 `test_images_shajing/`，所有输出默认到 `output/` 目录。

```bash
# 完整流水线（全部9个阶段）
python script/sam3_track_gen.py \
    --geotiff test_images_shajing/result.tif \
    --tiles-dir test_images_shajing/b3dm \
    --output-dir output

# 阶段1: B3DM → GLB 转换
python script/stages/s01_b3dm_convert.py \
    --tiles-dir test_images_shajing/b3dm --output-dir output

# 阶段2: 全图 SAM3 分割 + VLM 图像生成
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

# 运行指定的多个阶段
python script/sam3_track_gen.py \
    --stage ai_walls --stage ai_game_objects \
    --geotiff test_images_shajing/result.tif --output-dir output

# 启动 Web 编辑器套件
python script/track_session_anaylzer/run_analyzer.py
```

## SAM3 分割配置

8 类语义标签，支持回退提示词机制：

| 标签 | 提示词 | 阈值 | 回退提示词 | 说明 |
|------|--------|------|-----------|------|
| `road` | `race track surface` | 0.25 | `asphalt road`, `concrete road` | 赛道路面 |
| `grass` | `grass` | 0.4 | — | 草地缓冲区 |
| `sand` | `sand surface` | 0.4 | — | 砂石区 |
| `kerb` | `race track curb` | 0.2 | — | 路缘石 |
| `trees` | `forest` | 0.3 | — | 树木区域 |
| `building` | `building structure` | 0.4 | — | 建筑物 |
| `water` | `water pond` | 0.4 | — | 水域 |
| `concrete` | `concrete paved ground` | 0.4 | — | 混凝土铺装 |

回退机制：当主提示词在某个 clip 上未产生任何有效 mask 时，按顺序尝试回退提示词列表，直到检测成功。

## 碰撞对象命名规范

Assetto Corsa 游戏引擎通过对象名称识别碰撞类型：

| 前缀 | 类型 | 说明 |
|------|------|------|
| `1WALL_N` | 围墙 | 虚拟碰撞墙，防止赛车开出地图 |
| `1ROAD_N` | 路面 | 赛道铺装表面（高采样密度 0.5m） |
| `1GRASS_N` | 草地 | 赛道旁草地（低采样密度 2.0m） |
| `1SAND_N` | 砂石 | 砂石缓冲区（低采样密度 2.0m） |
| `1KERB_N` | 路缘 | 赛道路缘石（高采样密度 0.5m） |

所有碰撞对象统一放置在 Blender 的 `collision` Collection 中。

## 游戏对象规范

AI 生成的游戏对象（不可见，无网格），Z 轴为行驶方向，Y 轴朝上，高度为赛道表面之上 2 单位：

| 对象名 | 数量 | 说明 |
|--------|------|------|
| `AC_HOTLAP_START_0` | 1 | 计时赛起点，放在起点线前一个弯道的出弯处 |
| `AC_PIT_0` ~ `AC_PIT_N` | 8+ | 维修区位置 |
| `AC_START_0` ~ `AC_START_N` | 与 PIT 数量匹配 | 静止起步发车格 |
| `AC_TIME_N_L` / `AC_TIME_N_R` | 成对 | 计时点左右边界，每个组合弯一个 |

生成策略：VLM（Gemini 2.5 Pro）生成布局依赖对象（hotlap/pit/start），程序化中线分析生成计时点（timing）。所有对象经过 snap-to-road 后处理确保落在合理位置。

## Web 编辑器套件

基于 Leaflet.js 的交互式编辑器，用于 AI 输出的手动精调：

| 编辑器 | 用途 |
|--------|------|
| Session Analyzer | GPS 遥测数据可视化 + 圈速分析 |
| Centerline Editor | 赛道中线交互绘制与调整 |
| Layout Editor | 赛道布局（多布局支持）编辑 |
| Wall Editor | 围墙边界交互编辑 |
| Objects Editor | 游戏对象位置/朝向交互编辑 |
| Game Objects Editor | 游戏对象高级编辑器 |

启动方式：`python script/track_session_anaylzer/run_analyzer.py`
