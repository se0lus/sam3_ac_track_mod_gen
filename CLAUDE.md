# SAM3 赛道分割 — Assetto Corsa 赛道 Mod 自动生成器

无人机 2D 航拍 / 3D 倾斜摄影 → SAM3 语义分割 → Blender 三维处理 → Assetto Corsa 赛道 Mod。

详细项目文档见 [project.md](project.md)。

## 关键路径

| 路径 | 用途 |
|------|------|
| `script/pipeline_config.py` | **统一配置（唯一来源）** |
| `script/sam3_track_gen.py` | 主入口 CLI |
| `script/stages/s0N_*.py` | 各阶段（可独立运行） |
| `script/webTools/run_webtools.py` | Dashboard + Web 编辑器 |
| `blender_scripts/config.py` | Blender 配置（从 pipeline_config 导入） |
| `blender_scripts/sam3_actions/` | Blender 右键菜单 Action 插件 |
| `script/surface_extraction.py` | 碰撞命名 + 多边形工具（纯 Python，无 bpy） |
| `output/NN_stage_name/` | 各阶段输出 |
| `test_images_shajing/` | 测试数据集（只读，禁止写入） |
| `tests/` | 单元测试，镜像源代码结构 |

## 开发规则

1. **统一配置**: 所有配置集中在 `script/pipeline_config.py`，不要硬编码路径或参数。
2. **阶段独立**: 每个阶段 (`script/stages/s0N_*.py`) 必须可独立运行和测试。
3. **输出隔离**: 所有输出在 `output/` 下，绝不写入 `test_images_shajing/`。
4. **可测试性**: 耦合 Blender 的功能要隔离，大部分代码可脱离 Blender 测试。
5. **测试位置**: 测试文件放 `tests/`，镜像源代码结构。
6. **Blender Action**: 新 Action 放 `blender_scripts/sam3_actions/`，导出 `ACTION_SPECS` 列表，由 `blender_helpers.py` 自动发现注册。
7. **碰撞命名**: `1WALL_N`, `1ROAD_N`, `1SAND_N`, `1KERB_N`, `1GRASS_N`, `2ROAD_N`。使用 `surface_extraction.generate_collision_name()`。
8. **先计划后编码**: 每个模块编码前需完成计划并经 review。
9. **先测试后集成**: 模块测试通过后才可集成。
10. **物理空间优先**: 算法参数用物理单位（米、度），运行时按 GeoTIFF 分辨率转像素。

## 关键约定

- **碰撞 Collection**: 按 tag 分 — `collision_road`, `collision_kerb`, `collision_grass`, `collision_sand`, `collision_road2`；围墙用 `collision`。映射在 `surface_extraction.COLLISION_COLLECTION_MAP`。
- **Mask Collection**: 根 `mask_polygon_collection`，子 `mask_polygon_{tag}`。名称在 `config.ROOT_POLYGON_COLLECTION_NAME`。
- **采样密度**: road/kerb 0.1m, grass/sand/road2 2.0m。配置在 `config.SURFACE_SAMPLING_DENSITY_*`。
- **地形瓦片**: 在 `L{digits}` collection 中（如 L17, L18 ...）。
- **Blender 坐标系**: Y 轴朝上，mask 在 XZ 平面，raycast 方向 (0, -1, 0)。

## Web 前端规范 (webTools)

- **整洁统一**: 所有编辑器页面保持一致的视觉风格，复用 `style.css` 的 CSS 变量（`--bg`, `--panel`, `--border` 等）。
- **现代控件**: 禁止使用浏览器原生 radio/checkbox，用自定义 toggle switch、pill selector 等替代（参考 `dashboard.css` 中 `.s9-toggle` / `.s9-tag-pill`）。
- **滚动条**: 禁止嵌套滚动条。全局隐藏默认滚动条（`scrollbar-width: none`），需要滚动的容器用细滚动条（`scrollbar-width: thin; scrollbar-color: #333 transparent`）。
- **地图交互**: Leaflet 地图统一为**右键长按拖动**（非默认左键拖动），保持所有编辑器一致。
- **深色主题**: 背景 `#0b0f17` / `#1a1a2e`，面板半透明渐变，文字 `#e5e7eb`，muted `#94a3b8`，强调色 `#60a5fa`。

## 任务管理（两层机制）

### 第一层：`TODOLIST.md`（跨会话，用户驱动）

用户在此文件中管理长期任务，agent 每次会话开头读取。

```
## 待办        - [ ] 任务（优先级从上到下递减）
## 进行中      - [~] 任务
## 已完成      - [x] 任务 ✓ (备注)
```

- 用户随时可增删改任务，agent 每次循环前重新读取
- 任务需用户确认时标注等待原因，跳到下一个可执行任务

### 第二层：内置 Task 工具（会话内，agent 驱动）

`TaskCreate` / `TaskUpdate` / `TaskList` 用于会话内子任务拆分和进度跟踪。

- 从 `TODOLIST.md` 取到一个大任务后，用 `TaskCreate` 拆分为可执行的子步骤
- 子步骤间可设依赖（`blockedBy`），Team 模式下可分派给不同 agent（`owner`）
- 用户在终端实时看到当前执行的子任务状态
- 会话结束时自动清除，不持久化

### 协作流程

```
TODOLIST.md（用户写）→ agent 读取最高优先级任务
  → TaskCreate 拆分子步骤 → 逐步执行（TaskUpdate 跟踪进度）
  → 全部完成 → 更新 TODOLIST.md 标记 [x] ✓
```
