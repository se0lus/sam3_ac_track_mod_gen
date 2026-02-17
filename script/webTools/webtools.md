# webTools — SAM3 赛道生成工具 Web Dashboard + 编辑器

## 概述

`script/webTools/` 包含 Web Dashboard 和所有交互编辑器，通过统一的 HTTP 服务器提供。

## 文件结构

| 文件 | 说明 |
|------|------|
| `run_webtools.py` | 主 HTTP 服务器（Dashboard + 编辑器 + Pipeline API） |
| `dashboard.html/js/css` | Dashboard 主页（流水线可视化 + 执行控制 + 文件浏览） |
| `analyzer.html/js` | GPS 走线分析器（原 index.html + app.js） |
| `style.css` | 全局共享样式 |
| `map_rightdrag.js` | 右键拖拽共享工具 |
| `*_editor.html/js/css` | 各编辑器（centerline/layout/wall/objects/surface/gameobjects） |

## 启动方式

```bash
# 默认打开 Dashboard
python script/webTools/run_webtools.py

# 直接打开指定页面
python script/webTools/run_webtools.py --page analyzer.html
python script/webTools/run_webtools.py --page wall_editor.html

# 向后兼容
python script/track_session_anaylzer/run_analyzer.py
```

## API 端点

### 编辑器 API（已有）

| 方法 | 路径 | 说明 |
|------|------|------|
| GET/POST | `/api/walls` | 围墙 JSON |
| GET/POST | `/api/game_objects` | 游戏对象 JSON |
| GET/POST | `/api/track_layouts` | 赛道布局 JSON |
| GET | `/api/geo_metadata` | 地理元数据 |
| GET | `/api/centerline` | 中线数据 |
| GET | `/api/modelscale_image` | modelscale 图像 |
| GET/POST | `/api/layout_mask/{name}` | 布局 mask PNG |
| GET/POST | `/api/surface_mask/{tag}` | 表面 mask PNG |
| POST | `/api/centerline/regenerate` | 重新生成中线 |
| POST | `/api/vlm_objects/regenerate` | 重新生成 VLM 对象 |

### Dashboard API（新增）

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/pipeline/stages` | 阶段列表 + 元数据 |
| GET | `/api/pipeline/status` | 各阶段状态 |
| GET/POST | `/api/pipeline/config` | 获取/保存配置 |
| POST | `/api/pipeline/run` | 后台执行指定阶段 |
| POST | `/api/pipeline/stop` | 停止执行 |
| GET | `/api/files/list?path=...` | 文件列表 |
| GET | `/api/files/preview?path=...` | 文件预览 |
| GET | `/api/sse/pipeline` | SSE 日志 + 进度推送 |
| GET | `/tiles/{z}/{x}/{y}.png` | 离线瓦片代理 |

### 地图瓦片

瓦片文件位于 `test_images_shajing/map/`，通过 `/tiles/{z}/{x}/{y}.png` 路由提供。
所有编辑器和分析器的底图 URL 已统一为 `/tiles/` 路由。

## 配置持久化

Dashboard 配置保存在 `output/webtools_config.json`，包含：
- geotiff_path, tiles_dir, output_dir
- blender_exe, track_direction, gemini_api_key
