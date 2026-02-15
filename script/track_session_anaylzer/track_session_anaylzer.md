开发要求：
创建一个基于pc本地的赛道数据分析工具，可以是html前端页面
底图输入：在 test_session_record/map中，是根据leafleft方式切分好的高精度赛道地图瓦片，基于WGS84坐标系
分析输入：类似 test_session_record/session_20250223_112341_sz_karting_anti-cw_v3.csv;

可视化要求：
1.一个可放大，缩小拖动的底图
2.可以选择多组csv数据进行分析
3.针对每一组csv数据，使用不同的颜色，csv中的所有轨迹线在地图上绘制出来（要考虑性能，不能卡顿）
4.针对每一组csv数据，需要有一个独立的经度纬度的偏移调节，可以用于手动对轨迹线的位置进行一下微调，微调步长为+-1米 以及 +-0.1 米两种，交互要简洁清晰

---

已实现（本地网页工具）

位置：
- `script/track_session_anaylzer/index.html`
- `script/track_session_anaylzer/app.js`
- `script/track_session_anaylzer/style.css`
- `script/track_session_anaylzer/run_analyzer.ps1`

使用方法（推荐）：
1. 方式 A（推荐，最稳）：Python 直接启动
   - `python .\script\track_session_anaylzer\run_analyzer.py`
2. 方式 B（PowerShell wrapper，内部也是调用 Python）：
   - `.\script\track_session_anaylzer\run_analyzer.ps1`
3. 浏览器会自动打开（端口会自动选择可用端口，不一定是 8000）：
   - `http://127.0.0.1:<port>/script/track_session_anaylzer/index.html`
4. 在页面左侧选择一个或多个 CSV（支持多选）

功能说明：
- 底图：加载 `script/track_session_anaylzer/map/{z}/{x}/{y}.png` 本地瓦片（与页面同目录；需要通过本地 HTTP 服务访问，直接双击打开 `file://` 可能会加载失败）
- 多组 CSV：每份 CSV 自动分配颜色，轨迹用 Canvas 渲染（Leaflet preferCanvas）
- 轨迹分段：按 `fragment_id` / 时间断点 / 空间跳变自动断开，避免跨段直连
- 每组偏移：每个数据集卡片都有独立“东/北（米）”偏移，步长可选 1m 或 0.1m；支持一键归零
- 性能：提供“抽稀阈值（米）”滑条，点很密时可调大以提升流畅度