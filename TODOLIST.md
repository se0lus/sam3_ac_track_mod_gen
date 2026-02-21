# TODO List

> **使用说明**: 用户在此文件中添加待办任务，Claude Agent Team 会自动读取并执行。
>
> - 新任务写在 `## 待办` 下，格式: `- [ ] 任务描述`
> - 进行中的任务会被移到 `## 进行中`，格式: `- [~] 任务描述`
> - 完成的任务会被移到 `## 已完成`，格式: `- [x] 任务描述 (完成备注)`
> - 优先级从上到下递减（最上面的最优先）

---

## 待办


## 进行中


## 已完成
- [x] Stage 11 细节优化 (全小写文件夹 + layout命名取自上游 + models_{layout}.ini + ui/子目录对齐 + TrackMapGenerator map.png→layout/ + data/map.ini→layout/data/ + 布局显示名Dashboard配置 + LLM描述禁用改模板 + INI文件预览 + CLI layout-display-names/track-url 透传)
- [x] Stage 11 赛道打包 (KN5 分类复制保持原名 + models_*.ini 自动生成 + map.ini/cameras.ini 从 centerline 计算 + ui_track.json 自动填充(描述为用户手填或模板) + preview.png 航拍裁剪或占位图 + outline.png 从 map.png 复制 + Dashboard 赛道信息配置含描述字段 + CLI/webtools 全链路集成)
- [x] Stage 10 FBX INI 生成 + 纹理拷贝 (FBX to KN5 的 .fbx.ini 材质配置自动生成，碰撞/地形/游戏对象分类处理，ksAmbient/ksDiffuse/ksEmissive 参数暴露到 Dashboard，纹理从 Stage 9 拷贝，KN5 65535 顶点限制自动拆分，多布局游戏对象批次分组)
- [x] Stage 10 KN5 转换路径修复 (ksEditorAt.exe 需要绝对路径，改用 kn5_path/fbx_path 替代 basename)
- [x] Stage 10 模型导出 (弧长近邻投影拆分 road + 顶点分布中位数递归拆分其他 + AC 碰撞命名 + 自动检测瓦片层级 + 分批 FBX 导出无纹理 + 导出前清理非批次数据 + Dashboard UI)
- [x] Stage 9 非无头模式视口可视化 (正交俯视 + timer 延迟 5s + 瓦片逐步加载刷新 + 节流重绘 + Windows 消息泵送防冻结 + 基础瓦片后隐藏 mask)
- [x] 在步骤9中为网格提取工具增加一个简化网格的配置 (s9_mesh_simplify 开关 + weld 0.01m + decimate 0.5, 含 Dashboard UI + CLI 全链路)
- [x] 构建步骤10，基于步骤9或者9a（如果启用），进行模型拆分 (s10_model_export.py 骨架 + pipeline_config + sam3_track_gen + dashboard 注册，读取 09_result/final_track.blend)
- [x] 构建步骤9a，复制步骤9结果供用户在Blender中手动编辑 (s09a_manual_blender.py + pipeline_config + dashboard 注册，含 09_result junction)
- [x] 网格提取工具：kerb/road 使用原生3D瓦片网格作为碰撞网格 (terrain_mesh_extractor.py -- BVHTree mask containment + terrain face copy)
- [x] 布尔网格生成工具：根据mask和网格密度创建Grid平面 (boolean_mesh_generator.py -- Grid + Solidify + Boolean Intersect + terrain projection)
- [x] 网格提取工具和布尔网格生成工具在独立用例上测试好 (Blender headless test + 6项自动验证全PASS，含degenerate face修复)
- [x] 把网格提取工具和布尔网格生成工具应用在步骤9中 (Step 5 拆分为 5a terrain extraction + 5b boolean surfaces，Stage 9 完整运行 FINISHED)
