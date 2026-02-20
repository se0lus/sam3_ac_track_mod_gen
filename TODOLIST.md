# TODO List

> **使用说明**: 用户在此文件中添加待办任务，Claude Agent Team 会自动读取并执行。
>
> - 新任务写在 `## 待办` 下，格式: `- [ ] 任务描述`
> - 进行中的任务会被移到 `## 进行中`，格式: `- [~] 任务描述`
> - 完成的任务会被移到 `## 已完成`，格式: `- [x] 任务描述 ✓ (完成备注)`
> - 优先级从上到下递减（最上面的最优先）

---

## 待办

-[ ] 在步骤9的非无头模式下，希望用户能够看到blender中的对象在逐步加载的可视化过程，可以考虑在blender文件创建好，但是还没开始加载其他瓦片之前，先等待一小段时间（例如5秒），让blender整个主界面加载完全，然后把视口摄像机放到能俯视整个赛道的位置，并且把远平面拉到足够远（10000米）的位置，避免画面裁切，可以考虑把摄像机模式改为正交；之后再继续原有的操作


## 进行中


## 已完成
- [x] 在步骤9中为网格提取工具增加一个简化网格的配置 ✓ (s9_mesh_simplify 开关 + weld 0.01m + decimate 0.5, 含 Dashboard UI + CLI 全链路)
- [x] 构建步骤10，基于步骤9或者9a（如果启用），进行模型拆分，具体如何拆分后面再写要求，先把框架搭建出来 ✓ (s10_model_export.py 骨架 + pipeline_config + sam3_track_gen + dashboard 注册，读取 09_result/final_track.blend)
- [x] 构建步骤9a，复制步骤9结果供用户在Blender中手动编辑 ✓ (s09a_manual_blender.py + pipeline_config + dashboard 注册，含 09_result junction)
- [x] 网格提取工具：对于kerb/road，使用原生3D瓦片网格作为碰撞网格。根据mask多边形选区，竖直投影到3D瓦片，复制出匹配区域。kerb+road一起提取后分隔，确保无重叠且边缘对齐。 ✓ (`terrain_mesh_extractor.py` — BVHTree mask containment + terrain face copy)
- [x] 布尔网格生成工具：根据mask和网格密度创建Grid平面，mask实体化后布尔裁切，投影到地形表面。含degenerate geometry预清理。 ✓ (`boolean_mesh_generator.py` — Grid + Solidify + Boolean Intersect + terrain projection)
- [x] 网格提取工具 和 布尔网格生成工具 在独立用例上测试好 ✓ (Blender headless test + 6项自动验证全PASS，含degenerate face修复)
- [x] 把网格提取工具和布尔网格生成工具应用在步骤9中 ✓ (Step 5 拆分为 5a terrain extraction + 5b boolean surfaces，Stage 9 完整运行 FINISHED)

