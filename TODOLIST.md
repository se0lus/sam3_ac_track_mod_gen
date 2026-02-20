# TODO List

> **使用说明**: 用户在此文件中添加待办任务，Claude Agent Team 会自动读取并执行。
>
> - 新任务写在 `## 待办` 下，格式: `- [ ] 任务描述`
> - 进行中的任务会被移到 `## 进行中`，格式: `- [~] 任务描述`
> - 完成的任务会被移到 `## 已完成`，格式: `- [x] 任务描述 ✓ (完成备注)`
> - 优先级从上到下递减（最上面的最优先）

---

## 待办

- [ ] 完善step10赛道分割+导出；具体步骤：1. 从stage9拿到blender结果文件和texture纹理目录；2. 删除掉mask多边形和mask曲线，确保项目中只有即将要导入游戏的对象；3. 游戏中对单个网格的尺寸有约束，需要确保每个mesh对象的顶点数不超过21000；如果超过了，则需要对他进行拆分；拆分的方式是如果对象是1ROAD，则按赛道中心线分段拆分（需要做的比较干净不杂乱）；其他类型的对象则按XZ平面进行2D横竖2分，需要确保尽量少的拆分；拆分完成之后，针对碰撞对象的重命名，需要按1ROAD_1,1ROAD_2类似的方式依次命名，不能重名，其他对象则没有命名规则要求，不重名即可；4.对象分批，由于游戏中会将整个赛道分片加载，每个片对应我们的一次导出，因此我们需要先对所有的对象进行分类，放到对应的导出批次的collection里面；具体要求是每个批次最终导出的fbx文件尺寸不超过100MB；批次优先按：1.赛道碰撞路面和AC对象；2.其他碰撞对象；3.高精度环境，4.低精度环境 的顺序，如果每个批次的尺寸太大，还需要进行进一步拆分；拆分完成后保存一下blender文件以备检查；5.导出fbx文件，将上一步组织好的每个导出批次collection分别导出成一个fbx文件，检查一下最终文件的尺寸，导出参数：scale 0.01, -y forward, z up；

## 进行中


## 已完成
- [x] Stage 9 非无头模式视口可视化 ✓ (正交俯视 + timer 延迟 5s + 瓦片逐步加载刷新 + 节流重绘 + Windows 消息泵送防冻结 + 基础瓦片后隐藏 mask)
- [x] 在步骤9中为网格提取工具增加一个简化网格的配置 ✓ (s9_mesh_simplify 开关 + weld 0.01m + decimate 0.5, 含 Dashboard UI + CLI 全链路)
- [x] 构建步骤10，基于步骤9或者9a（如果启用），进行模型拆分，具体如何拆分后面再写要求，先把框架搭建出来 ✓ (s10_model_export.py 骨架 + pipeline_config + sam3_track_gen + dashboard 注册，读取 09_result/final_track.blend)
- [x] 构建步骤9a，复制步骤9结果供用户在Blender中手动编辑 ✓ (s09a_manual_blender.py + pipeline_config + dashboard 注册，含 09_result junction)
- [x] 网格提取工具：对于kerb/road，使用原生3D瓦片网格作为碰撞网格。根据mask多边形选区，竖直投影到3D瓦片，复制出匹配区域。kerb+road一起提取后分隔，确保无重叠且边缘对齐。 ✓ (`terrain_mesh_extractor.py` — BVHTree mask containment + terrain face copy)
- [x] 布尔网格生成工具：根据mask和网格密度创建Grid平面，mask实体化后布尔裁切，投影到地形表面。含degenerate geometry预清理。 ✓ (`boolean_mesh_generator.py` — Grid + Solidify + Boolean Intersect + terrain projection)
- [x] 网格提取工具 和 布尔网格生成工具 在独立用例上测试好 ✓ (Blender headless test + 6项自动验证全PASS，含degenerate face修复)
- [x] 把网格提取工具和布尔网格生成工具应用在步骤9中 ✓ (Step 5 拆分为 5a terrain extraction + 5b boolean surfaces，Stage 9 完整运行 FINISHED)

