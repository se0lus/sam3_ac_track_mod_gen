# TODO List

> **使用说明**: 用户在此文件中添加待办任务，Claude Agent Team 会自动读取并执行。
>
> - 新任务写在 `## 待办` 下，格式: `- [ ] 任务描述`
> - 进行中的任务会被移到 `## 进行中`，格式: `- [~] 任务描述`
> - 完成的任务会被移到 `## 已完成`，格式: `- [x] 任务描述 ✓ (完成备注)`
> - 优先级从上到下递减（最上面的最优先）

---

## 待办

- [ ] 网格提取工具 和 布尔网格生成工具 在一些独立用例上测试好
- [ ] 把 网格提取工具 和 布尔网格生成工具应用在步骤9中，替代现在的remesh实现，其中road和kerb应用网格提取工具，其他的地面用间距2米采样的布尔网格生成工具，然后直接采样贴合到瓦片表面；最后完整的运行一遍步骤9进行验证；

## 进行中


## 已完成

- [x] 网格提取工具：对于kerb/road，使用原生3D瓦片网格作为碰撞网格。根据mask多边形选区，竖直投影到3D瓦片，复制出匹配区域。kerb+road一起提取后分隔，确保无重叠且边缘对齐。 ✓ (`terrain_mesh_extractor.py` — BVHTree mask containment + terrain face copy)
- [x] 布尔网格生成工具：根据mask和网格密度创建Grid平面，mask实体化后布尔裁切，投影到地形表面。含degenerate geometry预清理。 ✓ (`boolean_mesh_generator.py` — Grid + Solidify + Boolean Intersect + terrain projection)

