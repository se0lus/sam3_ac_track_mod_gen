import numpy as np
from typing import List, Tuple, Union, cast
from PIL import Image
try:
    from scipy import ndimage  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    ndimage = None  # type: ignore

try:
    import matplotlib.pyplot as plt  # type: ignore
    import matplotlib.patches as patches  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    # 可选依赖：核心裁剪算法不需要matplotlib，仅可视化才需要
    plt = None  # type: ignore
    patches = None  # type: ignore


def _binary_dilation_rect_numpy(mask: np.ndarray, dilation_h_px: int, dilation_w_px: int) -> np.ndarray:
    """
    纯numpy实现的矩形膨胀（用于缺少scipy时的降级方案）。
    使用积分图在 O(HW) 时间内完成任意矩形窗口的“是否存在True”查询。
    """
    if dilation_h_px <= 0 and dilation_w_px <= 0:
        return mask
    h, w = mask.shape
    dh = max(0, int(dilation_h_px))
    dw = max(0, int(dilation_w_px))
    ph = dh
    pw = dw
    padded = np.pad(mask.astype(np.uint8), ((ph, ph), (pw, pw)), mode="constant", constant_values=0)
    # integral image with 1-based offset
    ii = np.pad(np.cumsum(np.cumsum(padded, axis=0), axis=1), ((1, 0), (1, 0)), mode="constant", constant_values=0)
    win_h = 2 * dh + 1
    win_w = 2 * dw + 1
    y0 = np.arange(h)
    x0 = np.arange(w)
    y1 = y0 + win_h
    x1 = x0 + win_w
    # broadcast to compute sums
    sum_win = (
        ii[np.ix_(y1, x1)]
        - ii[np.ix_(y0, x1)]
        - ii[np.ix_(y1, x0)]
        + ii[np.ix_(y0, x0)]
    )
    return sum_win > 0


def _edge_contact_stats(
    box: Tuple[float, float, float, float],
    mask: np.ndarray,
    width: int,
    height: int,
    border_thickness_px: int
) -> Tuple[float, float, float, float, int, int, int, int, int, int, int, int]:
    """
    返回边界strip在mask内的计数/尺寸信息，方便构造多种惩罚项。

    Returns:
        top_cnt, bottom_cnt, left_cnt, right_cnt,
        top_size, bottom_size, left_size, right_size,
        x_span, y_span, t, perimeter_strip_size
    """
    if border_thickness_px <= 0:
        return (0.0, 0.0, 0.0, 0.0, 1, 1, 1, 1, 0, 0, 0, 1)

    x_min, y_min, x_max, y_max = box
    x0 = int(round(x_min * width))
    y0 = int(round(y_min * height))
    x1 = int(round(x_max * width))
    y1 = int(round(y_max * height))

    x0 = max(0, min(x0, width))
    x1 = max(0, min(x1, width))
    y0 = max(0, min(y0, height))
    y1 = max(0, min(y1, height))
    if x1 <= x0 or y1 <= y0:
        return (0.0, 0.0, 0.0, 0.0, 1, 1, 1, 1, 0, 0, 0, 1)

    t = int(border_thickness_px)
    t = max(1, t)
    t = min(t, max(1, (x1 - x0) // 2), max(1, (y1 - y0) // 2))

    y_top1 = min(y1, y0 + t)
    y_bot0 = max(y0, y1 - t)
    top_m = mask[y0:y_top1, x0:x1]
    bottom_m = mask[y_bot0:y1, x0:x1]

    y_mid0 = y_top1
    y_mid1 = y_bot0
    if y_mid1 < y_mid0:
        y_mid0, y_mid1 = y0, y1

    x_left1 = min(x1, x0 + t)
    x_right0 = max(x0, x1 - t)
    left_m = mask[y_mid0:y_mid1, x0:x_left1]
    right_m = mask[y_mid0:y_mid1, x_right0:x1]

    top_cnt = float(np.sum(top_m))
    bottom_cnt = float(np.sum(bottom_m))
    left_cnt = float(np.sum(left_m))
    right_cnt = float(np.sum(right_m))

    top_size = int(top_m.size)
    bottom_size = int(bottom_m.size)
    left_size = int(left_m.size)
    right_size = int(right_m.size)

    x_span = int(x1 - x0)
    y_span = int(y1 - y0)
    perimeter_strip_size = max(1, top_size + bottom_size + left_size + right_size)
    return (
        top_cnt, bottom_cnt, left_cnt, right_cnt,
        max(1, top_size), max(1, bottom_size), max(1, left_size), max(1, right_size),
        x_span, y_span, t, perimeter_strip_size
    )


def _border_cut_cost(
    box: Tuple[float, float, float, float],
    mask: np.ndarray,
    width: int,
    height: int,
    border_thickness_px: int,
    dist_in_mask: Union[np.ndarray, None] = None,
    edge_total_weight: float = 1.0,
    edge_concentration_weight: float = 2.0,
    concentration_power: float = 2.0,
    core_weight: float = 0.1
) -> float:
    """
    计算一个clip边界“切到mask”的代价（越大越糟）。

    这里把 cut_weight 的含义改成更贴合需求：
    - 主要惩罚：clip四条边上有多少“边界长度”落在mask内（越多越糟）
    - 额外惩罚：mask是否“集中落在某一条边上”（例如整条上边都在mask内）——用幂次/平方加重
    - 可选的轻微附加项：如果提供 dist_in_mask，再对“切到核心”做一点点惩罚（core_weight较小）

    这样对于“水平细长路面”：
    - 若用水平边把路面分成两半：top/bottom某一条边会有很长一段在mask内 -> 代价很大
    - 若从中间竖切：left/right两条边各只覆盖路面宽度的一小段 -> 代价更小（更符合诉求）
    """
    eps = 1e-9
    (
        top_cnt, bottom_cnt, left_cnt, right_cnt,
        top_size, bottom_size, left_size, right_size,
        x_span, y_span, t, _
    ) = _edge_contact_stats(box, mask, width, height, border_thickness_px)

    edge_total = float(top_cnt + bottom_cnt + left_cnt + right_cnt)

    def _conc(cnt: float, size: int) -> float:
        s = float(size) + eps
        p = float(concentration_power)
        # cnt^p / s^(p-1) 的量纲仍为“像素数”，且当cnt接近s时会显著变大
        return (cnt ** p) / (s ** (p - 1.0))

    edge_conc = _conc(top_cnt, top_size) + _conc(bottom_cnt, bottom_size) + _conc(left_cnt, left_size) + _conc(right_cnt, right_size)

    # 可选：切到核心的轻微惩罚（默认权重很小）
    core_pen = 0.0
    if dist_in_mask is not None and core_weight > 0:
        # 缺少scipy时 dist_in_mask 可能为None，此分支不会进入
        x_min, y_min, x_max, y_max = box
        x0 = max(0, min(int(round(x_min * width)), width))
        y0 = max(0, min(int(round(y_min * height)), height))
        x1 = max(0, min(int(round(x_max * width)), width))
        y1 = max(0, min(int(round(y_max * height)), height))
        if x1 > x0 and y1 > y0:
            # 只取边界strip，并用mask筛选
            y_top1 = min(y1, y0 + t)
            y_bot0 = max(y0, y1 - t)
            top_m = mask[y0:y_top1, x0:x1]
            bottom_m = mask[y_bot0:y1, x0:x1]
            y_mid0 = y_top1
            y_mid1 = y_bot0
            if y_mid1 < y_mid0:
                y_mid0, y_mid1 = y0, y1
            x_left1 = min(x1, x0 + t)
            x_right0 = max(x0, x1 - t)
            left_m = mask[y_mid0:y_mid1, x0:x_left1]
            right_m = mask[y_mid0:y_mid1, x_right0:x1]
            vals: List[np.ndarray] = []
            if np.any(top_m):
                vals.append(dist_in_mask[y0:y_top1, x0:x1][top_m])
            if np.any(bottom_m):
                vals.append(dist_in_mask[y_bot0:y1, x0:x1][bottom_m])
            if np.any(left_m):
                vals.append(dist_in_mask[y_mid0:y_mid1, x0:x_left1][left_m])
            if np.any(right_m):
                vals.append(dist_in_mask[y_mid0:y_mid1, x_right0:x1][right_m])
            if len(vals) > 0:
                allv = np.concatenate(vals) if len(vals) > 1 else vals[0]
                if allv.size > 0:
                    core_pen = float(np.mean(allv)) * float(edge_total)

    return float(edge_total_weight) * float(edge_total) + float(edge_concentration_weight) * float(edge_conc) + float(core_weight) * float(core_pen)


def _check_overlap_requirement(
    box1: Tuple[float, float, float, float],
    box2: Tuple[float, float, float, float],
    clip_size: Tuple[float, float],
    min_overlap_ratio: float
) -> bool:
    """
    检查两个box是否满足最小重叠要求。
    只有当两个box相邻（距离很近）时，才需要满足最小重叠要求。
    
    Args:
        box1, box2: 格式为 (x_min, y_min, x_max, y_max)
        clip_size: clip尺寸 (width, height)
        min_overlap_ratio: 最小重叠比例（相对于clip尺寸）
    
    Returns:
        如果两个box不相邻，或相邻且满足最小重叠要求，返回True
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    clip_w, clip_h = clip_size
    
    # 计算两个box的中心距离
    center1_x = (x1_min + x1_max) / 2
    center1_y = (y1_min + y1_max) / 2
    center2_x = (x2_min + x2_max) / 2
    center2_y = (y2_min + y2_max) / 2
    
    dist_x = abs(center1_x - center2_x)
    dist_y = abs(center1_y - center2_y)
    
    # 如果距离较远（超过1.5倍clip尺寸），不需要重叠
    if dist_x > clip_w * 1.5 or dist_y > clip_h * 1.5:
        return True
    
    # 距离较近，需要检查重叠
    # 计算重叠区域
    overlap_x_min = max(x1_min, x2_min)
    overlap_y_min = max(y1_min, y2_min)
    overlap_x_max = min(x1_max, x2_max)
    overlap_y_max = min(y1_max, y2_max)
    
    if overlap_x_max <= overlap_x_min or overlap_y_max <= overlap_y_min:
        # 没有重叠，不满足要求
        return False
    
    # 有重叠，检查重叠是否满足要求
    overlap_w_actual = overlap_x_max - overlap_x_min
    overlap_h_actual = overlap_y_max - overlap_y_min
    
    # 如果重叠区域在某个维度上满足最小重叠要求，则认为满足
    if overlap_w_actual >= min_overlap_ratio * clip_w or overlap_h_actual >= min_overlap_ratio * clip_h:
        return True
    
    return False


def _get_coverage_score(
    box: Tuple[float, float, float, float],
    uncovered_mask: np.ndarray,
    width: int,
    height: int
) -> float:
    """
    计算一个box覆盖未覆盖区域的得分。
    
    Args:
        box: 格式为 (x_min, y_min, x_max, y_max)，归一化坐标
        uncovered_mask: 未覆盖区域的mask（像素坐标）
        width, height: 图像尺寸
    
    Returns:
        覆盖的未覆盖像素数量
    """
    x_min, y_min, x_max, y_max = box
    
    # 转换为像素坐标
    x_min_px = max(0, int(x_min * width))
    y_min_px = max(0, int(y_min * height))
    x_max_px = min(width, int(x_max * width))
    y_max_px = min(height, int(y_max * height))
    
    if x_max_px <= x_min_px or y_max_px <= y_min_px:
        return 0.0
    
    # 计算覆盖的未覆盖像素数量
    region = uncovered_mask[y_min_px:y_max_px, x_min_px:x_max_px]
    return float(np.sum(region))


def _greedy_cover_global(
    mask: np.ndarray,
    clip_size: Tuple[float, float],
    min_overlap_ratio: float,
    width: int,
    height: int,
    search_step: float = 0.01,
    dist_in_mask: Union[np.ndarray, None] = None,
    cut_weight: float = 1.0,
    border_thickness_ratio: float = 0.02,
    orig_mask: Union[np.ndarray, None] = None,
    density_weight: float = 0.15,
    min_uncovered_ratio: float = 0.03,
    min_mask_fill_ratio: float = 0.08,
    fill_penalty_weight: float = 1.0,
    progress_relax_ratio: float = 0.12,
    # 新增：显式“坏切法”惩罚与上下文约束
    bad_split_weight: float = 2.0,
    min_context_ratio: float = 0.12,
    # 新增：集合覆盖的惩罚放大与候选采样控制
    penalty_scale: float = 8.0,
    max_sample_points: int = 240,
    max_candidates: int = 2400,
    # 新增：硬约束式过滤，避免“某一条边几乎整条都贴在mask里”
    max_edge_contact_ratio: float = 0.75,
    max_edge_contact_ratio_late: float = 0.9
) -> List[Tuple[float, float, float, float]]:
    """
    使用全局贪心算法覆盖整个mask，使用最少的clip数量。
    允许一个clip覆盖多个连通区域，实现全局最优。
    
    Args:
        mask: 整个mask（像素坐标）
        clip_size: clip尺寸 (width, height)，归一化
        min_overlap_ratio: 最小重叠比例（相对于clip尺寸）
        width, height: 图像尺寸
        search_step: 搜索步长（归一化），用于生成候选位置
        dist_in_mask: 原始mask的距离变换（ndimage.distance_transform_edt(mask)），可选，用于轻微惩罚“切到核心”
        cut_weight: 边界切割惩罚权重（越大越倾向让clip边缘尽量少落在mask内，尤其避免“单边整条落在mask里”）
        border_thickness_ratio: 计算边界代价时使用的边界strip厚度，占clip短边的比例
        orig_mask: 用于计算clip内mask填充率的原始mask（建议传入未膨胀版本）
        density_weight: clip内mask像素奖励权重（鼓励每个clip“装得更满”）
        min_uncovered_ratio: 新增覆盖阈值（相对clip面积）。在覆盖进度较早时用于过滤“只蹭一点点”的clip，后期自动放宽
        min_mask_fill_ratio: clip内mask填充率的目标下限（相对clip面积）。低于该值会被惩罚，后期自动放宽
        fill_penalty_weight: 低填充惩罚权重（像素级）
        progress_relax_ratio: 进度放宽阈值（剩余未覆盖/总mask）。小于该值时逐步放宽最小新增覆盖与最小填充率
        bad_split_weight: “对边同时贴mask”的惩罚权重（用于避免把路面一条带状区域切成上下/左右两半）
        min_context_ratio: clip内至少需要的背景/留白比例（用于保留环境信息做进一步分割）
        penalty_scale: 将边界/坏切法惩罚放大进入 cost 的系数（越大越不愿切mask）
        max_sample_points: 每轮从未覆盖像素中采样的点数上限（用于生成局部候选）
        max_candidates: 每轮候选框上限（过大可能变慢）
        max_edge_contact_ratio: 单边贴mask比例上限（0-1）。越小越避免“整条边落在mask内”
        max_edge_contact_ratio_late: 后期补洞阶段允许的上限（建议 < 1.0，仍避免“整条边完全贴mask”）
    
    Returns:
        归一化的box列表
    """
    clip_w_norm, clip_h_norm = clip_size
    
    # 找到mask的有效区域边界框
    valid_pixels = np.where(mask)
    if len(valid_pixels[0]) == 0:
        return []
    
    min_y, max_y = valid_pixels[0].min(), valid_pixels[0].max() + 1
    min_x, max_x = valid_pixels[1].min(), valid_pixels[1].max() + 1
    
    # 转换为归一化坐标
    min_x_norm = max(0.0, min_x / width)
    min_y_norm = max(0.0, min_y / height)
    max_x_norm = min(1.0, max_x / width)
    max_y_norm = min(1.0, max_y / height)
    
    # 计算边距
    margin_w_norm = min_overlap_ratio * clip_w_norm
    margin_h_norm = min_overlap_ratio * clip_h_norm
    
    # 扩展边界以包含边距
    search_min_x = max(0.0, min_x_norm - margin_w_norm)
    search_min_y = max(0.0, min_y_norm - margin_h_norm)
    search_max_x = min(1.0 - clip_w_norm, max_x_norm + margin_w_norm)
    search_max_y = min(1.0 - clip_h_norm, max_y_norm + margin_h_norm)
    
    if search_max_x <= search_min_x or search_max_y <= search_min_y:
        # 区域太小，返回单个clip
        return [(search_min_x, search_min_y, search_min_x + clip_w_norm, search_min_y + clip_h_norm)]
    
    # 初始化未覆盖区域（整个mask）
    uncovered_mask = mask.copy()
    
    selected_boxes = []
    
    # 全局贪心选择：每次选择能覆盖最多未覆盖区域的clip
    # 允许更多clip（更少切到核心），所以适当提高上限
    max_iterations = 5000  # 防止无限循环
    iteration = 0

    clip_w_px = max(1, int(round(clip_w_norm * width)))
    clip_h_px = max(1, int(round(clip_h_norm * height)))
    clip_area_px = max(1, int(clip_w_px * clip_h_px))
    border_thickness_px = int(round(min(clip_w_px, clip_h_px) * float(border_thickness_ratio)))
    border_thickness_px = max(1, min(border_thickness_px, 12))

    if orig_mask is None:
        orig_mask = mask
    total_orig_pixels = int(np.sum(orig_mask))
    total_orig_pixels = max(1, total_orig_pixels)
    
    while np.any(uncovered_mask) and iteration < max_iterations:
        iteration += 1
        best_box = None
        best_score = -1

        remaining_uncovered = int(np.sum(uncovered_mask))
        progress = float(remaining_uncovered) / float(total_orig_pixels)  # 1.0 -> 0.0
        relax_factor = min(1.0, progress / max(1e-6, float(progress_relax_ratio)))

        # 早期避免“只蹭一点点”的clip，后期自动放宽
        effective_min_uncovered = int(round(float(clip_area_px) * float(min_uncovered_ratio) * float(relax_factor)))
        # 早期更强调clip内mask“装满”，后期允许为了补洞而低填充
        effective_min_fill = float(min_mask_fill_ratio) * float(relax_factor)
        
        # 生成候选位置：
        # - 粗网格提供基础覆盖
        # - 围绕“未覆盖像素”采样的局部候选提供更细的对齐能力（减少把赛道切成上下两半）
        step_x = max(search_step, clip_w_norm * (1 - min_overlap_ratio) * 0.35)
        step_y = max(search_step, clip_h_norm * (1 - min_overlap_ratio) * 0.35)
        
        # 如果还有未覆盖区域，动态调整搜索范围
        if np.any(uncovered_mask):
            uncovered_pixels = np.where(uncovered_mask)
            if len(uncovered_pixels[0]) > 0:
                uncovered_min_y = uncovered_pixels[0].min()
                uncovered_max_y = uncovered_pixels[0].max() + 1
                uncovered_min_x = uncovered_pixels[1].min()
                uncovered_max_x = uncovered_pixels[1].max() + 1
                
                # 扩展搜索范围以包含未覆盖区域
                uncovered_min_x_norm = max(0.0, (uncovered_min_x - int(clip_w_norm * width * 0.5)) / width)
                uncovered_min_y_norm = max(0.0, (uncovered_min_y - int(clip_h_norm * height * 0.5)) / height)
                uncovered_max_x_norm = min(1.0 - clip_w_norm, (uncovered_max_x + int(clip_w_norm * width * 0.5)) / width)
                uncovered_max_y_norm = min(1.0 - clip_h_norm, (uncovered_max_y + int(clip_h_norm * height * 0.5)) / height)
                
                # 合并搜索范围
                search_min_x = min(search_min_x, uncovered_min_x_norm)
                search_min_y = min(search_min_y, uncovered_min_y_norm)
                search_max_x = max(search_max_x, uncovered_max_x_norm)
                search_max_y = max(search_max_y, uncovered_max_y_norm)
        
        # 生成候选位置（去重后的小集合）
        cand_set: set[Tuple[float, float]] = set()

        def _add_cand(xn: float, yn: float):
            xn = float(max(0.0, min(xn, 1.0 - clip_w_norm)))
            yn = float(max(0.0, min(yn, 1.0 - clip_h_norm)))
            cand_set.add((round(xn, 6), round(yn, 6)))

        # 1) 粗网格
        gx = np.linspace(search_min_x, search_max_x, min(18, int((search_max_x - search_min_x) / step_x) + 1))
        gy = np.linspace(search_min_y, search_max_y, min(18, int((search_max_y - search_min_y) / step_y) + 1))
        for yn in gy:
            for xn in gx:
                _add_cand(float(xn), float(yn))
        # 2) 未覆盖像素驱动的局部候选（中心对齐 + 小幅抖动）
        uncovered_coords = np.argwhere(uncovered_mask)
        if uncovered_coords.size > 0 and max_sample_points > 0 and max_candidates > 0:
            stride = max(1, int(uncovered_coords.shape[0] // max(1, max_sample_points)))
            sampled = uncovered_coords[::stride]
            # 小抖动：允许把未覆盖点放在clip中心附近不同位置，帮助“把边放到背景上”
            dxs = [-0.18 * clip_w_norm, 0.0, 0.18 * clip_w_norm]
            dys = [-0.18 * clip_h_norm, 0.0, 0.18 * clip_h_norm]
            for (yy, xx) in sampled:
                base_x = (float(xx) / float(width)) - 0.5 * clip_w_norm
                base_y = (float(yy) / float(height)) - 0.5 * clip_h_norm
                for dx in dxs:
                    for dy in dys:
                        _add_cand(base_x + dx, base_y + dy)
                        if len(cand_set) >= max_candidates:
                            break
                    if len(cand_set) >= max_candidates:
                        break
                if len(cand_set) >= max_candidates:
                    break

        candidate_positions = list(cand_set)
        
        for (x_norm, y_norm) in candidate_positions:
                if x_norm + clip_w_norm > 1.0 or y_norm + clip_h_norm > 1.0:
                    continue
                
                box = (x_norm, y_norm, x_norm + clip_w_norm, y_norm + clip_h_norm)
                
                # 计算覆盖得分（只考虑未覆盖区域）
                coverage = _get_coverage_score(box, uncovered_mask, width, height)
                
                if coverage <= 0:
                    continue  # 这个位置不覆盖任何未覆盖区域

                if effective_min_uncovered > 0 and remaining_uncovered > effective_min_uncovered:
                    if coverage < effective_min_uncovered:
                        # 覆盖还很多时，过滤掉“只蹭一点点”的候选
                        continue

                # clip内mask总量（用原始mask算，更符合“路面像素装满”目标）
                mask_pixels = _get_coverage_score(box, orig_mask, width, height)
                fill_ratio = float(mask_pixels) / float(clip_area_px)
                context_ratio = max(0.0, 1.0 - float(fill_ratio))

                # 上下文留白：不希望clip几乎全是mask（缺少环境信息）
                if float(context_ratio) < float(min_context_ratio) * float(relax_factor):
                    # 覆盖还很多时更严格，后期放宽
                    continue

                # 计算“边界切到mask核心”的惩罚项：鼓励边缘尽量落在背景/或mask边缘附近
                cut_penalty = 0.0
                if orig_mask is not None and cut_weight > 0:
                    cut_penalty = _border_cut_cost(
                        box=box,
                        mask=orig_mask,
                        width=width,
                        height=height,
                        border_thickness_px=border_thickness_px,
                        dist_in_mask=dist_in_mask,
                        edge_total_weight=1.0,
                        edge_concentration_weight=5.0,
                        concentration_power=2.0,
                        core_weight=0.05
                    )

                # 额外：坏切法惩罚（对边同时贴mask -> 更像把路面切成两半）
                (
                    top_cnt, bottom_cnt, left_cnt, right_cnt,
                    top_size, bottom_size, left_size, right_size,
                    x_span, y_span, t, _
                ) = _edge_contact_stats(box, orig_mask, width, height, border_thickness_px)
                eps = 1e-9
                top_r = float(top_cnt) / (float(top_size) + eps)
                bottom_r = float(bottom_cnt) / (float(bottom_size) + eps)
                left_r = float(left_cnt) / (float(left_size) + eps)
                right_r = float(right_cnt) / (float(right_size) + eps)

                # 强约束：避免出现“某一条边几乎整条都贴在mask里”（典型不完整路面/切半）
                # 早期严格，后期放宽，但不放宽到 1.0（仍尽量避免整条边完全贴mask）
                edge_thr = float(max_edge_contact_ratio) + (float(max_edge_contact_ratio_late) - float(max_edge_contact_ratio)) * (1.0 - float(relax_factor))
                if max(top_r, bottom_r, left_r, right_r) > edge_thr:
                    continue
                # 以“对边比例乘积 * 对应边长度 * strip厚度”构成像素级代价
                bad_tb = float(top_r * bottom_r) * float(max(1, x_span)) * float(max(1, t))
                bad_lr = float(left_r * right_r) * float(max(1, y_span)) * float(max(1, t))
                bad_split = float(bad_tb + bad_lr)

                # 方案B：加权集合覆盖视角 —— 用 “新增覆盖 / (1 + 代价)” 排序，显式偏向更少clip
                penalty_px = float(cut_weight) * float(cut_penalty) + float(bad_split_weight) * float(bad_split)
                # 用边界strip总像素数归一化成“边界贴mask比例”的量级，再用 penalty_scale 放大进入 cost
                penalty_norm = penalty_px / float(max(1, top_size + bottom_size + left_size + right_size))
                cost = 1.0 + float(penalty_scale) * max(0.0, float(penalty_norm))
                score = float(coverage) / float(cost) + float(density_weight) * float(mask_pixels) / float(clip_area_px)

                if effective_min_fill > 0:
                    short = max(0.0, float(effective_min_fill) - float(fill_ratio))
                    if short > 0:
                        # 低填充惩罚也转为“对score的扣减”，保持尺度稳定
                        score -= float(fill_penalty_weight) * short
                
                # 检查是否与已选择的boxes满足重叠要求（如果相邻）
                valid = True
                if len(selected_boxes) > 0:
                    # 检查是否与所有已选择的boxes都满足重叠要求（如果相邻）
                    for selected_box in selected_boxes:
                        if not _check_overlap_requirement(box, selected_box, clip_size, min_overlap_ratio):
                            valid = False
                            break
                
                # 如果当前候选不满足重叠要求，但覆盖的未覆盖区域很多，可以放宽条件
                if not valid and coverage > 0:
                    # 检查是否覆盖了足够多的未覆盖区域（至少覆盖clip面积的10%）
                    min_required_coverage = clip_w_norm * clip_h_norm * width * height * 0.1
                    if coverage >= min_required_coverage:
                        # 允许这个候选，即使不满足重叠要求
                        valid = True
                
                if not valid:
                    continue
                
                if score > best_score:
                    best_score = score
                    best_box = box
        
        # 如果找不到满足条件的box，但还有未覆盖区域，尝试放宽条件
        if best_box is None or best_score <= 0:
            if np.any(uncovered_mask):
                # 最后一次尝试：选择能覆盖最多未覆盖区域的box，不考虑重叠要求
                best_box = None
                best_score = -1
                for (x_norm, y_norm) in candidate_positions:
                    if x_norm + clip_w_norm > 1.0 or y_norm + clip_h_norm > 1.0:
                        continue
                    box = (x_norm, y_norm, x_norm + clip_w_norm, y_norm + clip_h_norm)

                    # 即便在fallback阶段，也尽量不选“明显坏切法”的框
                    if orig_mask is not None:
                        mask_pixels = _get_coverage_score(box, orig_mask, width, height)
                        fill_ratio = float(mask_pixels) / float(clip_area_px)
                        context_ratio = max(0.0, 1.0 - float(fill_ratio))
                        if float(context_ratio) < float(min_context_ratio) * float(relax_factor):
                            continue

                        (
                            top_cnt, bottom_cnt, left_cnt, right_cnt,
                            top_size, bottom_size, left_size, right_size,
                            _, _, _, _
                        ) = _edge_contact_stats(box, orig_mask, width, height, border_thickness_px)
                        eps = 1e-9
                        top_r = float(top_cnt) / (float(top_size) + eps)
                        bottom_r = float(bottom_cnt) / (float(bottom_size) + eps)
                        left_r = float(left_cnt) / (float(left_size) + eps)
                        right_r = float(right_cnt) / (float(right_size) + eps)
                        edge_thr = float(max_edge_contact_ratio) + (float(max_edge_contact_ratio_late) - float(max_edge_contact_ratio)) * (1.0 - float(relax_factor))
                        if max(top_r, bottom_r, left_r, right_r) > edge_thr:
                            continue

                    coverage = _get_coverage_score(box, uncovered_mask, width, height)
                    if coverage > best_score:
                        best_score = coverage
                        best_box = box
                
                if best_box is None or best_score <= 0:
                    break  # 真的找不到任何能覆盖未覆盖区域的box了
            else:
                break  # 所有区域都已覆盖
        
        # 添加最佳box
        selected_boxes.append(best_box)
        
        # 更新未覆盖区域
        x_min_px = int(best_box[0] * width)
        y_min_px = int(best_box[1] * height)
        x_max_px = int(best_box[2] * width)
        y_max_px = int(best_box[3] * height)
        
        x_min_px = max(0, min(x_min_px, width))
        y_min_px = max(0, min(y_min_px, height))
        x_max_px = max(0, min(x_max_px, width))
        y_max_px = max(0, min(y_max_px, height))
        
        if x_max_px > x_min_px and y_max_px > y_min_px:
            # 将clip区域内的未覆盖像素标记为已覆盖
            uncovered_mask[y_min_px:y_max_px, x_min_px:x_max_px] = False
    
    return selected_boxes


def _verify_coverage(
    mask: np.ndarray,
    boxes: List[Tuple[float, float, float, float]],
    width: int,
    height: int
) -> Tuple[bool, float]:
    """
    验证boxes是否完全覆盖了mask。
    
    Args:
        mask: 要验证的mask（可以是整个mask或连通区域mask）
        boxes: 归一化的box列表
        width, height: 图像尺寸
    
    Returns:
        (是否完全覆盖, 覆盖率)
    """
    if len(boxes) == 0:
        return False, 0.0
    
    # 创建覆盖mask
    covered_mask = np.zeros_like(mask, dtype=bool)
    
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        x_min_px = max(0, int(x_min * width))
        y_min_px = max(0, int(y_min * height))
        x_max_px = min(width, int(x_max * width))
        y_max_px = min(height, int(y_max * height))
        
        if x_max_px > x_min_px and y_max_px > y_min_px:
            # 标记clip区域内的像素为已覆盖
            covered_mask[y_min_px:y_max_px, x_min_px:x_max_px] = True
    
    # 计算覆盖率（只考虑mask中的有效像素）
    total_pixels = np.sum(mask)
    covered_pixels = np.sum(covered_mask & mask)
    coverage_ratio = covered_pixels / total_pixels if total_pixels > 0 else 0.0
    
    return coverage_ratio >= 0.9999, coverage_ratio  # 允许0.01%的误差


def generate_clip_boxes(
    mask: Union[np.ndarray, Image.Image],
    clip_size: Tuple[float, float],
    overlap_ratio: float
) -> List[Tuple[float, float, float, float]]:
    """
    生成一系列归一化的裁剪框，用于在mask图像上进行滑动窗口裁剪。
    
    Args:
        mask: mask图像，可以是numpy数组或PIL Image，有效值为非零
        clip_size: 归一化的裁剪窗口尺寸 (width, height)，范围在0-1之间
        overlap_ratio: 最小重叠比例（相对于clip尺寸），范围在0-1之间。
                      表示相邻clip之间必须满足的最小重叠区域（相对于clip尺寸的比例）
    
    Returns:
        归一化的box列表，每个box格式为 (x_min, y_min, x_max, y_max)，值在0-1之间
    
    算法说明：
    0. 先对mask进行膨胀，膨胀的尺寸为overlap_ratio所对应的尺寸
    1. 使用全局贪心算法，在整个mask上优化clip位置，最小化全局clip数量
    2. 允许一个clip覆盖多个连通区域，实现全局最优解
    3. clip位置不要求行列对齐，可以灵活移动
    4. 相邻clip之间必须满足最小重叠要求（overlap_ratio）
    5. 额外优化目标：clip边界尽可能少“切到mask核心”（尤其是赛道路面这种大联通区域）。
    """
    # 将mask转换为numpy数组
    if isinstance(mask, Image.Image):
        mask_array = np.array(mask)
        if len(mask_array.shape) == 3:
            # 如果是RGB图像，转换为灰度图
            mask_array = np.any(mask_array > 0, axis=2)
        else:
            mask_array = mask_array > 0
    else:
        mask_array = mask
        if len(mask_array.shape) == 3:
            mask_array = np.any(mask_array > 0, axis=2)
        else:
            mask_array = mask_array > 0
    
    height, width = mask_array.shape
    orig_mask_array = mask_array.copy()

    # 用原始mask做距离变换：可选的轻微“切到核心”惩罚项
    # 缺少scipy时自动降级为None（仍可用边界贴mask长度/坏切法惩罚来优化）
    dist_in_mask = None
    if ndimage is not None:
        dist_in_mask = cast(np.ndarray, ndimage.distance_transform_edt(orig_mask_array.astype(np.uint8)))  # type: ignore
    
    # 在进行clip之前，先对mask进行膨胀（让覆盖更稳，同时为overlap留出缓冲）
    # 膨胀的尺寸为overlap_ratio所对应的尺寸
    dilation_w_px = int(overlap_ratio * clip_size[0] * width)
    dilation_h_px = int(overlap_ratio * clip_size[1] * height)
    
    # 创建结构元素用于膨胀（使用矩形结构元素）
    if dilation_w_px > 0 or dilation_h_px > 0:
        if ndimage is not None:
            # 创建结构元素，尺寸为 (2*dilation_h_px+1, 2*dilation_w_px+1)
            structure_size = (max(1, 2 * dilation_h_px + 1), max(1, 2 * dilation_w_px + 1))
            structure = np.ones(structure_size, dtype=bool)
            mask_array = ndimage.binary_dilation(mask_array, structure=structure)  # type: ignore
        else:
            mask_array = _binary_dilation_rect_numpy(mask_array, dilation_h_px, dilation_w_px)
    
    # 找到有效区域的边界框
    valid_pixels = np.where(mask_array)
    if len(valid_pixels[0]) == 0:
        # 如果没有有效像素，返回空列表
        return []
    
    # 使用全局贪心算法，在整个mask上优化clip数量
    # 这样可以允许一个clip覆盖多个连通区域，实现全局最优
    all_boxes = _greedy_cover_global(
        mask_array,
        clip_size,
        overlap_ratio,
        width,
        height,
        search_step=min(clip_size[0], clip_size[1]) * 0.01,  # 更细的搜索步长，给“避开核心边界”更多自由度
        dist_in_mask=dist_in_mask,
        cut_weight=1.0,
        border_thickness_ratio=0.02,
        orig_mask=orig_mask_array,
        density_weight=0.15,
        min_uncovered_ratio=0.03,
        min_mask_fill_ratio=0.08,
        fill_penalty_weight=1.0,
        progress_relax_ratio=0.12,
        bad_split_weight=2.0,
        min_context_ratio=0.12,
        penalty_scale=8.0,
        max_edge_contact_ratio=0.75,
        max_edge_contact_ratio_late=0.9
    )
    
    # 验证覆盖是否完整
    is_covered, coverage_ratio = _verify_coverage(mask_array, all_boxes, width, height)
    if not is_covered:
        # 如果覆盖率不够，尝试补充更多的clip
        # 找到未覆盖的区域
        covered_mask = np.zeros_like(mask_array, dtype=bool)
        for box in all_boxes:
            x_min, y_min, x_max, y_max = box
            x_min_px = max(0, int(x_min * width))
            y_min_px = max(0, int(y_min * height))
            x_max_px = min(width, int(x_max * width))
            y_max_px = min(height, int(y_max * height))
            if x_max_px > x_min_px and y_max_px > y_min_px:
                covered_mask[y_min_px:y_max_px, x_min_px:x_max_px] = True
        
        uncovered_mask = np.logical_and(mask_array.astype(bool), ~covered_mask)
        
        # 对未覆盖区域再次尝试覆盖（放宽重叠要求）
        if np.any(uncovered_mask):
            additional_boxes = _greedy_cover_global(
                uncovered_mask,
                clip_size,
                overlap_ratio * 0.5,  # 放宽重叠要求
                width,
                height,
                search_step=min(clip_size[0], clip_size[1]) * 0.01,  # 更细的搜索步长
                dist_in_mask=dist_in_mask,
                cut_weight=1.0,
                border_thickness_ratio=0.02,
                orig_mask=orig_mask_array,
                density_weight=0.15,
                min_uncovered_ratio=0.03,
                min_mask_fill_ratio=0.08,
                fill_penalty_weight=1.0,
                progress_relax_ratio=0.12,
                bad_split_weight=2.0,
                min_context_ratio=0.12,
                penalty_scale=8.0,
                max_edge_contact_ratio=0.75,
                max_edge_contact_ratio_late=0.9
            )
            all_boxes.extend(additional_boxes)
    
    # 去重：移除完全相同的boxes
    if len(all_boxes) == 0:
        return []
    
    # 使用集合去重（转换为元组以便使用set）
    unique_boxes = []
    seen_boxes = set()
    
    for box in all_boxes:
        # 将box四舍五入到小数点后6位，避免浮点误差导致的重复
        rounded_box = tuple(round(coord, 6) for coord in box)
        if rounded_box not in seen_boxes:
            seen_boxes.add(rounded_box)
            unique_boxes.append(box)
    
    return unique_boxes


def visualize_clip_boxes(
    mask: Union[np.ndarray, Image.Image],
    boxes: List[Tuple[float, float, float, float]],
    save_path: Union[str, None] = None,
    show_plot: bool = True
):
    """
    可视化mask图像和生成的裁剪框。
    
    Args:
        mask: mask图像，可以是numpy数组或PIL Image
        boxes: 归一化的box列表，格式为 (x_min, y_min, x_max, y_max)
        save_path: 保存图像的路径，如果为None则不保存
        show_plot: 是否显示图像
    """
    if plt is None or patches is None:
        raise ModuleNotFoundError(
            "未安装matplotlib，无法可视化。请先安装：pip install matplotlib"
        )
    # 将mask转换为numpy数组用于显示
    if isinstance(mask, Image.Image):
        mask_array = np.array(mask)
        if len(mask_array.shape) == 3:
            # 如果是RGB图像，转换为灰度图
            display_mask = np.any(mask_array > 0, axis=2)
        else:
            display_mask = mask_array > 0
        height, width = mask.size[1], mask.size[0]
    else:
        mask_array = mask
        if len(mask_array.shape) == 3:
            display_mask = np.any(mask_array > 0, axis=2)
        else:
            display_mask = mask_array > 0
        height, width = mask_array.shape
    
    # 创建图像显示
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # 显示mask（使用半透明红色）
    ax.imshow(display_mask, cmap='Reds', alpha=0.5, origin='upper')
    
    # 绘制每个裁剪框
    cmap = plt.get_cmap('tab20')
    colors = cmap(np.linspace(0, 1, len(boxes)))
    for idx, (x_min, y_min, x_max, y_max) in enumerate(boxes):
        # 将归一化坐标转换为像素坐标
        x_min_px = x_min * width
        y_min_px = y_min * height
        x_max_px = x_max * width
        y_max_px = y_max * height
        
        # 计算矩形的位置和尺寸
        rect_width = x_max_px - x_min_px
        rect_height = y_max_px - y_min_px
        
        # 创建矩形框
        rect = patches.Rectangle(
            (x_min_px, y_min_px),
            rect_width,
            rect_height,
            linewidth=2,
            edgecolor=colors[idx],
            facecolor='none',
            label=f'Clip {idx + 1}'
        )
        ax.add_patch(rect)
        
        # 添加编号标签
        ax.text(
            x_min_px + rect_width / 2,
            y_min_px + rect_height / 2,
            str(idx + 1),
            ha='center',
            va='center',
            fontsize=10,
            color='white',
            weight='bold',
            bbox=dict(boxstyle='round', facecolor=colors[idx], alpha=0.7)
        )
    
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # Note: image coordinate system y-axis points downward
    ax.set_xlabel('X (pixels)', fontsize=12)
    ax.set_ylabel('Y (pixels)', fontsize=12)
    ax.set_title(f'Clip Boxes Visualization ({len(boxes)} clips)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add legend if clip count is not too large
    if len(boxes) <= 20:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    
    # Save image
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    # 显示图像
    if show_plot:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    mask = Image.open("E:\\sam3_track_seg\\test_images\\result_mask0_prob(0.29).png")


    clip_size = (0.04835064467727697, 0.03690212900375415)
    overlap_ratio = 0.1
    boxes = generate_clip_boxes(mask, clip_size, overlap_ratio)
    
    print(f"Generated {len(boxes)} clip boxes:")
    for idx, box in enumerate(boxes):
        print(f"  Clip {idx + 1}: ({box[0]:.4f}, {box[1]:.4f}, {box[2]:.4f}, {box[3]:.4f})")
    
    # 可视化结果
    output_dir = "output"
    import os
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "clip_boxes_visualization.png")
    visualize_clip_boxes(mask, boxes, save_path=save_path, show_plot=True)