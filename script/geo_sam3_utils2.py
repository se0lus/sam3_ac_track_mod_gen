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


def compute_dilated_mask_for_clips(
    mask: Union[np.ndarray, Image.Image],
    clip_size: Tuple[float, float],
    overlap_ratio: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算“用于clip覆盖目标”的膨胀mask（与 generate_clip_boxes2 使用同一逻辑）。

    Returns:
        (orig_mask_bool, dilated_mask_bool)
    """
    # 转 bool mask
    if isinstance(mask, Image.Image):
        mask_array = np.array(mask)
        if len(mask_array.shape) == 3:
            orig = np.any(mask_array > 0, axis=2)
        else:
            orig = mask_array > 0
    else:
        mask_array = mask
        if len(mask_array.shape) == 3:
            orig = np.any(mask_array > 0, axis=2)
        else:
            orig = mask_array > 0

    dilated = orig.astype(bool).copy()
    height, width = dilated.shape

    dilation_w_px = int(overlap_ratio * clip_size[0] * width)
    dilation_h_px = int(overlap_ratio * clip_size[1] * height)

    if dilation_w_px > 0 or dilation_h_px > 0:
        if ndimage is not None:
            structure_size = (max(1, 2 * dilation_h_px + 1), max(1, 2 * dilation_w_px + 1))
            structure = np.ones(structure_size, dtype=bool)
            dilated = ndimage.binary_dilation(dilated, structure=structure)  # type: ignore
        else:
            dilated = _binary_dilation_rect_numpy(dilated, dilation_h_px, dilation_w_px)

    return orig.astype(bool), dilated.astype(bool)


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


def _overlap_stats_px(
    box1: Tuple[float, float, float, float],
    box2: Tuple[float, float, float, float],
    width: int,
    height: int
) -> Tuple[int, int, int]:
    """
    返回两个box的像素级重叠统计：(overlap_w_px, overlap_h_px, overlap_area_px)。
    使用与覆盖/更新一致的 int 截断方式，避免“理论重叠够但像素上不够/反之”的漂移。
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    a_x0 = max(0, min(width, int(x1_min * width)))
    a_y0 = max(0, min(height, int(y1_min * height)))
    a_x1 = max(0, min(width, int(x1_max * width)))
    a_y1 = max(0, min(height, int(y1_max * height)))

    b_x0 = max(0, min(width, int(x2_min * width)))
    b_y0 = max(0, min(height, int(y2_min * height)))
    b_x1 = max(0, min(width, int(x2_max * width)))
    b_y1 = max(0, min(height, int(y2_max * height)))

    ow = max(0, min(a_x1, b_x1) - max(a_x0, b_x0))
    oh = max(0, min(a_y1, b_y1) - max(a_y0, b_y0))
    return int(ow), int(oh), int(ow * oh)


def _check_overlap_requirement_v2(
    box1: Tuple[float, float, float, float],
    box2: Tuple[float, float, float, float],
    clip_area_px: int,
    min_overlap_ratio: float,
    width: int,
    height: int
) -> bool:
    """
    判断两个clip是否满足最小重叠要求（像素级、按“重叠面积比例”约束）。

    经验上这更贴合“直线段滑动窗口”的直觉：
    - 沿着赛道方向滑动时，一个维度可能接近100%重叠，另一个维度只需要 r 的步进
      => 重叠面积比例接近 r（而不是 r^2）。
    因此这里用 overlap_area / clip_area >= r。
    """
    if min_overlap_ratio <= 0:
        return True
    ow, oh, _ = _overlap_stats_px(box1, box2, width=width, height=height)
    if ow <= 0 or oh <= 0:
        return False
    _, _, oa = _overlap_stats_px(box1, box2, width=width, height=height)
    return float(oa) / float(max(1, int(clip_area_px))) >= float(min_overlap_ratio)


def _get_coverage_score(
    box: Tuple[float, float, float, float],
    uncovered_mask: np.ndarray,
    width: int,
    height: int
) -> float:
    """返回 box 内 True 像素数（用于覆盖/填充评分）。"""
    x_min, y_min, x_max, y_max = box
    x_min_px = max(0, int(x_min * width))
    y_min_px = max(0, int(y_min * height))
    x_max_px = min(width, int(x_max * width))
    y_max_px = min(height, int(y_max * height))
    if x_max_px <= x_min_px or y_max_px <= y_min_px:
        return 0.0
    return float(np.sum(uncovered_mask[y_min_px:y_max_px, x_min_px:x_max_px]))


def _count_true_segments_circular(arr: np.ndarray) -> int:
    """
    统计一维布尔数组中 True 的连续段数量（环形：首尾相接）。
    """
    if arr.size == 0:
        return 0
    a = arr.astype(bool)
    if not np.any(a):
        return 0
    # 线性段数
    seg = int(np.sum((~a[:-1]) & a[1:]))
    seg += 1 if a[0] else 0
    # 环形合并：首尾都为True时，首段与尾段应合并，段数-1
    if a[0] and a[-1] and seg > 1:
        seg -= 1
    return seg


def _perimeter_evenodd_score(
    box: Tuple[float, float, float, float],
    mask: np.ndarray,
    width: int,
    height: int,
    border_thickness_px: int = 1
) -> Tuple[float, int, int]:
    """
    根据“边框连续段偶/奇”给出评分（更偏惩罚项）。

    返回：(score, segment_count, border_true_count)
    - 直觉目标：优先让 segment_count 为偶数（更像“赛道从clip中间穿过并在边界留下两段出口/入口”）
      但在满足偶数段的前提下，希望边框上命中的mask像素越少越好（避免边界沿着路面中心切开很长一段）。
    - 因此这里返回“越大越好”的分数：
        - 偶数段：score = -border_true_count（越接近0越好）
        - 奇数段：score = -odd_penalty_factor*border_true_count（更强惩罚）
      （segment_count==0 或 border_true_count==0 时 score 为 0）
    """
    x_min, y_min, x_max, y_max = box
    x0 = max(0, min(width - 1, int(x_min * width)))
    y0 = max(0, min(height - 1, int(y_min * height)))
    x1 = max(0, min(width, int(x_max * width)))
    y1 = max(0, min(height, int(y_max * height)))
    if x1 <= x0 or y1 <= y0:
        return 0.0, 0, 0

    t = max(1, int(border_thickness_px))
    # clip边框strip（把厚度方向用 any 压成一维）
    top = mask[y0:min(height, y0 + t), x0:x1]
    bottom = mask[max(0, y1 - t):y1, x0:x1]
    left = mask[y0:y1, x0:min(width, x0 + t)]
    right = mask[y0:y1, max(0, x1 - t):x1]

    top_1d = np.any(top, axis=0) if top.size else np.zeros((x1 - x0,), dtype=bool)
    right_1d = np.any(right, axis=1) if right.size else np.zeros((y1 - y0,), dtype=bool)
    bottom_1d = np.any(bottom, axis=0) if bottom.size else np.zeros((x1 - x0,), dtype=bool)
    left_1d = np.any(left, axis=1) if left.size else np.zeros((y1 - y0,), dtype=bool)

    # 顺时针拼接：top -> right -> bottom(reverse) -> left(reverse)
    peri = np.concatenate([top_1d, right_1d, bottom_1d[::-1], left_1d[::-1]], axis=0)
    border_true = int(np.sum(peri))
    seg = _count_true_segments_circular(peri)
    if seg <= 0 or border_true <= 0:
        return 0.0, seg, border_true
    odd_penalty_factor = 2.5
    if seg % 2 == 0:
        score = -float(border_true)
    else:
        score = -float(odd_penalty_factor) * float(border_true)
    return score, seg, border_true


def _border_core_cut_penalty(
    box: Tuple[float, float, float, float],
    orig_mask: np.ndarray,
    dist_in_mask: Union[np.ndarray, None],
    width: int,
    height: int,
    border_thickness_px: int = 1
) -> float:
    """
    轻量版“切到mask核心”惩罚：统计边框strip内落在 orig_mask 上的 dist 值。
    dist 越大代表越靠近mask内部核心，惩罚越大。
    """
    if dist_in_mask is None:
        return 0.0
    x_min, y_min, x_max, y_max = box
    x0 = max(0, min(width - 1, int(x_min * width)))
    y0 = max(0, min(height - 1, int(y_min * height)))
    x1 = max(0, min(width, int(x_max * width)))
    y1 = max(0, min(height, int(y_max * height)))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    t = max(1, int(border_thickness_px))
    # 取四条边 strip 的并集（避免重复统计角点，按经验影响很小）
    top_y1 = min(height, y0 + t)
    bottom_y0 = max(0, y1 - t)
    left_x1 = min(width, x0 + t)
    right_x0 = max(0, x1 - t)

    penalty = 0.0
    # top & bottom
    if top_y1 > y0:
        m = orig_mask[y0:top_y1, x0:x1]
        if np.any(m):
            penalty += float(np.sum(dist_in_mask[y0:top_y1, x0:x1][m]))
    if y1 > bottom_y0:
        m = orig_mask[bottom_y0:y1, x0:x1]
        if np.any(m):
            penalty += float(np.sum(dist_in_mask[bottom_y0:y1, x0:x1][m]))
    # left & right
    if left_x1 > x0:
        m = orig_mask[y0:y1, x0:left_x1]
        if np.any(m):
            penalty += float(np.sum(dist_in_mask[y0:y1, x0:left_x1][m]))
    if x1 > right_x0:
        m = orig_mask[y0:y1, right_x0:x1]
        if np.any(m):
            penalty += float(np.sum(dist_in_mask[y0:y1, right_x0:x1][m]))
    return penalty


def _greedy_cover_global_v2(
    mask: np.ndarray,
    clip_size: Tuple[float, float],
    min_overlap_ratio: float,
    width: int,
    height: int,
    search_step: float,
    orig_mask: np.ndarray,
    dist_in_mask: Union[np.ndarray, None] = None,
    evenodd_weight: float = 1.0,
    core_cut_weight: float = 0.25,
    density_weight: float = 0.10,
    max_iterations: int = 5000,
    max_sample_points: int = 240,
    max_candidates: int = 3000,
    border_thickness_ratio: float = 0.02,
    # 新增：过度重叠惩罚（越大越倾向用“接近最小重叠”的步进）
    overlap_excess_weight: float = 0.35,
    # 新增：过度重叠的软上限（按 overlap_area/clip_area），超过后直接过滤
    max_overlap_area_ratio: float = 0.85,
    # 新增：当严格 overlap 找不到候选时的兜底（避免漏覆盖）
    allow_overlap_fallback: bool = True,
) -> List[Tuple[float, float, float, float]]:
    """
    全局覆盖（贪心近似），核心目标：
    - 覆盖膨胀后的mask（mask 参数）
    - 约束相邻clip具备最小重叠：候选必须与“至少一个已选clip”满足 overlap
    - 评分同时考虑：新增覆盖、clip内mask密度、边框偶/奇段奖励、切到核心惩罚
    """
    clip_w_norm, clip_h_norm = clip_size
    clip_w_px = max(1, int(round(clip_w_norm * width)))
    clip_h_px = max(1, int(round(clip_h_norm * height)))
    clip_area_px = max(1, int(clip_w_px * clip_h_px))
    border_thickness_px = int(round(min(clip_w_px, clip_h_px) * float(border_thickness_ratio)))
    border_thickness_px = max(1, min(border_thickness_px, 12))

    valid_pixels = np.where(mask)
    if len(valid_pixels[0]) == 0:
        return []

    min_y, max_y = valid_pixels[0].min(), valid_pixels[0].max() + 1
    min_x, max_x = valid_pixels[1].min(), valid_pixels[1].max() + 1
    min_x_norm = max(0.0, min_x / width)
    min_y_norm = max(0.0, min_y / height)
    max_x_norm = min(1.0, max_x / width)
    max_y_norm = min(1.0, max_y / height)

    # 搜索范围略扩（给overlap留空间）
    margin_w_norm = min_overlap_ratio * clip_w_norm
    margin_h_norm = min_overlap_ratio * clip_h_norm
    search_min_x = max(0.0, min_x_norm - margin_w_norm)
    search_min_y = max(0.0, min_y_norm - margin_h_norm)
    search_max_x = min(1.0 - clip_w_norm, max_x_norm + margin_w_norm)
    search_max_y = min(1.0 - clip_h_norm, max_y_norm + margin_h_norm)

    if search_max_x <= search_min_x or search_max_y <= search_min_y:
        return [(search_min_x, search_min_y, search_min_x + clip_w_norm, search_min_y + clip_h_norm)]

    uncovered_mask = mask.copy()
    selected_boxes: List[Tuple[float, float, float, float]] = []

    step_x = max(float(search_step), float(clip_w_norm) * 0.01)
    step_y = max(float(search_step), float(clip_h_norm) * 0.01)

    # 主循环
    for _ in range(int(max_iterations)):
        if not np.any(uncovered_mask):
            break

        # 候选点集合（top-left）
        cand_set: set[Tuple[float, float]] = set()

        def _add_cand(xn: float, yn: float):
            xn = float(max(0.0, min(xn, 1.0 - clip_w_norm)))
            yn = float(max(0.0, min(yn, 1.0 - clip_h_norm)))
            cand_set.add((round(xn, 6), round(yn, 6)))

        # 1) 粗网格（覆盖稳）
        nx = max(2, min(22, int((search_max_x - search_min_x) / step_x) + 2))
        ny = max(2, min(22, int((search_max_y - search_min_y) / step_y) + 2))
        gx = np.linspace(search_min_x, search_max_x, nx)
        gy = np.linspace(search_min_y, search_max_y, ny)
        for yn in gy:
            for xn in gx:
                _add_cand(float(xn), float(yn))

        # 2) 未覆盖像素驱动（更贴合赛道）
        uncovered_coords = np.argwhere(uncovered_mask)
        if uncovered_coords.size > 0 and max_sample_points > 0 and max_candidates > 0:
            stride = max(1, int(uncovered_coords.shape[0] // max(1, max_sample_points)))
            sampled = uncovered_coords[::stride]
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

        # 3) 基于已选box的“理想步进”候选（解决：直线区域重叠不足/或重叠过大）
        #    理想步长为 (1-r)*clip；这会把重叠“推向接近 r”，同时保证连贯性。
        if len(selected_boxes) > 0 and min_overlap_ratio > 0:
            stride_x = float(max(0.0, (1.0 - float(min_overlap_ratio)) * float(clip_w_norm)))
            stride_y = float(max(0.0, (1.0 - float(min_overlap_ratio)) * float(clip_h_norm)))
            # 只用最近几个box生成候选，避免爆炸
            for (sx0, sy0, _, _) in selected_boxes[-8:]:
                for dx in [0.0, stride_x, -stride_x, 0.5 * stride_x, -0.5 * stride_x]:
                    for dy in [0.0, stride_y, -stride_y, 0.5 * stride_y, -0.5 * stride_y]:
                        _add_cand(float(sx0 + dx), float(sy0 + dy))

        candidate_positions = list(cand_set)
        best_box: Union[Tuple[float, float, float, float], None] = None
        best_score = -1e18

        # 选一个能带来最大“综合收益”的clip
        for (x_norm, y_norm) in candidate_positions:
            box = (float(x_norm), float(y_norm), float(x_norm + clip_w_norm), float(y_norm + clip_h_norm))
            if box[2] > 1.0 or box[3] > 1.0:
                continue

            # 覆盖（只算未覆盖）
            new_cov = _get_coverage_score(box, uncovered_mask, width, height)
            if new_cov <= 0:
                continue

            # overlap：除第一个clip外，要求与至少一个已选clip满足重叠
            overlap_neighbor: Union[Tuple[float, float, float, float], None] = None
            overlap_ok = True
            chosen_overlap_area_ratio = 0.0
            if len(selected_boxes) > 0 and min_overlap_ratio > 0:
                overlap_ok = False
                # 选择“刚好满足下限且过量最小”的邻居，减少过度重叠
                best_excess = 1e18
                for sb in selected_boxes[-24:]:  # 只看最近一段，避免O(N^2)变慢
                    if _check_overlap_requirement_v2(
                        box, sb,
                        clip_area_px=clip_area_px,
                        min_overlap_ratio=min_overlap_ratio,
                        width=width,
                        height=height
                    ):
                        overlap_ok = True
                        _, _, oa = _overlap_stats_px(box, sb, width=width, height=height)
                        oa_r = float(oa) / float(max(1, clip_area_px))
                        # 过滤极端过度重叠（通常对应“几乎重复切同一块”）
                        if float(oa_r) > float(max_overlap_area_ratio):
                            continue
                        excess = float(max(0.0, float(oa_r) - float(min_overlap_ratio)))
                        if excess < best_excess:
                            best_excess = excess
                            overlap_neighbor = sb
                            chosen_overlap_area_ratio = float(oa_r)
                if not overlap_ok:
                    # 先跳过（严格阶段），后面若找不到任何严格候选再兜底
                    continue

            # clip内mask密度（用原始mask）
            mask_pixels = _get_coverage_score(box, orig_mask, width, height)
            fill_ratio = float(mask_pixels) / float(clip_area_px)

            # 边框偶/奇段奖励/惩罚（用原始mask，避免膨胀影响边界形态）
            evenodd_score, seg_cnt, border_true = _perimeter_evenodd_score(
                box=box,
                mask=orig_mask,
                width=width,
                height=height,
                border_thickness_px=1
            )

            # 切到核心惩罚（dist越大惩罚越大）
            core_penalty = _border_core_cut_penalty(
                box=box,
                orig_mask=orig_mask,
                dist_in_mask=dist_in_mask,
                width=width,
                height=height,
                border_thickness_px=border_thickness_px
            )

            # 评分改为“集合覆盖风格”：新增覆盖 / (1 + 代价)
            # 目的：减少 clip 数（优先选覆盖最大的），同时用代价项轻微抑制过度重叠/切到核心。
            overlap_excess = 0.0
            if overlap_neighbor is not None and min_overlap_ratio > 0:
                overlap_excess = float(max(0.0, float(chosen_overlap_area_ratio) - float(min_overlap_ratio)))
            # core_penalty 量纲很大，这里做一个温和归一化
            core_pen_norm = float(core_penalty) / float(max(1, border_thickness_px * (clip_w_px + clip_h_px)))
            cost = 1.0 + float(overlap_excess_weight) * float(overlap_excess) + float(core_cut_weight) * float(core_pen_norm)
            score = float(new_cov) / float(cost)
            # 密度与偶/奇段作为次级加成（不应压过“覆盖最大化”）
            score += float(density_weight) * float(mask_pixels) / float(max(1, clip_area_px))
            score += 0.02 * float(evenodd_weight) * float(evenodd_score) / float(max(1, clip_area_px))

            # 注意：偶/奇段逻辑已在 _perimeter_evenodd_score 内部体现为“惩罚项”
            # 这里不再额外给偶数段加分，避免与“偶数段但边框命中越少越好”的目标相冲突。
            # 轻微惩罚极低填充（通常是只蹭到一点mask）
            if fill_ratio < 0.03:
                score -= 0.25 * float(clip_area_px) * (0.03 - float(fill_ratio))

            if score > best_score:
                best_score = score
                best_box = box

        if best_box is None:
            # 严格 overlap 阶段没找到任何可用候选 -> 兜底：允许忽略 overlap，优先把剩余mask盖住
            if allow_overlap_fallback and np.any(uncovered_mask):
                best_box = None
                best_score = -1e18
                for (x_norm, y_norm) in candidate_positions:
                    box = (float(x_norm), float(y_norm), float(x_norm + clip_w_norm), float(y_norm + clip_h_norm))
                    if box[2] > 1.0 or box[3] > 1.0:
                        continue
                    new_cov = _get_coverage_score(box, uncovered_mask, width, height)
                    if new_cov <= 0:
                        continue
                    # fallback 直接按新增覆盖排序（确保不漏覆盖）
                    score = float(new_cov)
                    if score > best_score:
                        best_score = score
                        best_box = box
                if best_box is None:
                    break
            else:
                break

        selected_boxes.append(best_box)

        # 更新未覆盖
        x_min_px = max(0, int(best_box[0] * width))
        y_min_px = max(0, int(best_box[1] * height))
        x_max_px = min(width, int(best_box[2] * width))
        y_max_px = min(height, int(best_box[3] * height))
        if x_max_px > x_min_px and y_max_px > y_min_px:
            uncovered_mask[y_min_px:y_max_px, x_min_px:x_max_px] = False

    return selected_boxes


def generate_clip_boxes2(
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
    0. 算法的目标是针对赛道类的mask图像，使用一系列小的clip对其进行切分
    1. 先对mask进行膨胀，膨胀的尺寸为overlap_ratio所对应的尺寸，随后的clip操作，目标是覆盖所有膨胀后的mask像素
    2. 每个clip切到mask的部分，都需要确保与其他clip有overlap_ratio比例的重叠
    3. 使用全局动态规划算法，使用0.01个clip尺寸的步长进行动态规划搜索
    4. 搜索的优化目标是：最小化clip数量，最大化每个clip能切到的mask像素数量
    5. 将clip边框上的连续像素看作是一个一维数组，例如如果一条赛道的mask从中间穿过clip，则它会在边框上留下两端连续的有效区域。如果一个clip的连续有效区域个数是偶数，则奖励偶数个连续区域中的总像素数，否则惩罚奇数个连续区域中的总像素数（这一条的目标是尽量对赛道进行完整的切分，避免切割边缘在赛道的路面中心）
    """

    orig_mask_array, mask_array = compute_dilated_mask_for_clips(mask, clip_size, overlap_ratio)
    height, width = mask_array.shape

    # 用原始mask做距离变换：可选的轻微“切到核心”惩罚项
    # 缺少scipy时自动降级为None（仍可用边界贴mask长度/坏切法惩罚来优化）
    dist_in_mask = None
    if ndimage is not None:
        dist_in_mask = cast(np.ndarray, ndimage.distance_transform_edt(orig_mask_array.astype(np.uint8)))  # type: ignore
    
    # 找到有效区域的边界框
    valid_pixels = np.where(mask_array)
    if len(valid_pixels[0]) == 0:
        # 如果没有有效像素，返回空列表
        return []
    
    # --- 实现：全局覆盖（贪心近似 + 边框偶/奇段规则） ---
    # 说明：注释里提到“全局DP”，但在赛道mask这种长条形目标上，
    # 使用“全局候选 + 贪心集合覆盖 + 规则化评分”更稳、更易控时延；
    # 仍保留 0.01*clip 尺寸的搜索粒度来满足精细对齐需求。

    search_step = min(clip_size[0], clip_size[1]) * 0.01
    all_boxes: List[Tuple[float, float, float, float]] = []

    # 多轮补洞，直到完全覆盖膨胀mask（避免“剩下一点点像素一直没被盖住”）
    for pass_idx in range(6):
        covered_mask = np.zeros_like(mask_array, dtype=bool)
        for box in all_boxes:
            x_min, y_min, x_max, y_max = box
            x_min_px = max(0, int(x_min * width))
            y_min_px = max(0, int(y_min * height))
            x_max_px = min(width, int(x_max * width))
            y_max_px = min(height, int(y_max * height))
            if x_max_px > x_min_px and y_max_px > y_min_px:
                covered_mask[y_min_px:y_max_px, x_min_px:x_max_px] = True

        uncovered_mask = np.logical_and(mask_array.astype(bool), ~covered_mask) if len(all_boxes) > 0 else mask_array
        if not np.any(uncovered_mask):
            break

        # pass0：主覆盖（重叠更接近 r 且抑制过度重叠）
        # 后续pass：逐步放宽 min_overlap_ratio，优先补洞但仍有过度重叠上限
        relax = 1.0 if pass_idx == 0 else max(0.15, 1.0 - 0.18 * float(pass_idx))
        boxes_new = _greedy_cover_global_v2(
            mask=uncovered_mask,
            clip_size=clip_size,
            min_overlap_ratio=float(overlap_ratio) * float(relax),
            width=width,
            height=height,
            search_step=float(search_step),
            orig_mask=orig_mask_array,
            dist_in_mask=dist_in_mask,
            evenodd_weight=1.0,
            core_cut_weight=0.25,
            density_weight=0.10,
            border_thickness_ratio=0.02,
            max_candidates=3200,
            overlap_excess_weight=0.35 if pass_idx == 0 else 0.18,
            max_overlap_area_ratio=0.85,
            allow_overlap_fallback=True
        )
        if len(boxes_new) == 0:
            break
        all_boxes.extend(boxes_new)

        is_covered, _ = _verify_coverage(mask_array, all_boxes, width, height)
        if is_covered:
            break
    
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
    dilated_mask: Union[np.ndarray, Image.Image, None] = None,
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

    dilated_display = None
    if dilated_mask is not None:
        if isinstance(dilated_mask, Image.Image):
            dm = np.array(dilated_mask)
            if len(dm.shape) == 3:
                dilated_display = np.any(dm > 0, axis=2)
            else:
                dilated_display = dm > 0
        else:
            dm = dilated_mask
            if len(dm.shape) == 3:
                dilated_display = np.any(dm > 0, axis=2)
            else:
                dilated_display = dm > 0
    
    # 创建图像显示
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # 显示mask：原始(红) + 膨胀(蓝)
    ax.imshow(display_mask, cmap='Reds', alpha=0.45, origin='upper')
    if dilated_display is not None:
        ax.imshow(dilated_display, cmap='Blues', alpha=0.25, origin='upper')
    
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
    boxes = generate_clip_boxes2(mask, clip_size, overlap_ratio)
    _, dilated_mask = compute_dilated_mask_for_clips(mask, clip_size, overlap_ratio)
    
    print(f"Generated {len(boxes)} clip boxes:")
    for idx, box in enumerate(boxes):
        print(f"  Clip {idx + 1}: ({box[0]:.4f}, {box[1]:.4f}, {box[2]:.4f}, {box[3]:.4f})")
    
    # 可视化结果
    output_dir = "output"
    import os
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "clip_boxes_visualization.png")
    visualize_clip_boxes(mask, boxes, dilated_mask=dilated_mask, save_path=save_path, show_plot=True)