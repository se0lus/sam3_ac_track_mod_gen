"""
统一轮廓提取器 - 从composite一次性提取所有tag的轮廓，保证共享边界零缝隙

核心思想：
1. 对每个tag从composite提取轮廓（一次性）
2. 在提取过程中，同时识别哪些像素是共享边界
3. 将轮廓分段：共享段和独立段
4. 对每个段简化一次
5. 组装最终轮廓时，共享段被多个tag引用（同一个对象）
"""

import json
import numpy as np
import cv2
import os
import logging
from typing import Dict, List, Tuple, Optional, Any
from mask_merger import _triangulate_group_earcut

logger = logging.getLogger(__name__)

# tag_id -> 可读名（与 mask_gap_filler SURFACE_TAGS 顺序一致：sand, grass, road2, road, kerb）
TAG_ID_TO_NAME = {1: "sand", 2: "grass", 3: "road2", 4: "road", 5: "kerb"}

# Global debug directory
_debug_dir = None


def set_debug_dir(debug_dir: Optional[str]) -> None:
    """Set debug output directory"""
    global _debug_dir
    _debug_dir = debug_dir
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        logger.info(f"Unified contour extractor debug dir: {debug_dir}")


def extract_contours_with_boundary_info(
    composite: np.ndarray,
    tag_id: int
) -> List[Tuple[np.ndarray, List[int]]]:
    """从composite中提取tag的轮廓，同时标记每个像素的邻居tag

    Returns:
        List of (contour_pixels, neighbor_tags)
        - contour_pixels: shape (N, 1, 2) from cv2.findContours
        - neighbor_tags: List[int] of length N, 0表示无邻居或边界
    """
    h, w = composite.shape
    binary = (composite == tag_id).astype(np.uint8) * 255

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    result = []
    for contour in contours:
        pixels = contour.reshape(-1, 2)
        neighbor_tags = []

        for px, py in pixels:
            # 检查4邻域
            neighbors = set()
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = px + dx, py + dy
                if 0 <= nx < w and 0 <= ny < h:
                    neighbor_val = composite[ny, nx]
                    if neighbor_val != 0 and neighbor_val != tag_id:
                        neighbors.add(int(neighbor_val))

            # 记录最高优先级的邻居（如果有多个）
            neighbor_tags.append(max(neighbors) if neighbors else 0)

        result.append((contour, neighbor_tags))

    return result


def segment_and_simplify_contour(
    contour_pixels: np.ndarray,
    neighbor_tags: List[int],
    tag_id: int,
    epsilon: float
) -> List[dict]:
    """将轮廓分段并简化（处理闭合轮廓）"""
    pixels = contour_pixels.reshape(-1, 2)

    if len(pixels) < 2:
        return []

    # 处理闭合轮廓：如果首尾neighbor相同，先找到第一个不同的位置作为起点
    start_idx = 0
    if neighbor_tags[0] == neighbor_tags[-1]:
        for i in range(1, len(neighbor_tags)):
            if neighbor_tags[i] != neighbor_tags[0]:
                start_idx = i
                break

    # 重新排列，从start_idx开始
    if start_idx > 0:
        pixels = np.concatenate([pixels[start_idx:], pixels[:start_idx]])
        neighbor_tags = neighbor_tags[start_idx:] + neighbor_tags[:start_idx]

    segments = []
    current_segment = [pixels[0]]
    current_neighbor = neighbor_tags[0]

    for i in range(1, len(pixels)):
        px = pixels[i]
        neighbor = neighbor_tags[i]

        if neighbor != current_neighbor:
            # 保存当前段
            if len(current_segment) >= 2:
                seg_array = np.array(current_segment, dtype=np.float32).reshape(-1, 1, 2)
                simplified = cv2.approxPolyDP(seg_array, epsilon, False)

                segments.append({
                    'pixels': simplified.reshape(-1, 2),
                    'neighbor': current_neighbor,
                    'is_shared': current_neighbor != 0,
                    'tag_pair': (min(tag_id, current_neighbor), max(tag_id, current_neighbor)) if current_neighbor != 0 else None
                })

            current_segment = [px]
            current_neighbor = neighbor
        else:
            current_segment.append(px)

    # 保存最后一段（包括单像素段）
    if len(current_segment) >= 1:
        seg_array = np.array(current_segment, dtype=np.float32).reshape(-1, 1, 2)
        simplified = cv2.approxPolyDP(seg_array, epsilon, False)

        segments.append({
            'pixels': simplified.reshape(-1, 2),
            'neighbor': current_neighbor,
            'is_shared': current_neighbor != 0,
            'tag_pair': (min(tag_id, current_neighbor), max(tag_id, current_neighbor)) if current_neighbor != 0 else None
        })

    return segments


def pixel_to_geo(px: int, py: int, bounds: dict, canvas_w: int, canvas_h: int) -> Tuple[float, float]:
    """将像素坐标转换为地理坐标"""
    left, bottom, right, top = bounds['left'], bounds['bottom'], bounds['right'], bounds['top']
    geo_x = left + (px / canvas_w) * (right - left)
    geo_y = top - (py / canvas_h) * (top - bottom)
    return (geo_x, geo_y)


def build_shared_boundary_library(
    all_segments: Dict[int, List[List[dict]]],
    bounds: dict,
    canvas_w: int,
    canvas_h: int
) -> Dict[Tuple[int, int], List[dict]]:
    """从所有tag的段中识别共享边界

    Returns:
        {tag_pair: [{'tag_id': int, 'segment_idx': (contour_idx, seg_idx), 'geo_coords': [...]}]}
    """
    shared_boundaries = {}

    for tag_id, contours_segments in all_segments.items():
        for contour_idx, segments in enumerate(contours_segments):
            for seg_idx, segment in enumerate(segments):
                if segment['is_shared']:
                    pair = segment['tag_pair']

                    # 转换为地理坐标
                    geo_coords = [
                        pixel_to_geo(int(px), int(py), bounds, canvas_w, canvas_h)
                        for px, py in segment['pixels']
                    ]

                    if pair not in shared_boundaries:
                        shared_boundaries[pair] = []

                    shared_boundaries[pair].append({
                        'tag_id': tag_id,
                        'segment_idx': (contour_idx, seg_idx),
                        'geo_coords': geo_coords,
                        'pixels': segment['pixels']
                    })

    return shared_boundaries


def assemble_final_contour(
    segments: List[dict],
    tag_id: int,
    shared_boundaries: Dict[Tuple[int, int], List[dict]],
    bounds: dict,
    canvas_w: int,
    canvas_h: int
) -> List[Tuple[float, float]]:
    """组装最终轮廓，引用共享边界"""
    result_coords = []

    for segment in segments:
        if segment['is_shared']:
            pair = segment['tag_pair']
            # 从共享边界库中找到对应的坐标
            if pair in shared_boundaries:
                shared_segs = shared_boundaries[pair]
                # 使用第一个出现的共享段（保证所有tag使用相同坐标）
                result_coords.extend(shared_segs[0]['geo_coords'])
        else:
            # 使用独立段的坐标
            geo_coords = [
                pixel_to_geo(int(px), int(py), bounds, canvas_w, canvas_h)
                for px, py in segment['pixels']
            ]
            result_coords.extend(geo_coords)

    return result_coords


def extract_all_contours_unified(
    composite: np.ndarray,
    bounds: dict,
    canvas_w: int,
    canvas_h: int,
    simplify_epsilon: float = 1.0,
    min_contour_area: int = 100,
) -> Dict[int, List[dict]]:
    """统一提取所有tag的轮廓，保证共享边界零缝隙

    Args:
        composite: 合成mask，每个像素值为tag ID
        bounds: {'left', 'bottom', 'right', 'top'}
        canvas_w, canvas_h: composite的尺寸
        simplify_epsilon: 简化参数（像素单位）
        min_contour_area: 最小轮廓面积（像素）

    Returns:
        {tag_id: [{'vertices': [...], 'faces': [...]}]}
    """
    logger.info("=== Unified Contour Extraction ===")
    logger.info(f"Canvas: {canvas_w}x{canvas_h}, epsilon: {simplify_epsilon}, min_area: {min_contour_area}")

    # Step 1: 为每个tag提取轮廓并标记邻居
    all_tag_data = {}
    for tag_id in [1, 2, 3, 4, 5]:  # sand, grass, road2, road, kerb
        if np.any(composite == tag_id):
            n_pixels = int(np.count_nonzero(composite == tag_id))
            contours_info = extract_contours_with_boundary_info(composite, tag_id)
            if contours_info:
                all_tag_data[tag_id] = contours_info
                logger.info(f"Tag {tag_id}: {len(contours_info)} contours, {n_pixels} pixels")
            else:
                logger.warning(f"Tag {tag_id}: {n_pixels} pixels but no contours extracted!")
        else:
            logger.info(f"Tag {tag_id}: no pixels")

    # Step 2: 分段并简化
    logger.info("Step 2: Segmenting and simplifying contours...")
    all_segments = {}
    for tag_id, contours_info in all_tag_data.items():
        tag_segments = []
        n_filtered = 0
        for contour, neighbors in contours_info:
            # 过滤小轮廓
            area = cv2.contourArea(contour)
            if area < min_contour_area:
                n_filtered += 1
                continue

            segments = segment_and_simplify_contour(contour, neighbors, tag_id, simplify_epsilon)
            if segments:
                tag_segments.append(segments)

        if tag_segments:
            all_segments[tag_id] = tag_segments
            total_segments = sum(len(segs) for segs in tag_segments)
            logger.info(f"  Tag {tag_id}: {len(tag_segments)} contours, {total_segments} segments (filtered {n_filtered} small contours)")
        else:
            logger.warning(f"  Tag {tag_id}: all contours filtered out!")

    # Step 3: 构建共享边界库
    logger.info("Step 3: Building shared boundary library...")
    shared_boundaries = build_shared_boundary_library(all_segments, bounds, canvas_w, canvas_h)
    logger.info(f"  Found {len(shared_boundaries)} shared boundary pairs")
    for pair, segs in shared_boundaries.items():
        logger.info(f"    Pair {pair}: {len(segs)} segments")

    # Step 4: 为每个tag组装最终轮廓并三角化
    logger.info("Step 4: Assembling contours and triangulating...")
    result = {}
    for tag_id, tag_segments_list in all_segments.items():
        groups = []
        n_failed_triangulation = 0
        for contour_idx, segments in enumerate(tag_segments_list):
            geo_coords = assemble_final_contour(segments, tag_id, shared_boundaries, bounds, canvas_w, canvas_h)

            if len(geo_coords) >= 3:
                # 三角化 - 转换tuple为list
                outer_geo = [[float(x), float(y)] for x, y in geo_coords]
                faces = _triangulate_group_earcut(outer_geo, [])
                if faces:
                    groups.append({
                        'vertices': geo_coords,
                        'faces': faces
                    })
                else:
                    n_failed_triangulation += 1
                    logger.warning(f"  Tag {tag_id} contour {contour_idx}: triangulation failed ({len(geo_coords)} vertices)")
            else:
                logger.warning(f"  Tag {tag_id} contour {contour_idx}: too few vertices ({len(geo_coords)})")

        if groups:
            result[tag_id] = groups
            total_verts = sum(len(g['vertices']) for g in groups)
            total_faces = sum(len(g['faces']) for g in groups)
            logger.info(f"  Tag {tag_id}: {len(groups)} groups, {total_verts} vertices, {total_faces} faces")
            if n_failed_triangulation > 0:
                logger.warning(f"  Tag {tag_id}: {n_failed_triangulation} contours failed triangulation")
        else:
            logger.warning(f"  Tag {tag_id}: no valid groups generated!")

    logger.info(f"=== Extraction complete: {len(result)} tags with geometry ===")
    return result

