"""
简单的缝隙测量工具 - 分析 Blender JSON 中相邻多边形的顶点距离
"""
import json
import sys
from pathlib import Path
from typing import List, Tuple


def load_blender_json(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def measure_min_distances(json_path: str) -> None:
    """测量 Blender JSON 中所有多边形之间的最小顶点距离"""
    data = load_blender_json(json_path)
    polygons = data.get('polygons', {})

    all_polys = []
    for poly in polygons.get('include', []):
        pts = poly.get('points_xyz', [])
        if len(pts) >= 3:
            all_polys.append(pts)

    print(f"加载了 {len(all_polys)} 个多边形")

    if len(all_polys) < 2:
        print("多边形数量不足，无法测量间距")
        return

    # 测量相邻多边形的最小距离
    min_gaps = []
    for i in range(len(all_polys)):
        for j in range(i + 1, len(all_polys)):
            min_dist = float('inf')
            for pt1 in all_polys[i]:
                for pt2 in all_polys[j]:
                    dx = pt1[0] - pt2[0]
                    dy = pt1[1] - pt2[1]
                    dz = pt1[2] - pt2[2]
                    dist = (dx*dx + dy*dy + dz*dz) ** 0.5
                    if dist < min_dist:
                        min_dist = dist

            if min_dist < 1.0:  # 只记录小于1米的间距
                min_gaps.append(min_dist)

    if min_gaps:
        min_gaps.sort()
        print(f"\n找到 {len(min_gaps)} 对相邻多边形（间距 < 1m）:")
        print(f"  最小间距: {min_gaps[0]:.6f}m")
        print(f"  最大间距: {min_gaps[-1]:.6f}m")
        print(f"  平均间距: {sum(min_gaps)/len(min_gaps):.6f}m")
        print(f"  中位数: {min_gaps[len(min_gaps)//2]:.6f}m")
    else:
        print("未找到相邻多边形（间距 < 1m）")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python tests/measure_gaps.py <blender_json_path>")
        sys.exit(1)

    measure_min_distances(sys.argv[1])
