import os
import re
import json
import tempfile
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
from PIL import Image
from torchvision.transforms.transforms import F
from geo_tiff_image import GeoTiffImage
import rasterio
from rasterio.windows import Window
from rasterio.transform import from_bounds

try:
    import torch
except ImportError:
    torch = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    plt = None
    mpatches = None

def convert_mask_to_polygon(
    mask: Image.Image,
    error_threshold: float = 0.0001,
    ignore_holes: bool = False,
    size_threshold: float = 0.0001,
    use_pixel_center: bool = False,
) -> Dict[str, List[np.ndarray]]:
    """
    将 mask 转换为归一化的 polygon, 至少为3个点，否则输出空列表
    
    Args:
        mask: PIL Image，灰度图，0-255
        error_threshold: 误差阈值，用于多边形简化（相对于轮廓周长的比例），默认0.0001
                        例如：0.01 表示 epsilon = 轮廓周长 * 0.01
                        值越小，精度越高，但顶点数越多
        ignore_holes: 是否忽略内部轮廓（孔洞），默认False
        size_threshold: 大小阈值，像素点个数占mask图像总像素点个数的比例，默认0.0001
                        面积小于此阈值的轮廓将被忽略
        use_pixel_center: 是否使用像素中心坐标，默认False（使用像素外边界）
                         - False: 框住整个mask
                         - True: 使用像素中心坐标（整数坐标+0.5）
    
    Returns:
        Dict[str, List[np.ndarray]]:
            - include: 外轮廓（需要保留/填充的区域）
            - exclude: 内轮廓（孔洞，需要从 include 中扣除的区域）
            每个 polygon 为扁平化坐标 [x0, y0, x1, y1, ...]（归一化到[0, 1]）
    """
    def _ring_to_points(ring: List[Tuple[float, float]]) -> np.ndarray:
        """GeoJSON ring -> Nx2 points, drop duplicated last point if closed."""
        pts = np.asarray(ring, dtype=np.float64)
        if pts.shape[0] >= 2 and np.allclose(pts[0], pts[-1]):
            pts = pts[:-1]
        return pts

    def _polygon_area(points: np.ndarray) -> float:
        """Shoelace area for Nx2 polygon (not necessarily closed)."""
        if points.shape[0] < 3:
            return 0.0
        x = points[:, 0]
        y = points[:, 1]
        return 0.5 * float(np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

    def _polyline_rdp(points: np.ndarray, eps: float) -> np.ndarray:
        """Ramer–Douglas–Peucker simplification for an open polyline (keeps endpoints)."""
        if points.shape[0] <= 2 or eps <= 0:
            return points

        start = points[0]
        end = points[-1]
        seg = end - start
        seg_len2 = float(np.dot(seg, seg))

        if seg_len2 == 0.0:
            # Degenerate segment: distance to a point
            dists = np.linalg.norm(points[1:-1] - start, axis=1) if points.shape[0] > 2 else np.array([])
        else:
            # Perpendicular distance to segment
            v = points[1:-1] - start
            t = (v @ seg) / seg_len2
            t = np.clip(t, 0.0, 1.0)
            proj = start + np.outer(t, seg)
            dists = np.linalg.norm(points[1:-1] - proj, axis=1)

        if dists.size == 0:
            return points

        idx = int(np.argmax(dists))
        max_dist = float(dists[idx])
        if max_dist <= eps:
            return np.vstack([start, end])

        # recurse
        split = idx + 1
        left = _polyline_rdp(points[: split + 1], eps)
        right = _polyline_rdp(points[split:], eps)
        return np.vstack([left[:-1], right])

    def _simplify_closed_ring(points: np.ndarray, eps: float) -> np.ndarray:
        """
        Simplify a closed ring by cutting it into two open polylines between two far points,
        simplifying both, then stitching back.
        """
        if points.shape[0] < 4 or eps <= 0:
            return points

        # pick two far-ish points (2-pass farthest heuristic)
        p0 = points[0]
        ia = int(np.argmax(np.sum((points - p0) ** 2, axis=1)))
        pa = points[ia]
        ib = int(np.argmax(np.sum((points - pa) ** 2, axis=1)))
        if ia == ib:
            return points

        if ia > ib:
            ia, ib = ib, ia

        path1 = points[ia : ib + 1]  # pa -> pb
        path2 = np.vstack([points[ib:], points[: ia + 1]])  # pb -> pa (wrap)

        s1 = _polyline_rdp(path1, eps)
        s2 = _polyline_rdp(path2, eps)

        stitched = np.vstack([s1[:-1], s2[:-1]])
        # keep at least 3 vertices
        if stitched.shape[0] < 3:
            return points
        return stitched
    
    # 将PIL Image转换为numpy数组
    mask_array = np.array(mask)
    
    # 转换为灰度图（如果是彩色图）
    if len(mask_array.shape) == 3:
        # 使用第一个通道或转换为灰度
        mask_array = mask_array[:, :, 0]
    
    # 转换为二值mask（0和255）
    if mask_array.dtype != np.uint8:
        # 如果值在[0, 1]范围内，转换为[0, 255]
        if mask_array.max() <= 1.0:
            mask_array = (mask_array * 255).astype(np.uint8)
        else:
            mask_array = mask_array.astype(np.uint8)
    
    # 二值化：将非零值视为前景
    binary_mask = mask_array > 127
    
    # 获取图像尺寸用于归一化
    height, width = binary_mask.shape
    total_pixels = height * width
    min_area = int(total_pixels * size_threshold)

    include_polygons: List[np.ndarray] = []
    exclude_polygons: List[np.ndarray] = []

    # 目标：
    # - use_pixel_center=False: 尽可能“框住/包住”mask（允许略微外扩），并尽量减少锯齿
    # - use_pixel_center=True : 使用像素中心坐标系（+0.5），通常用于与像素中心对齐的场景
    #
    # 实现策略：优先使用 OpenCV 进行形态学平滑 + 轮廓提取 + 多边形简化。
    # 这样比直接栅格矢量化（严格像素阶梯边界）更不容易出现密集锯齿点。
    if cv2 is not None:
        binary_u8 = (binary_mask.astype(np.uint8) * 255)
        binary_u8 = np.ascontiguousarray(binary_u8)

        # 对 use_pixel_center=False 做“包住”与去锯齿处理：
        # 注意：当需要输出 exclude（孔洞）时，形态学操作可能改变孔洞拓扑；因此仅在 ignore_holes=True 时启用。
        if (not use_pixel_center) and ignore_holes:
            k = np.ones((3, 3), dtype=np.uint8)
            binary_u8 = cv2.morphologyEx(binary_u8, cv2.MORPH_CLOSE, k, iterations=1)
            binary_u8 = cv2.dilate(binary_u8, k, iterations=1)

        mode = cv2.RETR_EXTERNAL if ignore_holes else cv2.RETR_TREE
        contour_result = cv2.findContours(
            binary_u8,
            mode,
            cv2.CHAIN_APPROX_SIMPLE,  # 先压缩共线点，避免天然阶梯边界产生大量顶点
        )

        if len(contour_result) == 2:
            contours, hierarchy = contour_result
        else:
            contours = contour_result[1]
            hierarchy = contour_result[2]

        if not contours:
            return {"include": [], "exclude": []}

        def _contour_depth(idx: int) -> int:
            """
            hierarchy: [Next, Previous, First_Child, Parent]
            depth=0 表示外轮廓；depth=1 表示孔洞；depth=2 表示孔洞中的岛...
            """
            if hierarchy is None:
                return 0
            depth = 0
            parent = int(hierarchy[0][idx][3])
            while parent >= 0:
                depth += 1
                parent = int(hierarchy[0][parent][3])
            return depth

        for i, contour in enumerate(contours):
            depth = _contour_depth(i)
            poly_type = "include" if (depth % 2 == 0) else "exclude"
            if ignore_holes and poly_type == "exclude":
                continue

            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            # 多边形简化：epsilon 按周长比例，但对像素边界模式给一个最小下限（避免 error_threshold 太小几乎不简化）
            contour_perimeter = cv2.arcLength(contour, closed=True)
            if error_threshold > 0:
                epsilon = contour_perimeter * float(error_threshold)
                if not use_pixel_center:
                    epsilon = max(epsilon, 1.0)  # 至少 1px 量级的简化，显著减少锯齿顶点
                contour = cv2.approxPolyDP(contour, epsilon, closed=True)

            if len(contour) < 3:
                continue

            flattened = contour.reshape(-1, 2).astype(np.float64).flatten()

            if use_pixel_center:
                flattened = flattened + 0.5

            flattened[0::2] = flattened[0::2] / width
            flattened[1::2] = flattened[1::2] / height
            if poly_type == "include":
                include_polygons.append(flattened)
            else:
                exclude_polygons.append(flattened)

        return {"include": include_polygons, "exclude": exclude_polygons}

    # 无 OpenCV 时，使用 rasterio.features.shapes 将栅格前景转为像素边界多边形。
    # 注意：该方式天然是严格像素阶梯边界，必须依赖较强的简化来减轻锯齿。
    try:
        from rasterio.features import shapes as rio_shapes
        from affine import Affine

        transform = Affine.translation(0.5, 0.5) if use_pixel_center else Affine.identity()
        src = binary_mask.astype(np.uint8)

        for geom, val in rio_shapes(src, mask=binary_mask, transform=transform, connectivity=8):
            if not val:
                continue

            def _emit_ring(ring_coords: List[Tuple[float, float]], poly_type: str) -> None:
                pts = _ring_to_points(ring_coords)
                if pts.shape[0] < 3:
                    return

                area = _polygon_area(pts)
                if area < float(min_area):
                    return

                # 简化（可选）：基于周长的相对误差阈值
                if error_threshold > 0 and pts.shape[0] >= 4:
                    diffs = np.diff(np.vstack([pts, pts[:1]]), axis=0)
                    perimeter = float(np.sum(np.linalg.norm(diffs, axis=1)))
                    eps = perimeter * float(error_threshold)
                    if not use_pixel_center:
                        eps = max(eps, 1.0)  # 与 OpenCV 分支一致：给 1px 级别的简化下限
                    if eps > 0:
                        pts2 = _simplify_closed_ring(pts, eps)
                        if pts2.shape[0] >= 3:
                            pts = pts2

                flattened = pts.reshape(-1).copy()
                flattened[0::2] = flattened[0::2] / width
                flattened[1::2] = flattened[1::2] / height
                if poly_type == "include":
                    include_polygons.append(flattened)
                else:
                    exclude_polygons.append(flattened)

            if geom.get("type") == "Polygon":
                rings = geom.get("coordinates", [])
                if rings:
                    # exterior
                    _emit_ring(rings[0], "include")
                    # holes (as standalone polygons) if requested
                    if not ignore_holes:
                        for hole in rings[1:]:
                            _emit_ring(hole, "exclude")
            elif geom.get("type") == "MultiPolygon":
                for poly in geom.get("coordinates", []):
                    if not poly:
                        continue
                    _emit_ring(poly[0], "include")
                    if not ignore_holes:
                        for hole in poly[1:]:
                            _emit_ring(hole, "exclude")

        return {"include": include_polygons, "exclude": exclude_polygons}

    except Exception:
        raise ImportError("需要安装 opencv-python 或 rasterio 才能将 mask 转为 polygon")


def visualize_polygons_on_mask(mask: Image.Image, polygons: Any, 
                                 figsize: Tuple[int, int] = (10, 10),
                                 show_original_mask: bool = True,
                                 alpha: float = 0.5,
                                 edge_color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
                                 face_color: Optional[Tuple[float, float, float, float]] = None,
                                 linewidth: float = 2.0,
                                 save_path: Optional[str] = None) -> None:
    """
    在mask图像上可视化polygon轮廓
    
    Args:
        mask: PIL Image，原始mask图像
        polygons: convert_mask_to_polygon返回的polygon列表
        figsize: 图像大小，默认(10, 10)
        show_original_mask: 是否显示原始mask图像作为背景，默认True
        alpha: mask背景透明度，默认0.5
        edge_color: 轮廓线颜色（RGB，范围0-1），默认红色(1.0, 0.0, 0.0)
        face_color: 多边形填充颜色（RGBA），如果为None则不填充
        linewidth: 轮廓线宽度，默认2.0
        save_path: 保存路径，如果为None则只显示不保存
    """
    if plt is None or mpatches is None:
        raise ImportError("需要安装 matplotlib: pip install matplotlib")
    
    # 获取图像尺寸
    width, height = mask.size
    
    # 创建图像和轴
    fig, ax = plt.subplots(figsize=figsize)
    
    # 显示原始mask图像（如果需要）
    if show_original_mask:
        mask_array = np.array(mask.convert('RGB'))
        ax.imshow(mask_array, alpha=alpha, extent=(0, width, height, 0))
    else:
        # 只显示白色背景
        ax.imshow(np.ones((height, width, 3)), extent=(0, width, height, 0))
    
    # 兼容：允许传入 {"include": [...], "exclude": [...]} 或旧的 List[np.ndarray]
    if isinstance(polygons, dict):
        include_list = polygons.get("include", []) or []
        exclude_list = polygons.get("exclude", []) or []
    else:
        include_list = polygons
        exclude_list = []

    # 绘制 include（红色）
    for i, polygon_flat in enumerate(include_list):
        # 将归一化坐标转换回像素坐标
        coords = polygon_flat.copy()
        coords[0::2] = coords[0::2] * width   # x坐标
        coords[1::2] = coords[1::2] * height  # y坐标
        
        # 转换为点列表格式 [(x0, y0), (x1, y1), ...]
        points = [(coords[j], coords[j+1]) for j in range(0, len(coords), 2)]
        
        # 创建多边形patch
        polygon_patch = mpatches.Polygon(
            points,
            edgecolor=edge_color,
            facecolor=face_color,
            linewidth=linewidth,
            fill=face_color is not None
        )
        ax.add_patch(polygon_patch)

    # 绘制 exclude（蓝色虚线）
    if exclude_list:
        for i, polygon_flat in enumerate(exclude_list):
            coords = polygon_flat.copy()
            coords[0::2] = coords[0::2] * width
            coords[1::2] = coords[1::2] * height
            points = [(coords[j], coords[j+1]) for j in range(0, len(coords), 2)]
            polygon_patch = mpatches.Polygon(
                points,
                edgecolor=(0.0, 0.5, 1.0),
                facecolor=(0.0, 0.0, 0.0, 0.0),
                linewidth=max(1.0, linewidth * 0.9),
                fill=False,
                linestyle="--",
            )
            ax.add_patch(polygon_patch)
    
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # 注意：图像坐标系y轴是反向的
    ax.set_aspect('equal')
    ax.set_title(f'Polygon ({len(polygons)})')
    
    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 已保存可视化结果到: {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_polygons_on_image(image: Image.Image, polygons: Any,
                                 figsize: Tuple[int, int] = (10, 10),
                                 alpha: float = 0.3,
                                 edge_color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
                                 face_color: Optional[Tuple[float, float, float, float]] = None,
                                 linewidth: float = 2.0,
                                 save_path: Optional[str] = None) -> None:
    """
    在原始图像上可视化polygon轮廓（用于查看polygon在原始图像上的位置）
    
    Args:
        image: PIL Image，原始图像
        polygons: convert_mask_to_polygon返回的polygon列表（坐标必须与image尺寸匹配）
        figsize: 图像大小，默认(10, 10)
        alpha: 多边形填充透明度，默认0.3
        edge_color: 轮廓线颜色（RGB，范围0-1），默认红色(1.0, 0.0, 0.0)
        face_color: 多边形填充颜色（RGBA），如果为None则使用edge_color+alpha
        linewidth: 轮廓线宽度，默认2.0
        save_path: 保存路径，如果为None则只显示不保存
    """
    if plt is None or mpatches is None:
        raise ImportError("需要安装 matplotlib: pip install matplotlib")
    
    # 获取图像尺寸
    width, height = image.size
    
    # 创建图像和轴
    fig, ax = plt.subplots(figsize=figsize)
    
    # 显示原始图像
    ax.imshow(image)
    
    # 设置填充颜色（如果没有指定，使用edge_color+alpha）
    if face_color is None:
        face_color = (*edge_color, alpha)
    
    if isinstance(polygons, dict):
        include_list = polygons.get("include", []) or []
        exclude_list = polygons.get("exclude", []) or []
    else:
        include_list = polygons
        exclude_list = []

    # 绘制 include
    for i, polygon_flat in enumerate(include_list):
        # 将归一化坐标转换回像素坐标
        coords = polygon_flat.copy()
        coords[0::2] = coords[0::2] * width   # x坐标
        coords[1::2] = coords[1::2] * height  # y坐标
        
        # 转换为点列表格式 [(x0, y0), (x1, y1), ...]
        points = [(coords[j], coords[j+1]) for j in range(0, len(coords), 2)]
        
        # 创建多边形patch
        polygon_patch = mpatches.Polygon(
            points,
            edgecolor=edge_color,
            facecolor=face_color,
            linewidth=linewidth,
            fill=True
        )
        ax.add_patch(polygon_patch)

    # 绘制 exclude（蓝色虚线）
    if exclude_list:
        for i, polygon_flat in enumerate(exclude_list):
            coords = polygon_flat.copy()
            coords[0::2] = coords[0::2] * width
            coords[1::2] = coords[1::2] * height
            points = [(coords[j], coords[j+1]) for j in range(0, len(coords), 2)]
            polygon_patch = mpatches.Polygon(
                points,
                edgecolor=(0.0, 0.5, 1.0),
                facecolor=(0.0, 0.0, 0.0, 0.0),
                linewidth=max(1.0, linewidth * 0.9),
                fill=False,
                linestyle="--",
            )
            ax.add_patch(polygon_patch)
    
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # 注意：图像坐标系y轴是反向的
    ax.set_aspect('equal')
    ax.set_title(f'Polygon  ({len(polygons)})')
    ax.axis('off')  # 隐藏坐标轴
    
    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 已保存可视化结果到: {save_path}")
    else:
        plt.show()
    
    plt.close()

class GeoSam3Image:
    def __init__(self, image_path: str):
        """
        初始化 GeoSam3Image 类
        
        Args:
            image_path: GeoTIFF 图像文件路径
        
        功能：
            - 加载 GeoTIFF 图像到 geo_image 属性（GeoTiffImage 对象）
            - 尝试加载同目录下同名但带有 _modelscale.png 后缀的缩放图像
            - 尝试加载同目录下对应的一系列 mask 图像（格式：_mask{数字}_prob({数字}).png）
        """
        self.image_path = os.path.abspath(image_path)
        
        # 检查文件是否存在
        if not os.path.exists(self.image_path):
            raise FileNotFoundError(f"图像文件不存在: {self.image_path}")
        
        # 获取文件目录和基础文件名（不含扩展名）
        self.directory = os.path.dirname(self.image_path)
        self.basename = os.path.splitext(os.path.basename(self.image_path))[0]
        
        # 加载 GeoTIFF 图像
        self.geo_image = GeoTiffImage(self.image_path)
        
        # 尝试加载 _modelscale.png 图像
        self.model_scale_image: Optional[Image.Image] = None
        self._load_model_scale_image()
        
        # 加载 mask 图像
        self.masks: Dict[int, Dict[str, Any]] = {}  # {mask_index: {'image': PIL.Image, 'prob': float, 'tag': str, 'path': str}}
        self._load_mask_images()
    
    def _load_model_scale_image(self):
        """尝试加载同目录下同名但带有 _modelscale.png 后缀的图像"""
        model_scale_path = os.path.join(self.directory, f"{self.basename}_modelscale.png")
        if os.path.exists(model_scale_path):
            try:
                self.model_scale_image = Image.open(model_scale_path).convert('RGB')
                print(f"✓ 已加载模型缩放图像: {model_scale_path}")
            except Exception as e:
                print(f"⚠ 无法加载模型缩放图像 {model_scale_path}: {e}")
        else:
            print(f"ℹ 未找到模型缩放图像: {model_scale_path}")
    
    def _load_mask_images(self):
        """
        扫描并加载同目录下所有匹配的 mask 图像
        支持两种文件名格式：
        1. {basename}_mask{数字}_prob({数字}).png (tag为None)
        2. {basename}_tag_mask{数字}_prob({数字}).png (tag不为None)
        例如：result_mask0_prob(0.91).png, result_tag_mask0_prob(0.91).png
        """
        # 正则表达式匹配 mask 文件名模式
        # 匹配格式1：{basename}_mask{数字}_prob({数字}).png (tag为None)
        pattern1 = re.compile(rf"^{re.escape(self.basename)}_mask(\d+)_prob\(([\d.]+)\)\.png$")
        # 匹配格式2：{basename}_tag_mask{数字}_prob({数字}).png (tag不为None)
        pattern2 = re.compile(rf"^{re.escape(self.basename)}_(.+?)_mask(\d+)_prob\(([\d.]+)\)\.png$")
        
        # 扫描目录中的所有文件
        if not os.path.isdir(self.directory):
            print(f"⚠ 目录不存在: {self.directory}")
            return
        
        for filename in os.listdir(self.directory):
            match = None
            mask_index = None
            prob_value = None
            tag = None
            
            # 先尝试匹配格式2（带tag的格式）
            match = pattern2.match(filename)
            if match:
                tag = match.group(1)
                mask_index = int(match.group(2))
                prob_value = float(match.group(3))
            else:
                # 再尝试匹配格式1（不带tag的格式，tag为None）
                match = pattern1.match(filename)
                if match:
                    tag = None
                    mask_index = int(match.group(1))
                    prob_value = float(match.group(2))
            
            if match and mask_index is not None and prob_value is not None:
                mask_path = os.path.join(self.directory, filename)
                
                try:
                    mask_image = Image.open(mask_path)
                    self.masks[mask_index] = {
                        'image': mask_image,
                        'prob': prob_value,
                        'tag': tag,
                        'path': mask_path
                    }
                    tag_str = f" (tag: {tag})" if tag is not None else ""
                    print(f"✓ 已加载 mask {mask_index}: {filename} (概率: {prob_value}{tag_str})")
                except Exception as e:
                    print(f"⚠ 无法加载 mask 图像 {mask_path}: {e}")
        
        if len(self.masks) == 0:
            print(f"ℹ 未找到任何 mask 图像（匹配模式: {self.basename}_mask{{数字}}_prob({{数字}}).png 或 {self.basename}_tag_mask{{数字}}_prob({{数字}}).png）")
    
    def get_mask_by_index(self, index: int) -> Optional[Dict[str, Any]]:
        """
        根据索引获取 mask
        
        Args:
            index: mask 索引
            
        Returns:
            包含 'image', 'prob', 'tag', 'path' 的字典，如果不存在则返回 None
        """
        return self.masks.get(index)
    
    def get_all_masks(self) -> Dict[int, Dict[str, Any]]:
        """
        获取所有加载的 masks
        
        Returns:
            所有 masks 的字典，键为 mask 索引
        """
        return self.masks.copy()
    
    def get_sorted_masks(self, sort_by_prob: bool = True, reverse: bool = True) -> List[Dict[str, Any]]:
        """
        获取排序后的 mask 列表
        
        Args:
            sort_by_prob: 如果为 True，按概率排序；如果为 False，按索引排序
            reverse: 如果为 True，降序排列；如果为 False，升序排列
            
        Returns:
            排序后的 mask 列表，每个元素包含 'index', 'image', 'prob', 'path'
        """
        masks_list = [
            {
                'index': idx,
                'image': mask['image'],
                'prob': mask['prob'],
                'tag': mask.get('tag'),
                'path': mask['path']
            }
            for idx, mask in self.masks.items()
        ]
        
        if sort_by_prob:
            masks_list.sort(key=lambda x: x['prob'], reverse=reverse)
        else:
            masks_list.sort(key=lambda x: x['index'], reverse=reverse)
        
        return masks_list
    
    def has_model_scale_image(self) -> bool:
        """检查是否已加载模型缩放图像"""
        return self.model_scale_image is not None
    
    def has_masks(self) -> bool:
        """检查是否已加载任何 mask"""
        return len(self.masks) > 0
    
    def set_masks(self, masks: List[Dict[str, Any]]):
        """
        从外部设置一系列新的mask，覆盖掉原有的
        
        Args:
            masks: mask列表，每个元素应该包含：
                - 'image': PIL.Image 对象（必需）
                - 'prob': float 概率值（可选，默认为0.0）
                - 'index': int mask索引（可选，如果不提供则从0开始自动分配）
                - 'tag': str tag标签（可选，如果为None或不提供，则使用旧格式）
        
        示例:
            masks = [
                {'image': pil_image1, 'prob': 0.95, 'index': 0, 'tag': None},
                {'image': pil_image2, 'prob': 0.87, 'index': 1, 'tag': 'racing'},
            ]
            geo_sam3_image.set_masks(masks)
        """
        # 清空现有的masks
        self.masks.clear()
        
        # 处理新的masks
        for i, mask_data in enumerate(masks):
            # 验证必需字段
            if 'image' not in mask_data:
                raise ValueError(f"mask {i} 缺少必需的 'image' 字段")
            
            # 获取或自动分配索引
            mask_index = mask_data.get('index', i)
            
            # 获取概率值，默认为0.0
            prob_value = mask_data.get('prob', 0.0)
            
            # 获取tag，默认为None
            tag = mask_data.get('tag', None)
            
            # 构建path（如果提供了则使用，否则生成）
            mask_path = mask_data.get('path')
            if mask_path is None:
                if tag is not None:
                    mask_filename = f"{self.basename}_{tag}_mask{mask_index}_prob({prob_value:.2f}).png"
                else:
                    mask_filename = f"{self.basename}_mask{mask_index}_prob({prob_value:.2f}).png"
                mask_path = os.path.join(self.directory, mask_filename)
            
            # 存储mask数据
            self.masks[mask_index] = {
                'image': mask_data['image'],
                'prob': prob_value,
                'tag': tag,
                'path': mask_path
            }
        
        print(f"✓ 已设置 {len(self.masks)} 个新的 mask，覆盖了原有的 mask")
    
    def set_masks_from_inference_state(self, inference_state: Dict[str, Any], tag: Optional[str] = None):
        """
        从 inference_state 中提取 masks 和 scores，并设置到属性中
        
        Args:
            inference_state: SAM3 处理器返回的 inference_state 字典，应包含：
                - 'masks': Tensor 或 numpy 数组，形状为 [N, H, W] 或 [N, 1, H, W]
                - 'scores': 分数列表或数组，长度为 N
            tag: 可选的标签字符串，用于标识这些 masks。如果为 None，则使用旧格式（不带tag）
        
        功能：
            - 从 inference_state 中提取 masks 和 scores
            - 将 masks 从 Tensor/numpy 数组转换为 PIL Image 格式
            - 调用 set_masks 方法将处理后的 masks 设置到属性中
        
        示例:
            inference_state = processor.set_text_prompt(state=inference_state, prompt="racing track")
            geo_sam3_image.set_masks_from_inference_state(inference_state, tag="racing")
        """
        # 从 inference_state 中提取 masks 和 scores
        if 'masks' not in inference_state:
            raise ValueError("inference_state 中缺少 'masks' 字段")
        if 'scores' not in inference_state:
            raise ValueError("inference_state 中缺少 'scores' 字段")
        
        masks = inference_state["masks"]
        scores = inference_state["scores"]
        
        # 将 scores 转换为列表（如果还不是的话）
        if hasattr(scores, 'tolist'):
            scores = scores.tolist()
        elif hasattr(scores, 'cpu'):
            scores = scores.cpu().numpy().tolist()
        else:
            scores = list(scores)
        
        # 转换 masks 为 PIL Image 列表
        masks_images = []
        for mask in masks:
            # 将 Tensor 移到 CPU 并转换为 numpy 数组
            if torch is not None and isinstance(mask, torch.Tensor):
                mask_np = mask.cpu().numpy()
            else:
                mask_np = np.array(mask)
            
            # 移除多余的维度（如果有）
            if mask_np.ndim > 2:
                mask_np = mask_np.squeeze()
            
            # 如果是布尔类型，转换为 uint8 (0 和 255)
            if mask_np.dtype == bool:
                mask_np = mask_np.astype(np.uint8) * 255
            elif mask_np.dtype != np.uint8:
                # 如果不是 uint8，先归一化到 [0, 1] 再转换为 uint8
                if mask_np.max() > 1.0:
                    mask_np = (mask_np / mask_np.max() * 255).astype(np.uint8)
                else:
                    mask_np = (mask_np * 255).astype(np.uint8)
            
            masks_images.append(Image.fromarray(mask_np))
        
        # 构建 mask 数据列表并调用 set_masks
        mask_data_list = [
            {
                'image': mask_image,
                'prob': float(score),
                'index': i,
                'tag': tag  # 使用传入的tag参数
            }
            for i, (mask_image, score) in enumerate(zip(masks_images, scores))
        ]
        
        self.set_masks(mask_data_list)
    
    def generate_model_scale_image(self, max_size: int = 1008, 
                                    use_gpu: bool = True,
                                    device: Optional[str] = None,
                                    gpu_chunk_size: Optional[int] = None,
                                    band_indices: Optional[List[int]] = None) -> Image.Image:
        """
        通过 GeoTiffImage 的 scale_to_max_size 方法生成 model_scale_image
        
        Args:
            max_size: 目标最大边长（像素），默认为 1008
            use_gpu: 如果为 True 且 PyTorch 可用，使用 GPU 加速（推荐）
            device: GPU 设备，例如 'cuda:0'；如果为 None 且 use_gpu=True，自动选择
            gpu_chunk_size: GPU 模式下处理大图像时的分块大小（像素），None 表示自动选择
            band_indices: 要读取的波段索引列表（例如 [1,2,3] 用于 RGB），如果为 None 则自动选择
        
        Returns:
            生成的 PIL Image 对象（同时会保存到 self.model_scale_image）
        """
        try:
            # 调用 geo_image 的 scale_to_max_size 方法
            scaled_image = self.geo_image.scale_to_max_size(
                max_size=max_size,
                window=None,  # 读取整个图像
                band_indices=band_indices,
                use_gpu=use_gpu,
                device=device,
                gpu_chunk_size=gpu_chunk_size
            )
            
            # 确保图像为 RGB 模式
            if scaled_image.mode != 'RGB':
                scaled_image = scaled_image.convert('RGB')
            
            # 保存到 model_scale_image 属性
            self.model_scale_image = scaled_image
            print(f"✓ 已生成模型缩放图像，尺寸: {scaled_image.size} (最大边长: {max_size})")
            
            return scaled_image
            
        except Exception as e:
            print(f"⚠ 生成模型缩放图像失败: {e}")
            raise
    
    def crop_and_scale_to_gsd(self, normalized_bbox: Tuple[float, float, float, float],
                               target_gsd: float,
                               use_average_gsd: bool = True,
                               use_gpu: bool = False,
                               device: Optional[str] = None,
                               gpu_chunk_size: Optional[int] = None,
                               band_indices: Optional[List[int]] = None,
                               dst_image_path: Optional[str] = None) -> 'GeoSam3Image':
        """
        根据归一化坐标裁剪图像并缩放到目标GSD，返回新的 GeoSam3Image 实例
        
        Args:
            normalized_bbox: 归一化坐标 (left, top, right, bottom)，范围 [0, 1]
                           例如: (0.411111, 0.313723, 0.561111, 0.463723)
            target_gsd: 目标GSD值（cm/pixel），即每个像素对应的厘米数
            use_average_gsd: 如果为True，使用x和y方向GSD的平均值；如果为False，使用x方向GSD
            use_gpu: 如果为True且PyTorch可用，使用GPU加速（推荐）
            device: GPU设备，例如'cuda:0'；如果为None且use_gpu=True，自动选择
            gpu_chunk_size: GPU模式下处理大图像时的分块大小（像素），None表示自动选择
            band_indices: 要读取的波段索引列表（例如[1,2,3]用于RGB），如果为None则自动选择
            dst_image_path: 输出图像文件路径（正式存储位置），如果为None则自动创建临时文件
        
        Returns:
            新的 GeoSam3Image 实例，包含裁剪并缩放后的图像
        
        示例:
            # 裁剪并缩放到目标GSD，保存到指定路径
            new_geo_image = geo_sam3_image.crop_and_scale_to_gsd(
                normalized_bbox=(0.411111, 0.313723, 0.561111, 0.463723),
                target_gsd=4.0,
                dst_image_path='output_cropped.tif',
                use_gpu=True
            )
        """
        # 验证归一化坐标
        left_norm, top_norm, right_norm, bottom_norm = normalized_bbox
        if not (0 <= left_norm < right_norm <= 1 and 0 <= top_norm < bottom_norm <= 1):
            raise ValueError(f"归一化坐标必须在 [0, 1] 范围内，且 left < right, top < bottom。"
                           f"当前值: ({left_norm}, {top_norm}, {right_norm}, {bottom_norm})")
        
        # 获取原始图像尺寸
        width, height = self.geo_image.get_size()
        
        # 将归一化坐标转换为像素坐标
        left_px = int(left_norm * width)
        top_px = int(top_norm * height)
        right_px = int(right_norm * width)
        bottom_px = int(bottom_norm * height)
        
        # 确保坐标在有效范围内
        left_px = max(0, min(left_px, width - 1))
        top_px = max(0, min(top_px, height - 1))
        right_px = max(left_px + 1, min(right_px, width))
        bottom_px = max(top_px + 1, min(bottom_px, height))
        
        # 计算窗口尺寸
        window_width = right_px - left_px
        window_height = bottom_px - top_px
        
        # 创建rasterio窗口对象（使用位置参数：col_off, row_off, width, height）
        window = Window(left_px, top_px, window_width, window_height)  # type: ignore
        
        # 调用 scale_to_gsd 方法进行裁剪和缩放
        scaled_image = self.geo_image.scale_to_gsd(
            target_gsd=target_gsd,
            window=window,
            band_indices=band_indices,
            use_average_gsd=use_average_gsd,
            use_gpu=use_gpu,
            device=device,
            gpu_chunk_size=gpu_chunk_size
        )
        
        # 创建临时的 GeoTiffImage 对象来处理裁剪后的区域
        # 首先需要获取窗口对应的地理信息
        # 使用 rasterio.windows.transform 计算窗口的 transform
        # 注：部分静态检查器无法识别 rasterio.windows 子模块属性，这里用显式导入避免误报。
        from rasterio.windows import transform as window_transform_fn  # type: ignore
        window_transform = window_transform_fn(window, self.geo_image.transform)
        
        # 计算窗口的地理边界
        # window_transform 是一个 Affine 对象，包含地理变换信息
        window_bounds_left = float(window_transform.c)  # type: ignore
        window_bounds_top = float(window_transform.f)  # type: ignore
        window_bounds_right = window_bounds_left + window_width * abs(float(window_transform.a))  # type: ignore
        window_bounds_bottom = window_bounds_top + window_height * float(window_transform.e)  # type: ignore
        
        # 获取缩放后的图像尺寸
        scaled_width, scaled_height = scaled_image.size
        
        # 计算新的transform（基于缩放后的尺寸和目标GSD）
        # 保持地理边界不变，调整分辨率
        new_transform = from_bounds(
            min(window_bounds_left, window_bounds_right),
            min(window_bounds_bottom, window_bounds_top),
            max(window_bounds_left, window_bounds_right),
            max(window_bounds_bottom, window_bounds_top),
            scaled_width,
            scaled_height
        )
        
        # 将PIL图像转换为numpy数组
        img_array = np.array(scaled_image)
        
        # 处理不同的图像模式
        if len(img_array.shape) == 2:
            # 灰度图，需要添加波段维度
            img_array = img_array[np.newaxis, :, :]  # (1, H, W)
            num_bands = 1
        elif len(img_array.shape) == 3:
            # 多波段图像，PIL是(H, W, C)，需要转换为(C, H, W)
            img_array = np.transpose(img_array, (2, 0, 1))  # (C, H, W)
            num_bands = img_array.shape[0]
        else:
            raise ValueError(f"不支持的图像形状: {img_array.shape}")
        
        # 确定数据类型（使用numpy类型，rasterio会自动转换）
        dtype = img_array.dtype
        
        # 确定输出文件路径
        if dst_image_path is None:
            # 如果未指定路径，创建临时文件
            temp_fd, dst_image_path = tempfile.mkstemp(suffix='.tif', prefix='geosam3_crop_')
            os.close(temp_fd)  # 关闭文件描述符，但保留路径
        else:
            # 确保输出目录存在
            output_dir = os.path.dirname(os.path.abspath(dst_image_path))
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
        
        # 创建新的GeoTIFF文件
        with rasterio.open(
            dst_image_path,
            'w',
            driver='GTiff',
            height=scaled_height,
            width=scaled_width,
            count=num_bands,
            dtype=dtype,
            crs=self.geo_image.crs,
            transform=new_transform,
            compress='lzw'
        ) as dst:
            dst.write(img_array)
        
        # 创建新的 GeoSam3Image 实例
        new_geo_sam3_image = GeoSam3Image(dst_image_path)
        
        print(f"✓ 已裁剪并缩放图像:")
        print(f"  • 原始窗口: ({left_px}, {top_px}, {right_px}, {bottom_px})")
        print(f"  • 缩放后尺寸: {scaled_width} × {scaled_height} 像素")
        print(f"  • 目标GSD: {target_gsd} cm/pixel")
        print(f"  • 输出文件: {dst_image_path}")
        
        return new_geo_sam3_image
    
    def merge_all_masks(self, mode: str = 'union', 
                        target_size: Optional[tuple] = None) -> Optional[Image.Image]:
        """
        合并所有的 mask 到一张图上
        
        Args:
            mode: 合并模式，可选值：
                - 'union': 并集模式，所有 mask 区域都会显示（逻辑或操作）
                - 'max': 最大值模式，每个像素取所有 mask 中的最大值
                - 'sum': 求和模式，将所有 mask 的像素值相加（可能超过255）
            target_size: 目标图像尺寸 (width, height)，如果为 None 则使用第一个 mask 的尺寸
        
        Returns:
            合并后的 PIL Image 对象，如果没有 mask 则返回 None
        
        示例:
            merged_mask = geo_sam3_image.merge_all_masks(mode='union')
            if merged_mask:
                merged_mask.save('merged_masks.png')
        """
        if len(self.masks) == 0:
            print("ℹ 没有 mask 可以合并")
            return None
        
        # 获取所有 mask 图像
        mask_images = []
        sorted_indices = sorted(self.masks.keys())
        
        for mask_index in sorted_indices:
            mask_data = self.masks[mask_index]
            mask_image = mask_data['image']
            mask_images.append(mask_image)
        
        # 确定目标尺寸
        if target_size is None:
            # 使用第一个 mask 的尺寸
            target_size = mask_images[0].size
        else:
            # 确保 target_size 是 (width, height) 格式
            if len(target_size) != 2:
                raise ValueError("target_size 必须是 (width, height) 格式")
        
        # 将所有 mask 转换为相同尺寸的 numpy 数组
        mask_arrays = []
        for mask_image in mask_images:
            # 转换为灰度图（如果还不是）
            if mask_image.mode != 'L':
                mask_image = mask_image.convert('L')
            
            # 调整尺寸（如果需要）
            if mask_image.size != target_size:
                mask_image = mask_image.resize(target_size, Image.Resampling.LANCZOS)
            
            # 转换为 numpy 数组并归一化到 [0, 1]
            mask_array = np.array(mask_image, dtype=np.float32) / 255.0
            mask_arrays.append(mask_array)
        
        # 根据模式合并
        if mode == 'union':
            # 并集：逻辑或操作，取最大值
            merged_array = np.maximum.reduce(mask_arrays)
        elif mode == 'max':
            # 最大值模式（与 union 相同）
            merged_array = np.maximum.reduce(mask_arrays)
        elif mode == 'sum':
            # 求和模式
            merged_array = np.sum(mask_arrays, axis=0)
            # 限制在 [0, 1] 范围内
            merged_array = np.clip(merged_array, 0, 1)
        else:
            raise ValueError(f"不支持的合并模式: {mode}，支持的模式: 'union', 'max', 'sum'")
        
        # 转换回 PIL Image
        merged_array = (merged_array * 255).astype(np.uint8)
        merged_image = Image.fromarray(merged_array, mode='L')
        
        print(f"✓ 已合并 {len(mask_images)} 个 mask，模式: {mode}，尺寸: {target_size}")
        
        return merged_image
    
    def save(self, output_dir: Optional[str] = None, overwrite: bool = True, 
             save_model_scale_image: bool = True, save_masks: bool = True,
             save_masks_json: bool = True) -> Dict[str, Any]:
        """
        持久化保存 model_scale_image 和/或所有 masks 到文件
        
        Args:
            output_dir: 输出目录路径，如果为 None 则使用原始图像所在目录
            overwrite: 如果为 True，覆盖已存在的文件；如果为 False，跳过已存在的文件
            save_model_scale_image: 如果为 True，保存 model_scale_image；如果为 False，跳过
            save_masks: 如果为 True，保存所有 masks；如果为 False，跳过
        
        Returns:
            包含保存文件路径的字典：
            {
                'model_scale_image': '保存路径或None',
                'masks': ['mask0路径', 'mask1路径', ...],
                'masks_json': 'masks.json 保存路径或None'
            }
        """
        # 确定输出目录
        if output_dir is None:
            output_dir = self.directory
        else:
            output_dir = os.path.abspath(output_dir)
            # 确保输出目录存在
            os.makedirs(output_dir, exist_ok=True)
        
        saved_files = {
            'model_scale_image': None,
            'masks': [],
            'masks_json': None,
        }
        
        # 保存 model_scale_image
        if save_model_scale_image:
            if self.model_scale_image is not None:
                model_scale_path = os.path.join(output_dir, f"{self.basename}_modelscale.png")
                if os.path.exists(model_scale_path) and not overwrite:
                    print(f"ℹ 跳过已存在的模型缩放图像: {model_scale_path}")
                else:
                    try:
                        self.model_scale_image.save(model_scale_path)
                        saved_files['model_scale_image'] = model_scale_path
                        print(f"✓ 已保存模型缩放图像: {model_scale_path}")
                    except Exception as e:
                        print(f"⚠ 无法保存模型缩放图像 {model_scale_path}: {e}")
            else:
                print(f"ℹ 没有模型缩放图像需要保存")
        
        # 保存所有 masks
        if save_masks:
            if len(self.masks) > 0:
                # 如果使用 overwrite 方式，先清除同目录下的 mask 历史文件
                if overwrite:
                    # 匹配两种格式：带tag和不带tag的
                    pattern1 = re.compile(rf"^{re.escape(self.basename)}_mask\d+_prob\([\d.]+\)\.png$")
                    pattern2 = re.compile(rf"^{re.escape(self.basename)}_.+?_mask\d+_prob\([\d.]+\)\.png$")
                    deleted_count = 0
                    for filename in os.listdir(output_dir):
                        if pattern1.match(filename) or pattern2.match(filename):
                            old_mask_path = os.path.join(output_dir, filename)
                            try:
                                os.remove(old_mask_path)
                                deleted_count += 1
                                print(f"🗑 已删除历史 mask 文件: {filename}")
                            except Exception as e:
                                print(f"⚠ 无法删除历史 mask 文件 {old_mask_path}: {e}")
                    if deleted_count > 0:
                        print(f"✓ 已清除 {deleted_count} 个历史 mask 文件")
                
                # 按索引排序以确保输出顺序一致
                sorted_indices = sorted(self.masks.keys())
                for mask_index in sorted_indices:
                    mask_data = self.masks[mask_index]
                    mask_image = mask_data['image']
                    prob_value = mask_data['prob']
                    tag = mask_data.get('tag', None)
                    
                    # 构建文件名：根据tag决定格式
                    if tag is not None:
                        mask_filename = f"{self.basename}_{tag}_mask{mask_index}_prob({prob_value:.2f}).png"
                    else:
                        mask_filename = f"{self.basename}_mask{mask_index}_prob({prob_value:.2f}).png"
                    mask_path = os.path.join(output_dir, mask_filename)
                    
                    if os.path.exists(mask_path) and not overwrite:
                        print(f"ℹ 跳过已存在的 mask {mask_index}: {mask_path}")
                    else:
                        try:
                            mask_image.save(mask_path)
                            saved_files['masks'].append(mask_path)
                            tag_str = f" (tag: {tag})" if tag is not None else ""
                            print(f"✓ 已保存 mask {mask_index}{tag_str}: {mask_path}")
                        except Exception as e:
                            print(f"⚠ 无法保存 mask {mask_index} {mask_path}: {e}")
            else:
                print(f"ℹ 没有 mask 图像需要保存")

        # 保存 masks 的 json（包含 polygon + 地理坐标信息）
        if save_masks and save_masks_json:
            try:
                json_result = self.save_masks_to_json_file(output_dir=output_dir, overwrite=overwrite)
                saved_files['masks_json'] = json_result.get('json_path')
            except Exception as e:
                print(f"⚠ 无法保存 masks json: {e}")
        
        return saved_files

    def save_masks_to_json_file(self, output_dir: Optional[str] = None, overwrite: bool = True) -> Dict[str, Any]:
        """
        将 `self.masks` 导出为 `*_masks.json`（polygon + 地理坐标 + 置信度/标签）。

        ## masks json 结构（version=1）

        顶层：
        - `meta`: 图像与地理元信息（基于 GeoTIFF 原图）
        - `masks`: mask 列表（每个 mask 会包含 polygon 以及多种坐标系下的点）

        ```json
        {
          "meta": {
            "version": 1,
            "source_image_path": "原始影像路径",
            "basename": "不含扩展名的基名",
            "image_size": { "width": 0, "height": 0 },
            "model_scale_size": { "width": 0, "height": 0 } | null,
            "geo": {
              "crs": "源 CRS 字符串(如 EPSG:xxxx) | null",
              "transform": [a, b, c, d, e, f],
              "bounds": { "left": 0, "bottom": 0, "right": 0, "top": 0 }
            }
          },
          "masks": [
            {
              "index": 0,
              "tag": "可选标签" | null,
              "prob": 0.0,
              "mask_size": { "width": 0, "height": 0 },
              "pixel_scale_to_original": { "sx": 0.0, "sy": 0.0 },
              "polygons": {
                "include": [
                  {
                    "norm_xy": [[x, y], ...],
                    "pixel_xy_in_mask": [[x, y], ...],
                    "pixel_xy_in_original": [[x, y], ...],
                    "geo_xy": [[x, y], ...]
                  }
                ],
                "exclude": [
                  {
                    "norm_xy": [[x, y], ...],
                    "pixel_xy_in_mask": [[x, y], ...],
                    "pixel_xy_in_original": [[x, y], ...],
                    "geo_xy": [[x, y], ...]
                  }
                ]
              }
            }
          ]
        }
        ```

        字段说明：
        - `meta.geo.transform`: GeoTIFF 仿射变换 6 参数（等价于 rasterio/Affine 的前 6 项）。
        - `polygons.include/exclude`: 由 `convert_mask_to_polygon()` 从二值 mask 提取出的外轮廓/内洞（若无则为空数组）。
        - `norm_xy`: **mask 空间归一化坐标**，x/y ∈ [0,1]（点对形式：[[x0,y0],[x1,y1],...]）。
        - `pixel_xy_in_mask`: **mask 图像像素坐标**（由 `norm_xy * mask_size` 得到）。
        - `pixel_xy_in_original`: **原图像素坐标**（由 mask 像素按 `sx/sy` 缩放映射到原图）。
        - `geo_xy`: **源 CRS 下的地理坐标**（由原图像素坐标通过 `geo_image.pixel_to_geo()` 转换）。
        """
        # 确定输出目录
        if output_dir is None:
            output_dir = self.directory
        else:
            output_dir = os.path.abspath(output_dir)
            os.makedirs(output_dir, exist_ok=True)

        if len(self.masks) == 0:
            print("ℹ 没有 mask 可导出 json")
            return {"json_path": None, "mask_count": 0}

        json_path = os.path.join(output_dir, f"{self.basename}_masks.json")
        if os.path.exists(json_path) and not overwrite:
            print(f"ℹ 跳过已存在的 masks json: {json_path}")
            return {"json_path": json_path, "mask_count": len(self.masks)}

        # 图像/地理元信息（基于 GeoTIFF 原图）
        orig_w, orig_h = self.geo_image.get_size()
        bounds = self.geo_image.get_bounds()
        crs_obj = self.geo_image.get_crs()
        crs_str = str(crs_obj) if crs_obj is not None else None
        transform = self.geo_image.get_geo_transform()
        transform_6 = list(transform)  # type: ignore[arg-type]

        meta: Dict[str, Any] = {
            "version": 1,
            "source_image_path": self.image_path,
            "basename": self.basename,
            "image_size": {"width": int(orig_w), "height": int(orig_h)},
            "model_scale_size": (
                {"width": int(self.model_scale_image.size[0]), "height": int(self.model_scale_image.size[1])}
                if self.model_scale_image is not None
                else None
            ),
            "geo": {
                "crs": crs_str,
                "transform": transform_6,
                "bounds": {
                    "left": float(bounds.left),
                    "bottom": float(bounds.bottom),
                    "right": float(bounds.right),
                    "top": float(bounds.top),
                },
            },
        }

        def _flat_to_pairs(flat: List[float]) -> List[List[float]]:
            # flat = [x0,y0,x1,y1,...]
            if not flat:
                return []
            if len(flat) % 2 != 0:
                flat = flat[:-1]
            return [[float(flat[i]), float(flat[i + 1])] for i in range(0, len(flat), 2)]

        def _norm_pairs_to_pixel(pairs: List[List[float]], w: int, h: int) -> List[List[float]]:
            return [[float(x) * float(w), float(y) * float(h)] for x, y in pairs]

        def _pixel_pairs_to_geo(pairs_px_in_orig: List[List[float]]) -> List[List[float]]:
            if not pairs_px_in_orig:
                return []
            cols = np.asarray([p[0] for p in pairs_px_in_orig], dtype=np.float64)
            rows = np.asarray([p[1] for p in pairs_px_in_orig], dtype=np.float64)
            xs, ys = self.geo_image.pixel_to_geo(rows, cols)
            xs_arr = np.asarray(xs, dtype=np.float64)
            ys_arr = np.asarray(ys, dtype=np.float64)
            return [[float(x), float(y)] for x, y in zip(xs_arr.tolist(), ys_arr.tolist())]

        def _poly_list(flat_list: List[np.ndarray], mask_w: int, mask_h: int, sx: float, sy: float) -> List[Dict[str, Any]]:
            out_polys: List[Dict[str, Any]] = []
            for poly_flat_np in flat_list:
                poly_flat: List[float] = poly_flat_np.astype(float).tolist()
                pairs_norm = _flat_to_pairs(poly_flat)
                pairs_px_in_mask = _norm_pairs_to_pixel(pairs_norm, mask_w, mask_h)
                pairs_px_in_orig = [[p[0] * sx, p[1] * sy] for p in pairs_px_in_mask]
                pairs_geo = _pixel_pairs_to_geo(pairs_px_in_orig)
                out_polys.append(
                    {
                        "norm_xy": pairs_norm,  # [0,1] in mask space
                        "pixel_xy_in_mask": pairs_px_in_mask,
                        "pixel_xy_in_original": pairs_px_in_orig,
                        "geo_xy": pairs_geo,  # in source CRS
                    }
                )
            return out_polys

        masks_out: List[Dict[str, Any]] = []
        for mask_index in sorted(self.masks.keys()):
            mask_data = self.masks[mask_index]
            mask_image: Image.Image = mask_data["image"]
            prob_value = float(mask_data.get("prob", 0.0))
            tag = mask_data.get("tag", None)

            mask_w, mask_h = mask_image.size
            sx = float(orig_w) / float(max(1, mask_w))
            sy = float(orig_h) / float(max(1, mask_h))

            polygons = convert_mask_to_polygon(mask_image)
            include_list = polygons.get("include", []) or []
            exclude_list = polygons.get("exclude", []) or []

            masks_out.append(
                {
                    "index": int(mask_index),
                    "tag": tag,
                    "prob": prob_value,
                    "mask_size": {"width": int(mask_w), "height": int(mask_h)},
                    "pixel_scale_to_original": {"sx": sx, "sy": sy},
                    "polygons": {
                        "include": _poly_list(include_list, mask_w, mask_h, sx, sy),
                        "exclude": _poly_list(exclude_list, mask_w, mask_h, sx, sy),
                    },
                }
            )

        payload: Dict[str, Any] = {
            "meta": meta,
            "masks": masks_out,
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        print(f"✓ 已保存 masks json: {json_path}")
        return {"json_path": json_path, "mask_count": len(masks_out)}

if __name__ == "__main__":
    mask = Image.open("E:\\sam3_track_seg\\test_images\\clips\\road\\clip_2_road_mask1_prob(0.51).png")
    polygons = convert_mask_to_polygon(mask)
    print(f"include: {len(polygons.get('include', []))} 个, exclude: {len(polygons.get('exclude', []))} 个")
    # 可视化 polygon
    visualize_polygons_on_mask(mask, polygons)