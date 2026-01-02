
from tkinter import N
import rasterio
from rasterio.windows import Window
from rasterio.transform import xy, from_bounds
from rasterio.io import MemoryFile
from rasterio import warp
from rasterio.enums import Resampling
import numpy as np
from PIL import Image
from typing import Optional, Tuple, Union
import math

# 尝试导入PyTorch，用于GPU加速
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# load a very large tiff image to PIL format, and get the geo data
class GeoTiffImage():
    def __init__(self, image_path):
        """
        初始化GeoTiffImage类
        
        Args:
            image_path: GeoTIFF图像文件路径
        """
        self.image_path = image_path
        self.dataset = rasterio.open(image_path, 'r')
        self.memory_file = None  # 用于保存内存文件引用，防止被垃圾回收
        
        # 获取地理信息
        self.transform = self.dataset.transform
        self.crs = self.dataset.crs
        self.bounds = self.dataset.bounds
        self.width = self.dataset.width
        self.height = self.dataset.height
        self.count = self.dataset.count  # 波段数

    def _interpolate_multistage(
        self,
        x: "torch.Tensor",
        target_size_hw: Tuple[int, int],
        *,
        mode: str = "bicubic",
        align_corners: bool = False,
        antialias: bool = True,
        min_step_scale: float = 0.25,
    ) -> "torch.Tensor":
        """
        使用多阶段的 torch.nn.functional.interpolate 来避免极端缩放比时的 CUDA shared memory 限制。

        设计目标：
        - 不改变插值类型：仍然使用 bicubic
        - 下采样时保持 antialias=True 的语义（每个阶段都是温和下采样）
        - 将一次性极端缩放拆分为若干次缩放，降低单次算子内部缓冲区/共享内存压力

        Args:
            x: 4D tensor, shape = (N, C, H, W)
            target_size_hw: (target_h, target_w)
            mode/align_corners/antialias: 透传给 F.interpolate
            min_step_scale: 每一步的最小缩放比例（0<min_step_scale<1）。
                例如 0.25 表示每步最多缩小 4 倍；越大表示每步缩小更少、阶段更多、更稳但更慢。
        """
        # 允许在未安装 torch 的情况下 import 本文件；只有调用到这里才会用 torch
        import torch.nn.functional as F

        if x.dim() != 4:
            raise ValueError(f"interpolate expects 4D tensor (N,C,H,W), got {tuple(x.shape)}")

        target_h, target_w = int(target_size_hw[0]), int(target_size_hw[1])
        if target_h <= 0 or target_w <= 0:
            raise ValueError(f"target_size_hw must be positive, got {(target_h, target_w)}")

        _, _, src_h, src_w = x.shape
        if src_h == target_h and src_w == target_w:
            return x

        # 如果是上采样（或非严格下采样），保持原来的单步行为
        s_h = target_h / float(src_h)
        s_w = target_w / float(src_w)
        if s_h >= 1.0 and s_w >= 1.0:
            return F.interpolate(
                x,
                size=(target_h, target_w),
                mode=mode,
                align_corners=align_corners,
                antialias=False,
            )

        # 数值保护
        if not (0.0 < min_step_scale < 1.0):
            min_step_scale = 0.25

        # 计算需要的阶段数：确保每个维度单步缩放比例都 >= min_step_scale
        def _steps_needed(s: float) -> int:
            if s >= 1.0:
                return 1
            # log(s) / log(min_step_scale) 为正数（都为负），ceil 后为最小满足条件的阶段数
            return max(1, int(math.ceil(math.log(max(s, 1e-12)) / math.log(min_step_scale))))

        n_steps = max(_steps_needed(s_h), _steps_needed(s_w))
        if n_steps <= 1:
            return F.interpolate(
                x,
                size=(target_h, target_w),
                mode=mode,
                align_corners=align_corners,
                antialias=bool(antialias),
            )

        # 生成中间尺寸序列（最后一步精确到 target）
        # 重要：中间尺寸必须位于 [min(src, target), max(src, target)] 区间内。
        # 之前错误地用 target 作为上界，会导致第一步直接“跳到 target”，使多阶段退化为单步。
        low_h, high_h = (target_h, src_h) if target_h <= src_h else (src_h, target_h)
        low_w, high_w = (target_w, src_w) if target_w <= src_w else (src_w, target_w)

        sizes: list[Tuple[int, int]] = []
        for i in range(1, n_steps):
            ih = int(round(src_h * (s_h ** (i / n_steps))))
            iw = int(round(src_w * (s_w ** (i / n_steps))))
            ih = max(low_h, min(ih, high_h))
            iw = max(low_w, min(iw, high_w))
            if sizes and sizes[-1] == (ih, iw):
                continue
            sizes.append((ih, iw))

        if not sizes or sizes[-1] != (target_h, target_w):
            sizes.append((target_h, target_w))

        out = x
        for (ih, iw) in sizes:
            # 仅在“确实下采样”时开启 antialias
            _, _, cur_h, cur_w = out.shape
            step_antialias = bool(antialias) and (ih < cur_h or iw < cur_w)
            out = F.interpolate(
                out,
                size=(ih, iw),
                mode=mode,
                align_corners=align_corners,
                antialias=step_antialias,
            )
        return out
        
    def get_pil_image(self, window: Optional[Window] = None, band_indices: Optional[list] = None) -> Image.Image:
        """
        读取GeoTIFF图像并转换为PIL Image格式
        
        Args:
            window: 可选的rasterio Window对象，用于读取图像的特定区域（窗口读取，适合大图像）
                    如果为None，则读取整个图像
            band_indices: 要读取的波段索引列表（例如[1,2,3]用于RGB），如果为None则读取所有波段
        
        Returns:
            PIL Image对象
        """
        if band_indices is None:
            # 如果未指定波段，根据波段数决定
            if self.count == 1:
                # 单波段灰度图
                band_indices = [1]
            elif self.count >= 3:
                # 多波段，默认取前3个作为RGB
                band_indices = [1, 2, 3]
            else:
                # 2个波段的情况
                band_indices = list(range(1, self.count + 1))
        
        # 读取指定波段
        bands_data = []
        for band_idx in band_indices:
            if window is None:
                band_data = self.dataset.read(band_idx)
            else:
                band_data = self.dataset.read(band_idx, window=window)
            bands_data.append(band_data)
        
        # 堆叠波段
        if len(bands_data) == 1:
            # 单波段灰度图
            img_array = bands_data[0]
        else:
            # 多波段图像，堆叠为 (C, H, W) 格式
            img_array = np.stack(bands_data, axis=0)
            # 转换为 (H, W, C) 格式用于PIL
            img_array = np.transpose(img_array, (1, 2, 0))
        
        # 归一化数据到0-255范围（如果数据不在这个范围）
        if img_array.dtype != np.uint8:
            # 根据数据类型进行归一化
            if img_array.max() > 255:
                # 可能是16位或浮点数据
                img_array = (img_array / img_array.max() * 255).astype(np.uint8)
            else:
                img_array = img_array.astype(np.uint8)
        
        # 转换为PIL Image
        if len(img_array.shape) == 2:
            # 灰度图
            pil_image = Image.fromarray(img_array, mode='L')
        elif img_array.shape[2] == 1:
            # 单通道，转换为灰度
            pil_image = Image.fromarray(img_array[:, :, 0], mode='L')
        elif img_array.shape[2] == 3:
            # RGB图像
            pil_image = Image.fromarray(img_array, mode='RGB')
        elif img_array.shape[2] == 4:
            # RGBA图像
            pil_image = Image.fromarray(img_array, mode='RGBA')
        else:
            # 其他情况，取前3个通道作为RGB
            pil_image = Image.fromarray(img_array[:, :, :3], mode='RGB')
        
        return pil_image
    
    def pixel_to_geo(self, row: Union[int, np.ndarray], col: Union[int, np.ndarray]) -> Tuple:
        """
        将像素坐标转换为地理坐标
        
        Args:
            row: 像素行坐标（y坐标）
            col: 像素列坐标（x坐标）
        
        Returns:
            地理坐标 (x, y) 元组，如果是数组则返回numpy数组
        """
        return xy(self.transform, row, col, offset='center')
    
    def geo_to_pixel(self, x: Union[float, np.ndarray], y: Union[float, np.ndarray]) -> Tuple:
        """
        将地理坐标转换为像素坐标
        
        Args:
            x: 地理坐标x（经度或投影坐标）
            y: 地理坐标y（纬度或投影坐标）
        
        Returns:
            像素坐标 (col, row) 元组，如果是数组则返回numpy数组
        """
        # 使用rasterio的transform进行逆变换
        row, col = rasterio.transform.rowcol(self.transform, x, y, op=rasterio.enums.TransformMethod.centered)
        return col, row
    
    def get_window_from_geo_bounds(self, min_x: float, min_y: float, max_x: float, max_y: float) -> Window:
        """
        根据地理边界创建窗口对象
        
        Args:
            min_x: 最小x坐标（地理坐标）
            min_y: 最小y坐标（地理坐标）
            max_x: 最大x坐标（地理坐标）
            max_y: 最大y坐标（地理坐标）
        
        Returns:
            rasterio Window对象
        """
        # 将地理坐标转换为像素坐标
        row_min, col_min = rasterio.transform.rowcol(self.transform, min_x, max_y)
        row_max, col_max = rasterio.transform.rowcol(self.transform, max_x, min_y)
        
        # 确保坐标在图像范围内
        row_min = max(0, min(row_min, self.height))
        row_max = max(0, min(row_max, self.height))
        col_min = max(0, min(col_min, self.width))
        col_max = max(0, min(col_max, self.width))
        
        # 创建窗口
        window = Window.from_slices((row_min, row_max), (col_min, col_max))
        return window
    
    def get_geo_transform(self) -> rasterio.Affine:
        """
        获取地理变换矩阵
        
        Returns:
            rasterio Affine变换对象
        """
        return self.transform
    
    def get_crs(self):
        """
        获取坐标参考系统(CRS)
        
        Returns:
            CRS对象
        """
        return self.crs
    
    def get_bounds(self) -> rasterio.coords.BoundingBox:
        """
        获取地理边界
        
        Returns:
            BoundingBox对象，包含(left, bottom, right, top)
        """
        return self.bounds
    
    def get_size(self) -> Tuple[int, int]:
        """
        获取图像尺寸
        
        Returns:
            (width, height) 元组
        """
        return (self.width, self.height)
    
    def get_count(self) -> int:
        """
        获取波段数
        
        Returns:
            波段数量
        """
        return self.count
    
    def get_gsd(self) -> Tuple[float, float]:
        """
        计算图像的地面采样距离(GSD)，单位为cm/pixel
        
        GSD表示每个像素在地面上代表的实际距离。
        返回值表示每个像素对应的厘米数。
        
        Returns:
            (gsd_x, gsd_y) 元组，分别表示x方向和y方向的GSD值（cm/pixel）
            如果无法确定单位或转换失败，会抛出异常
        
        Note:
            - 对于投影坐标系（如UTM），通常单位是米，可以直接转换
            - 对于地理坐标系（经纬度），单位是度，需要转换为米
            注：对于地理坐标系，使用图像中心纬度进行转换
        """
        # 从transform获取像素分辨率
        # transform.a 是x方向的像素宽度（地理单位/像素）
        # transform.e 是y方向的像素高度（地理单位/像素，通常是负数）
        pixel_size_x = abs(self.transform.a)  # x方向分辨率
        pixel_size_y = abs(self.transform.e)  # y方向分辨率
        
        # 判断CRS类型
        if self.crs is None:
            raise ValueError("无法确定坐标参考系统(CRS)，无法计算GSD")
        
        # 检查是否是地理坐标系（经纬度）
        is_geographic = self.crs.is_geographic
        
        if is_geographic:
            # 地理坐标系（经纬度），单位是度
            # 需要转换为米：1度经度 ≈ 111000 * cos(纬度) 米，1度纬度 ≈ 111000 米
            # 使用图像中心点的纬度进行转换
            center_lat = (self.bounds.bottom + self.bounds.top) / 2
            center_lat_rad = math.radians(center_lat)
            
            # 1度纬度对应的米数（近似常数）
            meters_per_degree_lat = 111000.0
            # 1度经度对应的米数（随纬度变化）
            meters_per_degree_lon = 111000.0 * math.cos(center_lat_rad)
            
            # 将度转换为米
            pixel_size_x_meters = pixel_size_x * meters_per_degree_lon
            pixel_size_y_meters = pixel_size_y * meters_per_degree_lat
        else:
            # 投影坐标系，通常单位是米
            # 对于某些特殊情况，可能需要检查units，但大多数投影坐标系使用米
            pixel_size_x_meters = pixel_size_x
            pixel_size_y_meters = pixel_size_y
        
        # 计算GSD (cm/pixel)
        # 如果1像素 = pixel_size_x_meters 米
        # 那么1像素 = pixel_size_x_meters * 100 厘米
        # 所以 cm/pixel = pixel_size_x_meters * 100
        gsd_x = pixel_size_x_meters * 100  # cm/pixel
        gsd_y = pixel_size_y_meters * 100  # cm/pixel
        
        return (gsd_x, gsd_y)
    
    def scale_to_max_size(self, max_size: int, window: Optional[Window] = None,
                          band_indices: Optional[list] = None,
                          use_gpu: bool = True,
                          device: Optional[Union[str, torch.device]] = None,
                          gpu_chunk_size: Optional[int] = None) -> Image.Image:
        """
        将图像等比例缩放到指定的最大像素尺寸（最大边长=指定尺寸）
        
        Args:
            max_size: 目标最大边长（像素），图像会被缩放使得 max(width, height) = max_size
            window: 可选的rasterio Window对象，用于读取图像的特定区域
                    如果为None，则读取整个图像
            band_indices: 要读取的波段索引列表（例如[1,2,3]用于RGB），如果为None则读取所有波段
            use_gpu: 如果为True且PyTorch可用，使用GPU加速（推荐，性能最佳）；
                    需要安装PyTorch并具有可用的CUDA GPU
            device: GPU设备，例如'cuda:0'或torch.device('cuda:0')；如果为None且use_gpu=True，自动选择
            gpu_chunk_size: GPU模式下处理大图像时的分块大小（像素），None表示自动选择；
                           对于超大图像，分块可以避免GPU内存不足
        
        Returns:
            缩放后的PIL Image对象
        
        Note:
            - 缩放会保持宽高比
            - 如果原始图像的最大边长小于max_size，图像会被放大
            - 如果原始图像的最大边长大于max_size，图像会被缩小
            - use_gpu=True时使用PyTorch的bicubic插值，GPU加速，性能最佳
        """
        # 确定处理窗口的尺寸
        if window is None:
            src_width = self.width
            src_height = self.height
        else:
            src_width = window.width
            src_height = window.height
        
        # 计算当前最大边长
        current_max_size = max(src_width, src_height)
        
        # 计算缩放比例
        scale_factor = max_size / current_max_size
        
        # 使用GPU加速
        if use_gpu:
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch未安装，无法使用GPU加速。请安装PyTorch: pip install torch")
            if device is None:
                if torch.cuda.is_available():
                    device = torch.device('cuda')
                else:
                    raise RuntimeError("CUDA不可用，无法使用GPU加速")
            elif isinstance(device, str):
                device = torch.device(device)
            return self._scale_by_factor_gpu(
                scale_factor=scale_factor,
                window=window,
                band_indices=band_indices,
                device=device,
                gpu_chunk_size=gpu_chunk_size
            )
        else:
            # CPU模式暂不支持，提示使用GPU模式
            raise NotImplementedError("scale_to_max_size目前仅支持GPU模式。请设置use_gpu=True")
    
    def scale_to_gsd(self, target_gsd: float, window: Optional[Window] = None, 
                     band_indices: Optional[list] = None, 
                     use_average_gsd: bool = True,
                     use_gpu: bool = False,
                     device: Optional[Union[str, torch.device]] = None,
                     gpu_chunk_size: Optional[int] = None) -> Image.Image:
        """
        将图像等比例缩放到指定的GSD值，单位为cm/pixel
        
        Args:
            target_gsd: 目标GSD值（cm/pixel），即每个像素对应的厘米数
            window: 可选的rasterio Window对象，用于读取图像的特定区域
                    如果为None，则读取整个图像
            band_indices: 要读取的波段索引列表（例如[1,2,3]用于RGB），如果为None则读取所有波段
            use_average_gsd: 如果为True，使用x和y方向GSD的平均值；如果为False，使用x方向GSD
            use_gpu: 如果为True且PyTorch可用，使用GPU加速（推荐，性能最佳）；
                    需要安装PyTorch并具有可用的CUDA GPU
            device: GPU设备，例如'cuda:0'或torch.device('cuda:0')；如果为None且use_gpu=True，自动选择
            gpu_chunk_size: GPU模式下处理大图像时的分块大小（像素），None表示自动选择；
                           对于超大图像，分块可以避免GPU内存不足
        
        Returns:
            缩放后的PIL Image对象
        
        Note:
            - 缩放比例 = current_gsd / target_gsd
            - 如果目标GSD大于当前GSD，图像会被缩小
            - 如果目标GSD小于当前GSD，图像会被放大
            - use_gpu=True时使用PyTorch的bicubic插值，GPU加速，性能最佳
            - use_gpu=False时使用rasterio.warp的LANCZOS重采样，CPU单线程
        """
        # 获取当前图像的GSD值
        current_gsd_x, current_gsd_y = self.get_gsd()
        
        # 确定使用的当前GSD值
        if use_average_gsd:
            current_gsd = (current_gsd_x + current_gsd_y) / 2.0
        else:
            current_gsd = current_gsd_x
        
        # 计算缩放比例
        scale_factor = current_gsd / target_gsd
        
        # 优先使用GPU加速
        if use_gpu:
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch未安装，无法使用GPU加速。请安装PyTorch: pip install torch")
            if device is None:
                if torch.cuda.is_available():
                    device = torch.device('cuda')
                else:
                    raise RuntimeError("CUDA不可用，无法使用GPU加速")
            elif isinstance(device, str):
                device = torch.device(device)
            return self._scale_to_gsd_gpu(target_gsd, scale_factor, window, band_indices, 
                                         device, gpu_chunk_size)
        else:
            # 使用rasterio.warp进行CPU重采样（单线程）
            return self._scale_to_gsd_warp(target_gsd, scale_factor, window, band_indices)
    
    def _scale_by_factor_gpu(self, scale_factor: float,
                             window: Optional[Window] = None,
                             band_indices: Optional[list] = None,
                             device: Optional[torch.device] = None,
                             gpu_chunk_size: Optional[int] = None) -> Image.Image:
        """
        使用PyTorch GPU加速进行高效重采样（内部通用方法，基于缩放比例）
        
        优势：
        - GPU并行处理，性能远超CPU
        - 所有波段同时处理，不需要循环
        - 对于大图像，使用分块处理避免内存不足
        - 使用双三次插值（bicubic），质量接近LANCZOS
        
        注意：
        - 需要PyTorch和CUDA支持
        - 自动使用分块处理以避免GPU内存不足
        
        Args:
            scale_factor: 缩放比例，例如 0.5 表示缩小到50%，2.0 表示放大到200%
            window: 可选的rasterio Window对象，用于读取图像的特定区域
            band_indices: 要读取的波段索引列表
            device: GPU设备
            gpu_chunk_size: GPU模式下处理大图像时的分块大小（像素）
        
        Returns:
            缩放后的PIL Image对象
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 清理GPU缓存
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # 确定要处理的波段
        if band_indices is None:
            if self.count == 1:
                band_indices = [1]
            elif self.count >= 3:
                band_indices = [1, 2, 3]
            else:
                band_indices = list(range(1, self.count + 1))
        
        # 确定处理窗口
        if window is None:
            src_width = self.width
            src_height = self.height
        else:
            src_width = window.width
            src_height = window.height
        
        # 计算新尺寸（保持宽高比）
        new_width = int(src_width * scale_factor)
        new_height = int(src_height * scale_factor)
        new_width = max(1, new_width)
        new_height = max(1, new_height)
        
        num_bands = len(band_indices)
        
        # 估算所需内存
        # 输入内存（假设使用uint8读取，然后转换为float32）
        input_memory_mb = (num_bands * src_height * src_width * 4) / (1024 * 1024)  # float32 = 4 bytes
        output_memory_mb = (num_bands * new_height * new_width * 4) / (1024 * 1024)
        
        # 检查可用GPU内存
        if device.type == 'cuda':
            # 获取可用GPU内存（MB）
            total_memory_mb = torch.cuda.get_device_properties(device.index if device.index is not None else 0).total_memory / (1024 * 1024)
            reserved_memory_mb = torch.cuda.memory_reserved(device.index if device.index is not None else 0) / (1024 * 1024)
            free_memory_mb = total_memory_mb - reserved_memory_mb
            
            # 保守估计：只使用可用内存的80%，确保有足够空间
            available_memory_mb = free_memory_mb * 0.8
        else:
            available_memory_mb = float('inf')
        
        # 确定分块大小
        # 对于大图像，强制使用分块处理以避免内存问题
        if gpu_chunk_size is not None:
            chunk_size = gpu_chunk_size
        elif input_memory_mb > 500 or (device.type == 'cuda' and input_memory_mb + output_memory_mb > available_memory_mb):
            # 自动计算合适的分块大小
            # 每个块的内存 = num_bands * chunk_size^2 * 4 (float32) * 3 (输入+输出+临时缓冲区)
            if device.type == 'cuda' and available_memory_mb < float('inf'):
                max_chunk_memory_mb = available_memory_mb
            else:
                # CPU模式或无法获取GPU内存信息时，使用保守值
                max_chunk_memory_mb = 500  # 500MB
            max_chunk_pixels = int((max_chunk_memory_mb * 1024 * 1024) / (num_bands * 4 * 3))  # 输入+输出+临时
            max_chunk_size = int(np.sqrt(max_chunk_pixels))
            # 限制在合理范围内
            chunk_size = min(max_chunk_size, src_height, src_width, 8192)
            chunk_size = max(chunk_size, 256)  # 最小256像素
        else:
            chunk_size = None
        
        # 如果图像较小且内存充足，一次性处理
        if chunk_size is None or (src_height <= chunk_size and src_width <= chunk_size):
            # 一次性读取和处理
            src_data_list = []
            for band_idx in band_indices:
                if window is None:
                    band_data = self.dataset.read(band_idx)
                else:
                    band_data = self.dataset.read(band_idx, window=window)
                
                if band_data.dtype != np.uint8:
                    if band_data.max() > 255:
                        band_data = (band_data / band_data.max() * 255).astype(np.uint8)
                    else:
                        band_data = band_data.astype(np.uint8)
                src_data_list.append(band_data)
            
            src_array = np.stack(src_data_list, axis=0).astype(np.float32)
            src_tensor = torch.from_numpy(src_array).to(device).unsqueeze(0)
            
            try:
                with torch.no_grad():
                    # 对于极端下采样比例（例如 0.01 级别），单步 bicubic+antialias
                    # 可能触发 CUDA shared memory 限制；这里改用多阶段下采样。
                    if scale_factor < 1.0:
                        # 先按“每步最多缩小 4 倍”尝试；如果仍触发 shared memory，再自动加密步骤
                        retry_min_step_scales = (0.25, 0.5, 0.75)
                        last_err: Optional[RuntimeError] = None
                        for mss in retry_min_step_scales:
                            try:
                                dst_tensor = self._interpolate_multistage(
                                    src_tensor,
                                    (new_height, new_width),
                                    mode="bicubic",
                                    align_corners=False,
                                    antialias=True,
                                    min_step_scale=mss,
                                )
                                last_err = None
                                break
                            except RuntimeError as e:
                                # 仅对 interpolate 的 shared-memory/算法限制报错进行重试
                                msg = str(e)
                                if (
                                    "Provided interpolation parameters can not be handled" in msg
                                    or "Too much shared memory required" in msg
                                ):
                                    last_err = e
                                    continue
                                raise
                        if last_err is not None:
                            raise last_err
                    else:
                        dst_tensor = torch.nn.functional.interpolate(
                            src_tensor,
                            size=(new_height, new_width),
                            mode='bicubic',
                            align_corners=False,
                            antialias=False
                        )
                dst_array = dst_tensor.squeeze(0).cpu().numpy()
                del src_tensor, dst_tensor
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            except RuntimeError as e:
                if "out of memory" in str(e):
                    # 如果还是OOM，强制使用分块处理
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    chunk_size = min(1024, src_height, src_width)
                    dst_array = self._scale_to_gsd_gpu_chunked_from_disk(
                        window, band_indices, src_width, src_height,
                        new_width, new_height, device, chunk_size, scale_factor < 1.0
                    )
                else:
                    raise
        else:
            # 使用分块处理，直接从磁盘读取
            dst_array = self._scale_to_gsd_gpu_chunked_from_disk(
                window, band_indices, src_width, src_height,
                new_width, new_height, device, chunk_size, scale_factor < 1.0
            )
        
        # 裁剪到[0, 255]范围并转换类型
        dst_array = np.clip(dst_array, 0, 255).astype(np.uint8)
        
        # 转换为PIL Image格式
        if num_bands == 1:
            img_array = dst_array[0]
            pil_image = Image.fromarray(img_array, mode='L')
        elif num_bands == 3:
            img_array = np.transpose(dst_array, (1, 2, 0))
            pil_image = Image.fromarray(img_array, mode='RGB')
        elif num_bands == 4:
            img_array = np.transpose(dst_array, (1, 2, 0))
            pil_image = Image.fromarray(img_array, mode='RGBA')
        else:
            img_array = np.transpose(dst_array[:3], (1, 2, 0))
            pil_image = Image.fromarray(img_array, mode='RGB')
        
        return pil_image
    
    def _scale_to_gsd_gpu(self, target_gsd: float, scale_factor: float,
                          window: Optional[Window] = None,
                          band_indices: Optional[list] = None,
                          device: Optional[torch.device] = None,
                          gpu_chunk_size: Optional[int] = None) -> Image.Image:
        """
        使用PyTorch GPU加速进行高效重采样（内部方法，基于GSD）
        
        此方法现在调用通用的 _scale_by_factor_gpu 方法
        """
        return self._scale_by_factor_gpu(
            scale_factor=scale_factor,
            window=window,
            band_indices=band_indices,
            device=device,
            gpu_chunk_size=gpu_chunk_size
        )
    
    def _scale_to_gsd_gpu_chunked_from_disk(self, window: Optional[Window],
                                             band_indices: list,
                                             src_width: int, src_height: int,
                                             new_width: int, new_height: int,
                                             device: torch.device,
                                             chunk_size: int,
                                             use_antialias: bool) -> np.ndarray:
        """
        从磁盘分块读取并处理大图像的GPU缩放（内部方法）
        
        这个方法直接从磁盘分块读取数据，避免一次性加载整个图像到内存或GPU
        """
        num_bands = len(band_indices)
        
        # 在CPU上创建输出数组
        output_array = np.zeros((num_bands, new_height, new_width), dtype=np.float32)
        
        # 计算每个块的大小
        h_chunk_size = min(chunk_size, src_height)
        w_chunk_size = min(chunk_size, src_width)
        
        # 分块处理
        for h_start in range(0, src_height, h_chunk_size):
            h_end = min(h_start + h_chunk_size, src_height)
            
            # 计算输出位置
            h_dst_start = int(h_start * new_height / src_height)
            if h_end >= src_height:
                h_dst_end = new_height
            else:
                h_dst_end = int(h_end * new_height / src_height)
            
            for w_start in range(0, src_width, w_chunk_size):
                w_end = min(w_start + w_chunk_size, src_width)
                
                # 计算输出位置
                w_dst_start = int(w_start * new_width / src_width)
                if w_end >= src_width:
                    w_dst_end = new_width
                else:
                    w_dst_end = int(w_end * new_width / src_width)
                
                # 创建窗口读取当前块
                if window is None:
                    # 直接使用chunk坐标
                    chunk_window = Window(w_start, h_start, w_end - w_start, h_end - h_start)
                else:
                    # 需要将chunk坐标转换为原始图像的坐标
                    chunk_window = Window(
                        window.col_off + w_start,
                        window.row_off + h_start,
                        w_end - w_start,
                        h_end - h_start
                    )
                
                # 读取当前块的所有波段
                chunk_data_list = []
                for band_idx in band_indices:
                    band_chunk = self.dataset.read(band_idx, window=chunk_window)
                    
                    # 归一化到uint8
                    if band_chunk.dtype != np.uint8:
                        if band_chunk.max() > 255:
                            band_chunk = (band_chunk / band_chunk.max() * 255).astype(np.uint8)
                        else:
                            band_chunk = band_chunk.astype(np.uint8)
                    chunk_data_list.append(band_chunk)
                
                # 堆叠为 (C, H, W) 格式
                chunk_array = np.stack(chunk_data_list, axis=0).astype(np.float32)
                
                # 转换到GPU
                chunk_tensor = torch.from_numpy(chunk_array).to(device).unsqueeze(0)
                
                # 计算输出块尺寸
                dst_chunk_height = h_dst_end - h_dst_start
                dst_chunk_width = w_dst_end - w_dst_start
                
                # GPU缩放
                with torch.no_grad():
                    if use_antialias:
                        # 多阶段下采样以避免极端缩放触发 CUDA shared memory 限制
                        retry_min_step_scales = (0.25, 0.5, 0.75)
                        last_err: Optional[RuntimeError] = None
                        for mss in retry_min_step_scales:
                            try:
                                scaled_chunk = self._interpolate_multistage(
                                    chunk_tensor,
                                    (dst_chunk_height, dst_chunk_width),
                                    mode="bicubic",
                                    align_corners=False,
                                    antialias=True,
                                    min_step_scale=mss,
                                )
                                last_err = None
                                break
                            except RuntimeError as e:
                                msg = str(e)
                                if (
                                    "Provided interpolation parameters can not be handled" in msg
                                    or "Too much shared memory required" in msg
                                ):
                                    last_err = e
                                    continue
                                raise
                        if last_err is not None:
                            raise last_err
                    else:
                        scaled_chunk = torch.nn.functional.interpolate(
                            chunk_tensor,
                            size=(dst_chunk_height, dst_chunk_width),
                            mode='bicubic',
                            align_corners=False,
                            antialias=False
                        )
                
                # 转回CPU并写入输出数组
                output_array[:, h_dst_start:h_dst_end, w_dst_start:w_dst_end] = \
                    scaled_chunk.squeeze(0).cpu().numpy()
                
                # 清理GPU内存
                del chunk_tensor, scaled_chunk
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        return output_array
    
    
    def _scale_to_gsd_warp(self, target_gsd: float, scale_factor: float,
                           window: Optional[Window] = None,
                           band_indices: Optional[list] = None) -> Image.Image:
        """
        使用rasterio.warp进行CPU重采样（内部方法，单线程）
        
        优势：
        - 内存高效（分块处理，不一次性加载整个图像）
        - 处理大图像时性能稳定
        """
        # 确定要处理的波段
        if band_indices is None:
            if self.count == 1:
                band_indices = [1]
            elif self.count >= 3:
                band_indices = [1, 2, 3]
            else:
                band_indices = list(range(1, self.count + 1))
        
        # 确定处理窗口
        if window is None:
            src_width = self.width
            src_height = self.height
            src_transform = self.transform
            src_bounds = self.bounds
        else:
            src_width = window.width
            src_height = window.height
            # 计算窗口的transform
            src_transform = rasterio.windows.transform(window, self.transform)
            # 计算窗口的bounds
            left = src_transform.c
            top = src_transform.f
            right = left + src_width * abs(src_transform.a)
            bottom = top + src_height * src_transform.e
            src_bounds = rasterio.coords.BoundingBox(
                min(left, right), min(bottom, top),
                max(left, right), max(bottom, top)
            )
        
        # 计算新尺寸（保持宽高比）
        new_width = int(src_width * scale_factor)
        new_height = int(src_height * scale_factor)
        new_width = max(1, new_width)
        new_height = max(1, new_height)
        
        # 计算新的transform（保持bounds不变，调整分辨率）
        new_transform = from_bounds(
            src_bounds.left, src_bounds.bottom,
            src_bounds.right, src_bounds.top,
            new_width, new_height
        )
        
        # 设置重采样方法（LANCZOS对应GDAL的lanczos）
        resampling_method = Resampling.lanczos
        
        # 读取源数据并进行重采样
        # 使用warp.reproject可以分块处理，内存效率高
        num_bands = len(band_indices)
        output_data = np.zeros((num_bands, new_height, new_width), dtype=np.uint8)
        
        # 对每个波段进行重采样（单线程，顺序处理）
        for i, band_idx in enumerate(band_indices):
            # 读取源波段数据
            if window is None:
                src_data = self.dataset.read(band_idx)
            else:
                src_data = self.dataset.read(band_idx, window=window)
            
            # 如果数据不是uint8，需要归一化
            if src_data.dtype != np.uint8:
                if src_data.max() > 255:
                    src_data = (src_data / src_data.max() * 255).astype(np.uint8)
                else:
                    src_data = src_data.astype(np.uint8)
            
            # 使用warp.reproject进行重采样
            warp.reproject(
                source=src_data,
                destination=output_data[i],
                src_transform=src_transform,
                src_crs=self.crs,
                dst_transform=new_transform,
                dst_crs=self.crs,
                resampling=resampling_method
            )
        
        # 转换为PIL Image格式
        if num_bands == 1:
            # 灰度图
            img_array = output_data[0]
            pil_image = Image.fromarray(img_array, mode='L')
        elif num_bands == 3:
            # RGB图像
            img_array = np.transpose(output_data, (1, 2, 0))  # (C, H, W) -> (H, W, C)
            pil_image = Image.fromarray(img_array, mode='RGB')
        elif num_bands == 4:
            # RGBA图像
            img_array = np.transpose(output_data, (1, 2, 0))
            pil_image = Image.fromarray(img_array, mode='RGBA')
        else:
            # 其他情况，取前3个通道作为RGB
            img_array = np.transpose(output_data[:3], (1, 2, 0))
            pil_image = Image.fromarray(img_array, mode='RGB')
        
        return pil_image
    
    
    def update_dataset_from_image(self, pil_image: Image.Image, preserve_bounds: bool = True):
        """
        将转换后的PIL图像替换为类内部的dataset对象
        
        Args:
            pil_image: PIL Image对象，要替换为新的dataset
            preserve_bounds: 如果为True，保持原始的地理边界（bounds）；如果为False，保持原始transform的像素分辨率
        
        Note:
            - 如果preserve_bounds=True：保持原始bounds，调整transform以适应新尺寸
            - 如果preserve_bounds=False：保持原始transform的像素分辨率，bounds会改变
            - 关闭并替换原有的dataset，更新所有相关属性（width, height, transform, bounds等）
        """
        # 将PIL Image转换为numpy数组
        img_array = np.array(pil_image)
        
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
        
        # 获取新图像的尺寸
        new_height, new_width = img_array.shape[1], img_array.shape[2]
        
        # 保存原始属性
        old_width = self.width
        old_height = self.height
        old_transform = self.transform
        old_bounds = self.bounds
        old_crs = self.crs
        
        # 计算新的transform
        if preserve_bounds:
            # 保持原始bounds，调整transform
            # 从原始bounds创建新的transform
            new_transform = from_bounds(
                old_bounds.left,
                old_bounds.bottom,
                old_bounds.right,
                old_bounds.top,
                new_width,
                new_height
            )
            new_bounds = old_bounds
        else:
            # 保持原始transform的像素分辨率，bounds会改变
            # transform的像素分辨率（a和e）保持不变
            new_transform = old_transform
            
            # 根据新的transform和图像尺寸计算新的bounds
            # 使用rasterio的方法计算bounds
            left = new_transform.c
            top = new_transform.f
            right = left + new_width * abs(new_transform.a)
            # e通常是负数，所以bottom = top + height * e
            bottom = top + new_height * new_transform.e
            
            # 创建新的bounds（确保顺序正确：left, bottom, right, top）
            new_bounds = rasterio.coords.BoundingBox(
                min(left, right),  # left
                min(bottom, top),  # bottom
                max(left, right),  # right
                max(bottom, top)   # top
            )
        
        # 关闭旧的dataset和memory_file
        if hasattr(self, 'dataset') and self.dataset is not None:
            try:
                self.dataset.close()
            except Exception:
                pass
        
        if hasattr(self, 'memory_file') and self.memory_file is not None:
            try:
                self.memory_file.close()
            except Exception:
                pass
        
        # 创建内存中的GeoTIFF
        memfile = MemoryFile()
        
        # 确定数据类型
        if img_array.dtype == np.uint8:
            dtype = rasterio.uint8
        elif img_array.dtype == np.uint16:
            dtype = rasterio.uint16
        elif img_array.dtype == np.int16:
            dtype = rasterio.int16
        elif img_array.dtype == np.float32:
            dtype = rasterio.float32
        elif img_array.dtype == np.float64:
            dtype = rasterio.float64
        else:
            dtype = rasterio.uint8  # 默认
        
        # 创建新的dataset
        new_dataset = memfile.open(
            driver='GTiff',
            height=new_height,
            width=new_width,
            count=num_bands,
            dtype=dtype,
            crs=old_crs,
            transform=new_transform,
            compress='lzw'  # 使用压缩以减少内存占用
        )
        
        # 写入数据
        new_dataset.write(img_array)
        
        # 更新所有属性（保存memory_file引用以防止被垃圾回收）
        self.memory_file = memfile
        self.dataset = new_dataset
        self.transform = new_transform
        self.crs = old_crs
        self.bounds = new_bounds
        self.width = new_width
        self.height = new_height
        self.count = num_bands
    
    def save_to_geotiff(self, output_path: str, compress: str = 'lzw', 
                        tiled: bool = False, blockxsize: int = 256, 
                        blockysize: int = 256, **kwargs):
        """
        将当前的dataset保存为带地理信息的GeoTIFF文件
        
        Args:
            output_path: 输出GeoTIFF文件路径
            compress: 压缩方式，可选值：'none', 'lzw', 'deflate', 'jpeg'等，默认为'lzw'
            tiled: 是否使用分块存储（tiled），默认为False（使用strip存储）
            blockxsize: 分块的x方向大小（仅在tiled=True时有效），默认为256
            blockysize: 分块的y方向大小（仅在tiled=True时有效），默认为256
            **kwargs: 其他传递给rasterio.open的参数
        
        Note:
            - 保存所有地理信息（transform, crs, bounds等）
            - 保存所有波段的数据
            - 保持原始数据类型
        """
        # 读取所有波段的数据
        data = self.dataset.read()
        
        # 确定数据类型（所有波段应该使用相同的数据类型）
        dtype = self.dataset.dtypes[0]  # 使用第一个波段的数据类型
        
        # 准备写入参数
        write_kwargs = {
            'driver': 'GTiff',
            'height': self.height,
            'width': self.width,
            'count': self.count,
            'dtype': dtype,
            'crs': self.crs,
            'transform': self.transform,
            'compress': compress,
        }
        
        # 如果使用分块存储
        if tiled:
            write_kwargs['tiled'] = True
            write_kwargs['blockxsize'] = blockxsize
            write_kwargs['blockysize'] = blockysize
        
        # 复制NoData值（如果有）
        if self.dataset.nodata is not None:
            write_kwargs['nodata'] = self.dataset.nodata
        
        # 添加其他自定义参数（可能会覆盖上面的设置）
        write_kwargs.update(kwargs)
        
        # 写入文件
        with rasterio.open(output_path, 'w', **write_kwargs) as dst:
            dst.write(data)
            
            # 复制所有标签和描述信息
            try:
                if hasattr(self.dataset, 'tags') and self.dataset.tags():
                    dst.update_tags(**self.dataset.tags())
            except Exception:
                pass  # 忽略标签复制错误
            
            # 复制每个波段的描述信息
            for i in range(1, self.count + 1):
                try:
                    desc = self.dataset.descriptions[i - 1]
                    if desc:
                        dst.set_band_description(i, desc)
                except Exception:
                    pass  # 忽略描述信息复制错误
        
        # 更新image_path属性（可选，但这样对象就知道当前文件路径）
        self.image_path = output_path
    
    def close(self):
        """
        关闭rasterio数据集，释放资源
        """
        if hasattr(self, 'dataset') and self.dataset is not None:
            try:
                self.dataset.close()
            except Exception:
                pass  # 忽略关闭时的错误
        
        if hasattr(self, 'memory_file') and self.memory_file is not None:
            try:
                self.memory_file.close()
            except Exception:
                pass  # 忽略关闭时的错误
    
    def __enter__(self):
        """支持上下文管理器"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """支持上下文管理器"""
        self.close()
    
    def __del__(self):
        """析构函数，确保资源被释放"""
        self.close()


#test the class
if __name__ == "__main__":
    print("=" * 70)
    print("GeoTiffImage 类测试")
    print("=" * 70)
    
    image_path = "E:\\sam3_track_seg\\test_images\\result.tif"
    print(f"\n📁 加载图像: {image_path}")
    geo_tiff_image = GeoTiffImage(image_path)
    
    print("\n" + "-" * 70)
    print("📊 图像基本信息")
    print("-" * 70)
    # image = geo_tiff_image.get_pil_image()
    width, height = geo_tiff_image.get_size()
    print(f"  • 图像尺寸 (PIL): {width} × {height} 像素")
    size = geo_tiff_image.get_size()
    print(f"  • 图像尺寸 (GeoTIFF): {size[0]} × {size[1]} 像素")
    print(f"  • 波段数量: {geo_tiff_image.get_count()}")
    
    print("\n" + "-" * 70)
    print("🌍 地理信息")
    print("-" * 70)
    transform = geo_tiff_image.get_geo_transform()
    print(f"  • 地理变换矩阵:")
    print(f"    {transform}")
    crs = geo_tiff_image.get_crs()
    print(f"  • 坐标参考系统 (CRS): {crs}")
    bounds = geo_tiff_image.get_bounds()
    print(f"  • 地理边界:")
    print(f"    - 左边界 (left):   {bounds.left:.6f}")
    print(f"    - 下边界 (bottom): {bounds.bottom:.6f}")
    print(f"    - 右边界 (right):  {bounds.right:.6f}")
    print(f"    - 上边界 (top):    {bounds.top:.6f}")
    
    print("\n" + "-" * 70)
    print("📏 地面采样距离 (GSD)")
    print("-" * 70)
    gsd_x, gsd_y = geo_tiff_image.get_gsd()
    print(f"  • X方向 GSD: {gsd_x:.4f} cm/pixel")
    print(f"  • Y方向 GSD: {gsd_y:.4f} cm/pixel")
    print(f"  • 平均 GSD:  {(gsd_x + gsd_y) / 2:.4f} cm/pixel")
    
    # print("\n" + "-" * 70)
    # print("🔄 图像缩放操作 to gsd")
    # print("-" * 70)
    # target_gsd = 4.0
    # print(f"  • 目标 GSD: {target_gsd} cm/pixel")
    # print(f"  • 执行缩放...")
    
    # # 使用GPU加速（推荐，性能最佳）
    # # 如果GPU不可用，会自动回退到CPU模式
    # new_image = None
    # try:
    #     new_image = geo_tiff_image.scale_to_gsd(target_gsd, use_gpu=True, device='cuda:0')
    #     print(f"  • 使用GPU加速")
    # except (ImportError, RuntimeError) as e:
    #     print(f"  • GPU不可用，使用CPU模式: {e}")
    #     # 使用CPU模式（单线程warp）
    #     new_image = geo_tiff_image.scale_to_gsd(target_gsd, use_gpu=False)
    
    # new_width, new_height = new_image.size
    # print(f"  • 缩放后尺寸: {new_width} × {new_height} 像素")

    #test scale to max size
    print("\n" + "-" * 70)
    print("🔄 图像缩放操作 to max size")
    print("-" * 70)
    target_max_size = 1008
    print(f"  • 目标最大尺寸: {target_max_size} 像素")
    print(f"  • 执行缩放...")
    new_image = geo_tiff_image.scale_to_max_size(1008, use_gpu=True, device='cuda:0')
    new_width, new_height = new_image.size
    print(f"  • 缩放后尺寸: {new_width} × {new_height} 像素")

    
    print("\n" + "-" * 70)
    print("💾 保存处理结果")
    print("-" * 70)
    output_path = "E:\\sam3_track_seg\\test_images\\result_scaled_1008.tif"
    print(f"  • 更新数据集...")
    geo_tiff_image.update_dataset_from_image(new_image)
    print(f"  • 保存到: {output_path}")
    geo_tiff_image.save_to_geotiff(output_path)
    print(f"  ✅ 保存完成!")
    
    print("\n" + "-" * 70)
    print("🔚 清理资源")
    print("-" * 70)
    geo_tiff_image.close()
    print("  ✅ 资源已释放")
    
    print("\n" + "=" * 70)
    print("测试完成!")
    print("=" * 70)