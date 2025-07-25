"""
WanGP Tasks 工具模块
提供通用工具函数和辅助功能
"""

import os
import re
import logging
import numpy as np
import torch
from typing import Union, Optional
from datetime import datetime


def setup_logging(
    level: str = "INFO", format_str: Optional[str] = None
) -> logging.Logger:
    """设置日志配置"""
    if format_str is None:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_str,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("tasks.log", encoding="utf-8"),
        ],
    )

    logger = logging.getLogger("wangp.tasks")
    return logger


def sanitize_file_name(name: str) -> str:
    """清理文件名，只保留字母数字和下划线"""
    return re.sub(r"[^a-zA-Z0-9_\-.]", "_", name)


def save_video(
    frames: Union[torch.Tensor, np.ndarray], output_path: str, fps: int = 24
) -> None:
    """
    保存视频到指定路径，使用 ffmpegcv 实现

    Args:
        frames: 视频帧 torch.Tensor，格式应为 [C, F, H, W]，数据范围 [-1, 1]
        output_path: 输出路径
        fps: 帧率
    """
    try:
        import ffmpegcv
    except ImportError:
        print("ffmpegcv 未安装，尝试使用 imageio 作为备选方案")
        return _save_video_imageio(frames, output_path, fps)

    # 确保输入是 torch.Tensor
    if not isinstance(frames, torch.Tensor):
        frames = torch.tensor(frames)

    if frames.device != torch.device("cpu"):
        frames = frames.cpu()

    print(f"输入张量形状: {frames.shape}")
    print(f"输入张量数据范围: [{frames.min():.3f}, {frames.max():.3f}]")

    # 处理张量格式和归一化
    # 1. 限制数据范围到 [-1, 1]
    frames = frames.clamp(-1, 1)

    # 2. 处理张量格式
    if len(frames.shape) == 4:  # 假设是 [C, F, H, W] 格式
        print(f"处理4D张量: {frames.shape}")
        # 转换为 [F, C, H, W] 然后逐帧处理
        frames = frames.permute(1, 0, 2, 3)  # [C, F, H, W] -> [F, C, H, W]
        print(f"维度转换后: {frames.shape}")

        # 逐帧归一化并转换为 [F, H, W, C]
        processed_frames = []
        for i in range(frames.shape[0]):
            frame = frames[i]  # [C, H, W]
            # 归一化 [-1, 1] -> [0, 255]
            frame = ((frame + 1.0) * 127.5).clamp(0, 255)
            # 转换为 [H, W, C]
            frame = frame.permute(1, 2, 0)
            processed_frames.append(frame)

        processed_frames = torch.stack(processed_frames, dim=0)  # [F, H, W, C]
        print(f"处理后张量形状: {processed_frames.shape}")
    else:
        print(f"处理非4D张量: {frames.shape}")
        processed_frames = frames
        if len(processed_frames.shape) == 4 and processed_frames.shape[0] == 3:
            # 可能是 [C, H, W, F] 需要转换
            processed_frames = processed_frames.permute(3, 1, 2, 0)  # [F, H, W, C]
            print(f"维度转换后: {processed_frames.shape}")
        # 归一化 [-1, 1] -> [0, 255]
        processed_frames = ((processed_frames + 1.0) * 127.5).clamp(0, 255)

    # 3. 转换为 uint8 numpy 数组
    frames_numpy = processed_frames.type(torch.uint8).numpy()
    print(f"最终数组形状: {frames_numpy.shape}, dtype: {frames_numpy.dtype}")
    print(f"最终数据范围: [{frames_numpy.min()}, {frames_numpy.max()}]")

    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 4. 使用 ffmpegcv 保存视频
    try:
        with ffmpegcv.VideoWriter(output_path, None, fps) as out:
            for frame in frames_numpy:
                out.write(frame)

        print(f"视频保存成功: {output_path}")

    except Exception as e:
        print(f"ffmpegcv 保存失败: {e}")
        print("尝试使用 imageio 作为备选方案...")
        _save_video_imageio(frames, output_path, fps)


def _save_video_imageio(
    frames: Union[torch.Tensor, np.ndarray], output_path: str, fps: int = 24
) -> None:
    """
    使用 imageio 保存视频的备选方案
    """
    import imageio

    # 确保输入是 torch.Tensor
    if not isinstance(frames, torch.Tensor):
        frames = torch.tensor(frames)

    if frames.device != torch.device("cpu"):
        frames = frames.cpu()

    # 处理张量格式和归一化
    frames = frames.clamp(-1, 1)

    if len(frames.shape) == 4:  # [C, F, H, W] 格式
        frames = frames.permute(1, 0, 2, 3)  # [C, F, H, W] -> [F, C, H, W]

        processed_frames = []
        for i in range(frames.shape[0]):
            frame = frames[i]  # [C, H, W]
            frame = (frame + 1.0) / 2.0  # [-1, 1] -> [0, 1]
            frame = frame.permute(1, 2, 0)  # [C, H, W] -> [H, W, C]
            processed_frames.append(frame)

        processed_frames = torch.stack(processed_frames, dim=0)
    else:
        processed_frames = frames
        if len(processed_frames.shape) == 4 and processed_frames.shape[0] == 3:
            processed_frames = processed_frames.permute(3, 1, 2, 0)
        processed_frames = (processed_frames + 1.0) / 2.0

    # 转换为 uint8
    processed_frames = (processed_frames * 255).type(torch.uint8)
    frames_numpy = processed_frames.numpy()

    # 使用 imageio 保存
    try:
        writer = imageio.get_writer(output_path, fps=fps, codec="libx264")
        for frame in frames_numpy:
            frame = np.ascontiguousarray(frame)
            writer.append_data(frame)
        writer.close()
        print(f"使用 imageio 保存成功: {output_path}")
    except Exception as e:
        print(f"imageio 也保存失败: {e}")
        raise


def get_video_info(path: str) -> Optional[dict]:
    """
    获取视频信息

    Returns:
        包含帧数、分辨率、fps等信息的字典，失败时返回None
    """
    try:
        import imageio

        reader = imageio.get_reader(path)
        meta = reader.get_meta_data()

        # 获取视频信息
        frame_count = 0
        try:
            frame_count = len(reader)
        except Exception:
            # 如果无法直接获取长度，设为0
            frame_count = 0

        info = {
            "frames": frame_count,
            "width": meta.get("size", [0, 0])[0] if meta.get("size") else 0,
            "height": meta.get("size", [0, 0])[1] if meta.get("size") else 0,
            "fps": meta.get("fps", 30),
        }

        reader.close()
        return info
    except Exception as e:
        print(f"获取视频信息失败: {e}")
        return None


def parse_resolution(resolution: str) -> tuple[int, int]:
    """
    解析分辨率字符串

    Args:
        resolution: 格式如 "720x480"

    Returns:
        (width, height) 元组
    """
    if "x" not in resolution:
        raise ValueError(f"Invalid resolution format: {resolution}")

    width_str, height_str = resolution.split("x")
    width, height = int(width_str), int(height_str)

    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid resolution dimensions: {width}x{height}")

    return width, height


def generate_output_filename(
    prompt: str, model_type: str, timestamp: Optional[str] = None
) -> str:
    """
    生成输出文件名

    Args:
        prompt: 提示词
        model_type: 模型类型
        timestamp: 时间戳，如果为None则自动生成

    Returns:
        安全的文件名（不包含扩展名）
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    safe_prompt = sanitize_file_name(prompt[:50])
    safe_model = sanitize_file_name(model_type)

    return f"{safe_prompt}_{safe_model}_{timestamp}"


def ensure_directory_exists(path: str) -> None:
    """确保目录存在，如果不存在则创建"""
    os.makedirs(path, exist_ok=True)


def get_available_devices() -> list[str]:
    """获取可用的计算设备"""
    devices = ["cpu"]

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            devices.append(f"cuda:{i}")
        devices.append("cuda")  # 默认cuda设备

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append("mps")

    return devices


def format_memory_usage() -> str:
    """格式化内存使用情况"""
    if not torch.cuda.is_available():
        return "CUDA not available"

    allocated = torch.cuda.memory_allocated() / 1024**3  # GB
    reserved = torch.cuda.memory_reserved() / 1024**3  # GB

    return f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"


def cleanup_gpu_memory() -> None:
    """清理GPU内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class ProgressTracker:
    """进度跟踪器"""

    def __init__(self, total_steps: int, logger: Optional[logging.Logger] = None):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = None
        self.logger = logger or logging.getLogger(__name__)

    def start(self):
        """开始跟踪"""
        self.start_time = datetime.now()
        self.logger.info(f"开始处理，总步数: {self.total_steps}")

    def update(self, step: int, message: str = ""):
        """更新进度"""
        self.current_step = step
        progress = (step / self.total_steps) * 100

        elapsed = datetime.now() - self.start_time if self.start_time else None
        eta_str = ""

        if elapsed and step > 0:
            eta_seconds = (elapsed.total_seconds() / step) * (self.total_steps - step)
            eta_str = f" - ETA: {eta_seconds:.0f}s"

        log_msg = f"步骤 {step}/{self.total_steps} ({progress:.1f}%){eta_str}"
        if message:
            log_msg += f" - {message}"

        self.logger.info(log_msg)

    def finish(self, message: str = "完成"):
        """完成跟踪"""
        if self.start_time:
            elapsed = datetime.now() - self.start_time
            self.logger.info(f"{message} - 总耗时: {elapsed.total_seconds():.2f}s")
