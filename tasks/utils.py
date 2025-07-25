"""
WanGP Tasks 工具模块
提供通用工具函数和辅助功能
"""

import os
import re
import logging
import cv2
import numpy as np
import torch
from typing import Union, Optional
from datetime import datetime


def setup_logging(level: str = "INFO", format_str: Optional[str] = None) -> logging.Logger:
    """设置日志配置"""
    if format_str is None:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_str,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("tasks.log", encoding="utf-8")
        ]
    )
    
    logger = logging.getLogger("wangp.tasks")
    return logger


def sanitize_file_name(name: str) -> str:
    """清理文件名，只保留字母数字和下划线"""
    return re.sub(r"[^a-zA-Z0-9_\-.]", "_", name)


def save_video(frames: Union[torch.Tensor, np.ndarray], output_path: str, fps: int = 24) -> None:
    """
    保存视频到指定路径
    
    Args:
        frames: 视频帧 [T, H, W, C] numpy 或 torch.Tensor
        output_path: 输出路径
        fps: 帧率
    """
    # 转换为 numpy
    if hasattr(frames, "cpu"):
        frames = frames.cpu().numpy()
    
    # 确保数据类型正确
    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)
    
    # 获取视频尺寸
    height, width = frames.shape[1], frames.shape[2]
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 设置编码器
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 写入帧
    for frame in frames:
        # OpenCV 使用 BGR 格式
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(bgr_frame)
    
    out.release()


def get_video_info(path: str) -> Optional[dict]:
    """
    获取视频信息
    
    Returns:
        包含帧数、分辨率、fps等信息的字典，失败时返回None
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    
    info = {
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS)
    }
    
    cap.release()
    return info


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


def generate_output_filename(prompt: str, model_type: str, timestamp: Optional[str] = None) -> str:
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
    reserved = torch.cuda.memory_reserved() / 1024**3   # GB
    
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
