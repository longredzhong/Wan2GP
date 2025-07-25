"""
WanGP Tasks Package
独立的任务执行系统，支持各种 AI 视频生成模型
"""

from .model_loader import ModelLoader, ModelFactory
from .config_manager import ConfigManager, TaskConfig
from .task_runner import CLITaskRunner
from .utils import setup_logging, save_video, sanitize_file_name

__version__ = "1.0.0"
__all__ = [
    "ModelLoader", 
    "ModelFactory",
    "ConfigManager",
    "TaskConfig",
    "CLITaskRunner",
    "setup_logging",
    "save_video",
    "sanitize_file_name",
]
