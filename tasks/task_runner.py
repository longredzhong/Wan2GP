"""
WanGP CLI 任务运行器模块
"""

import os
import time
import torch
import logging
from typing import List, Any, Optional

from .config_manager import TaskConfig, ConfigManager
from .model_loader import ModelFactory
from .utils import save_video, ProgressTracker, cleanup_gpu_memory, format_memory_usage

logger = logging.getLogger(__name__)


class CLITaskRunner:
    """CLI 任务运行器，负责执行生成任务"""

    def __init__(self, model_factory: ModelFactory, config_manager: ConfigManager):
        self.model_factory = model_factory
        self.config_manager = config_manager

    def execute_task(self, task_config: TaskConfig) -> str:
        """执行单个生成任务"""
        logger.info(
            f"开始执行任务: {task_config.model_type} - '{task_config.prompt[:50]}...'"
        )

        # 加载模型
        pipeline = self.model_factory.create_pipeline(
            model_type=task_config.model_type,
            device="cuda",  # 暂时硬编码
            dtype="bfloat16",
        )
        pipeline._interrupt = False  # type: ignore
        # 准备生成参数
        gen_params = task_config.get_generation_params()

        # 修正：确保为 WanAny2V 使用正确的 'input_prompt' 参数
        if "prompt" in gen_params:
            gen_params["input_prompt"] = gen_params.pop("prompt")

        # 设置随机种子
        if gen_params.get("seed") is not None:
            torch.manual_seed(gen_params["seed"])
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(gen_params["seed"])

        logger.info(f"生成参数: {gen_params}")
        logger.info(format_memory_usage())

        try:
            # 创建进度跟踪器
            progress_tracker = ProgressTracker(
                total_steps=gen_params["num_inference_steps"], logger=logger
            )
            gen_params["callback"] = lambda step, t, latents: progress_tracker.update(
                step
            )

            # 执行生成
            progress_tracker.start()
            start_time = time.time()

            if hasattr(pipeline, "__call__"):
                result = pipeline(**gen_params)
            elif hasattr(pipeline, "generate"):
                result = pipeline.generate(**gen_params)
            else:
                raise TypeError(f"模型管道 {type(pipeline)} 没有可调用的生成方法")

            generation_time = time.time() - start_time
            progress_tracker.finish(f"生成完成，耗时 {generation_time:.2f}s")

            # 保存结果
            output_path = self._save_result(result, task_config)
            return output_path

        except Exception as e:
            logger.error(f"任务执行失败: {e}", exc_info=True)
            raise
        finally:
            # 清理内存
            cleanup_gpu_memory()
            logger.info(f"任务完成后的内存情况: {format_memory_usage()}")

    def batch_execute(self, tasks: List[TaskConfig]) -> List[str]:
        """批量执行任务"""
        results = []
        total_tasks = len(tasks)

        for i, task_config in enumerate(tasks):
            logger.info(f"正在执行批处理任务 {i+1}/{total_tasks}")
            try:
                result_path = self.execute_task(task_config)
                results.append(result_path)
            except Exception as e:
                logger.error(f"批处理任务 {i+1} 失败: {e}")
                results.append(f"FAILED: {e}")

        return results

    def _save_result(self, result: Any, task_config: TaskConfig) -> str:
        """保存生成结果"""
        output_dir = task_config.output_path
        os.makedirs(output_dir, exist_ok=True)

        filename = task_config.get_output_filename()
        output_path = os.path.join(output_dir, filename)

        # 提取视频帧
        frames = self._extract_frames(result)
        if frames is None:
            raise ValueError("无法从模型输出中提取视频帧")

        # 保存视频
        save_video(frames=frames, output_path=output_path, fps=task_config.fps)
        logger.info(f"视频已保存到: {output_path}")

        # 保存元数据
        metadata_path = output_path.replace(".mp4", "_meta.json")
        self.config_manager.save_task_config(task_config, metadata_path)
        logger.info(f"元数据已保存到: {metadata_path}")

        return output_path

    def _extract_frames(self, result: Any) -> Optional[torch.Tensor]:
        """从不同类型的模型输出中提取视频帧"""
        frames = None
        if hasattr(result, "frames"):
            frames = result.frames
        elif hasattr(result, "videos"):
            frames = result.videos
        elif isinstance(result, torch.Tensor):
            frames = result
        elif (
            isinstance(result, list) and result and isinstance(result[0], torch.Tensor)
        ):
            frames = result[0]

        if frames is None:
            return None

        # 统一格式为 [B, T, H, W, C] -> [T, H, W, C]
        if frames.ndim == 5:
            frames = frames[0]

        return frames

    def test_model_loading(self, model_type: str) -> bool:
        """测试模型加载而不执行生成"""
        logger.info(f"开始测试模型加载: {model_type}")

        if not self.config_manager.validate_model_type(model_type):
            logger.error(f"✗ 无效的模型类型: {model_type}")
            return False
        logger.info("✓ 模型类型有效")

        try:
            self.model_loader.download_model(model_type)
            logger.info("✓ 文件下载（或检查）完成")

            # 这里可以添加更多检查，例如检查文件是否存在等
            model_filename = self.model_loader.get_model_filename(model_type)
            if not os.path.exists(model_filename):
                logger.error(f"✗ 模型主文件不存在: {model_filename}")
                return False
            logger.info(f"✓ 模型主文件存在: {model_filename}")

            logger.info(f"✓ 模型 {model_type} 加载测试通过")
            return True
        except Exception as e:
            logger.error(f"✗ 模型加载测试失败: {e}", exc_info=True)
            return False

    def cleanup(self):
        """清理资源"""
        self.model_factory.cleanup()
        logger.info("任务运行器已清理")
