"""
WanGP CLI Task Runner - New Entry Point
"""

import argparse
import sys

from tasks.config_manager import ConfigManager, TaskConfig
from tasks.model_loader import ModelLoader, ModelFactory
from tasks.task_runner import CLITaskRunner
from tasks.utils import setup_logging

def main():
    """主函数：解析命令行参数并执行任务"""
    
    # 设置日志
    logger = setup_logging(level="INFO")

    parser = argparse.ArgumentParser(
        description="WanGP CLI Task Runner - Independent video generation without Gradio interface"
    )

    # 基本参数
    parser.add_argument(
        "--config", type=str, help="Path to task configuration file (JSON)"
    )
    parser.add_argument("--prompt", type=str, help="Text prompt for video generation")
    parser.add_argument(
        "--model", type=str, default="t2v", help="Model type (default: t2v)"
    )
    parser.add_argument(
        "--output", type=str, default="outputs", help="Output directory"
    )

    # 生成参数
    parser.add_argument(
        "--steps", type=int, default=20, help="Number of inference steps"
    )
    parser.add_argument("--guidance", type=float, default=7.0, help="Guidance scale")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument(
        "--resolution", type=str, default="720x480", help="Video resolution"
    )
    parser.add_argument("--length", type=int, default=49, help="Video length in frames")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second")

    # 系统参数
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="Data type (bfloat16/float16/float32)",
    )

    # 工具命令
    parser.add_argument(
        "--create-samples",
        action="store_true",
        help="Create sample configuration files",
    )
    parser.add_argument("--batch", type=str, help="Path to batch configuration file")
    parser.add_argument(
        "--test-only",
        action="store_true", 
        help="Test model loading without executing generation"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available models"
    )

    args = parser.parse_args()

    # 初始化核心组件
    try:
        config_manager = ConfigManager()
        model_loader = ModelLoader(config_manager)
        model_factory = ModelFactory(config_manager, model_loader)
        runner = CLITaskRunner(model_factory, config_manager)
    except Exception as e:
        logger.error(f"初始化失败: {e}", exc_info=True)
        sys.exit(1)

    # 处理命令
    try:
        if args.create_samples:
            config_manager.create_sample_configs()
            return
            
        if args.list_models:
            logger.info("可用的模型类型:")
            for model_type in sorted(config_manager.get_all_model_types()):
                logger.info(f"- {model_type}")
            return

        if args.batch:
            # 批量处理
            logger.info(f"加载批处理配置文件: {args.batch}")
            tasks = config_manager.load_task_config(args.batch)
            logger.info(f"开始处理 {len(tasks)} 个任务的批处理")
            results = runner.batch_execute(tasks)
            
            successful = [r for r in results if not r.startswith("FAILED")]
            logger.info(f"批处理完成: {len(successful)}/{len(results)} 个任务成功")

        elif args.config:
            # 单个配置文件
            logger.info(f"加载任务配置文件: {args.config}")
            task_config = config_manager.load_task_config(args.config)
            result_path = runner.execute_task(task_config)
            logger.info(f"任务成功完成: {result_path}")

        else:
            # 命令行参数
            if not args.prompt and not args.test_only:
                parser.error("必须提供 --config, --prompt, 或 --test-only 参数之一")

            if args.test_only:
                # 仅测试模型加载
                success = runner.test_model_loading(args.model)
                if success:
                    logger.info(f"✓ 模型 {args.model} 加载测试通过")
                else:
                    logger.error(f"✗ 模型 {args.model} 加载测试失败")
                    sys.exit(1)
                return

            # 从命令行参数创建任务配置
            config_dict = {
                "model_type": args.model,
                "prompt": args.prompt,
                "resolution": args.resolution,
                "video_length": args.length,
                "steps": args.steps,
                "guidance_scale": args.guidance,
                "fps": args.fps,
                "output_path": args.output,
                "seed": args.seed,
            }
            task_config = TaskConfig.from_dict(config_dict)
            result_path = runner.execute_task(task_config)
            logger.info(f"任务成功完成: {result_path}")

    except Exception as e:
        logger.error(f"任务执行失败: {e}", exc_info=True)
        sys.exit(1)
    finally:
        runner.cleanup()


if __name__ == "__main__":
    main()
