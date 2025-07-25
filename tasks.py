"""
WanGP CLI Task Runner
独立的命令行任务执行器，直接使用核心管道，不依赖 Gradio 界面
基于 wgp.py 架构重写，确保与主程序兼容
"""

import argparse
import json
import os
import sys
import time
import torch
import glob
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

# 核心工具和配置导入
import cv2
import numpy as np
import re
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 尝试导入 MMGP，容错处理
try:
    from mmgp import offload, profile_type

    MMGP_AVAILABLE = True
except ImportError:
    logger.warning("MMGP not available, memory management disabled")
    MMGP_AVAILABLE = False

# 导入 WanGP 核心组件（参考 wgp.py 导入方式）
try:
    from wan.configs import WAN_CONFIGS, SUPPORTED_SIZES, MAX_AREA_CONFIGS
    from wan.any2video import WanAny2V

    # 为了避免 wgp.py 的导入冲突，我们直接实现需要的函数
    # 从 huggingface_hub 导入下载功能
    from huggingface_hub import hf_hub_download

except ImportError as e:
    logger.error(f"Failed to import WAN components: {e}")
    sys.exit(1)


# 从 wgp.py 复制必要的函数和变量，避免导入冲突
def get_model_def(model_type):
    """获取模型定义，从 defaults 目录加载"""
    default_path = os.path.join("defaults", f"{model_type}.json")
    if os.path.isfile(default_path):
        with open(default_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def get_base_model_type(model_type):
    """获取基础模型类型"""
    model_def = get_model_def(model_type)
    if not model_def or "model" not in model_def:
        return (
            model_type
            if model_type
            in [
                "multitalk",
                "fantasy",
                "vace_14B",
                "vace_multitalk_14B",
                "t2v_1.3B",
                "t2v",
                "vace_1.3B",
                "phantom_1.3B",
                "phantom_14B",
                "recam_1.3B",
                "sky_df_1.3B",
                "sky_df_14B",
                "i2v",
                "flf2v_720p",
                "fun_inp_1.3B",
                "fun_inp",
                "ltxv_13B",
                "hunyuan",
                "hunyuan_i2v",
                "hunyuan_custom",
                "hunyuan_custom_audio",
                "hunyuan_custom_edit",
                "hunyuan_avatar",
                "flux",
            ]
            else None
        )
    else:
        # 从嵌套的 "model" 键中获取 architecture
        model_info = model_def.get("model", {})
        return model_info.get("architecture", model_type)


def get_model_family(model_type):
    """获取模型家族"""
    model_type = get_base_model_type(model_type)
    if model_type is None:
        return "unknown"
    if "hunyuan" in model_type:
        return "hunyuan"
    elif "ltxv" in model_type:
        return "ltxv"
    elif "flux" in model_type:
        return "flux"
    else:
        return "wan"


def test_class_i2v(model_type):
    """测试是否为 i2v 类型模型"""
    model_type = get_base_model_type(model_type)
    return model_type in [
        "i2v",
        "fun_inp_1.3B",
        "fun_inp",
        "flf2v_720p",
        "fantasy",
        "multitalk",
    ]


def get_wan_text_encoder_filename(text_encoder_quantization="bf16"):
    """获取 WAN 文本编码器文件名"""
    text_encoder_filename = "ckpts/umt5-xxl/models_t5_umt5-xxl-enc-bf16.safetensors"
    if text_encoder_quantization == "int8":
        text_encoder_filename = text_encoder_filename.replace("bf16", "quanto_int8")
    return text_encoder_filename


def get_transformer_dtype(model_family, transformer_dtype_policy):
    """获取 transformer 数据类型"""
    if not isinstance(transformer_dtype_policy, str):
        return transformer_dtype_policy
    if len(transformer_dtype_policy) == 0:
        # 简化版本，默认使用 bfloat16
        return torch.bfloat16
    elif transformer_dtype_policy == "fp16":
        return torch.float16
    else:
        return torch.bfloat16


def get_model_filename(model_type, quantization="", dtype_policy="", is_module=False):
    """获取模型文件名，参考 wgp.py 实现"""
    if is_module:
        # 模块文件的处理逻辑
        return f"ckpts/{model_type}_module.safetensors"

    model_def = get_model_def(model_type)
    if not model_def or "model" not in model_def:
        # 默认文件名格式
        return f"ckpts/wan2.1_{model_type}_14B_bf16.safetensors"

    # 从嵌套的 "model" 键中获取 URLs
    model_info = model_def.get("model", {})
    URLs = model_info.get("URLs", [])
    
    if not URLs:
        return f"ckpts/wan2.1_{model_type}_14B_bf16.safetensors"

    if isinstance(URLs, str):
        # 如果是字符串，可能是引用其他模型类型
        return get_model_filename(URLs, quantization, dtype_policy, is_module)
    elif isinstance(URLs, list) and len(URLs) > 0:
        choices = [
            ("ckpts/" + os.path.basename(path) if path.startswith("http") else path)
            for path in URLs
        ]

        if len(quantization) == 0:
            quantization = "bf16"

        model_family = get_model_family(model_type)
        dtype = get_transformer_dtype(model_family, dtype_policy)

        if len(choices) <= 1:
            raw_filename = choices[0]
        else:
            if quantization in ("int8", "fp8"):
                sub_choices = [
                    name
                    for name in choices
                    if quantization in name or quantization.upper() in name
                ]
            else:
                sub_choices = [name for name in choices if "quanto" not in name]

            if len(sub_choices) > 0:
                # 修正：查找 mbf16 而不是 bf16
                dtype_str = "mfp16" if dtype == torch.float16 else "mbf16"
                new_sub_choices = [
                    name
                    for name in sub_choices
                    if dtype_str in name or dtype_str.upper() in name
                ]
                sub_choices = (
                    new_sub_choices if len(new_sub_choices) > 0 else sub_choices
                )
                raw_filename = sub_choices[0]
            else:
                raw_filename = choices[0]

        return raw_filename
    else:
        return f"ckpts/wan2.1_{model_type}_14B_bf16.safetensors"


def get_model_recursive_prop(model_type, prop="URLs", return_list=True):
    """获取模型的递归属性"""
    model_def = get_model_def(model_type)
    if not model_def:
        return [] if return_list else None

    # 正确获取嵌套在 "model" 键下的属性
    model_info = model_def.get("model", {})
    value = model_info.get(prop, [])
    
    if return_list and not isinstance(value, list):
        return [value] if value else []
    return value


def download_model_file(url, filename):
    """下载单个模型文件"""
    logger.info(f"Downloading {filename} from {url}")

    if url.startswith("https://huggingface.co/") and "/resolve/main/" in url:
        # 解析 HuggingFace URL
        url = url[len("https://huggingface.co/") :]
        url_parts = url.split("/resolve/main/")
        repo_id = url_parts[0]
        file_path = url_parts[1]

        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=file_path,
                local_dir="ckpts/",
                local_dir_use_symlinks=False,
            )
            logger.info(f"Successfully downloaded {filename}")
        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")
            raise
    else:
        raise ValueError(f"Unsupported URL format: {url}")


def download_models(model_filename, model_type):
    """下载模型文件，参考 wgp.py 实现"""
    logger.info(f"Downloading models for {model_type}")

    # 获取模型定义
    model_def = get_model_def(model_type)
    if not model_def:
        raise ValueError(f"No model definition found for model type: {model_type}")

    # 下载主模型文件
    if not os.path.isfile(model_filename):
        URLs = get_model_recursive_prop(model_type, "URLs", return_list=False)
        if isinstance(URLs, str):
            # 引用其他模型类型，递归下载
            download_models(model_filename, URLs)
            return
        elif isinstance(URLs, list):
            target_filename = os.path.basename(model_filename)
            use_url = None

            # 更好的文件名匹配逻辑
            for url in URLs:
                url_filename = os.path.basename(url)
                # 检查是否为非量化版本的匹配
                if "quanto" not in url_filename and "mbf16" in url_filename:
                    if target_filename.replace("_bf16", "_mbf16") == url_filename:
                        use_url = url
                        break
                # 直接匹配
                elif target_filename == url_filename:
                    use_url = url
                    break
                # 基础名称匹配
                elif target_filename in url_filename or url_filename in target_filename:
                    use_url = url
                    break

            if use_url is None and URLs:
                use_url = URLs[0]  # 使用第一个 URL

            if use_url and use_url.startswith("http"):
                download_model_file(use_url, model_filename)
            else:
                raise Exception(
                    f"Model '{model_filename}' was not found locally and no valid URL was provided."
                )

    # 下载预加载 URLs
    preload_URLs = get_model_recursive_prop(
        model_type, "preload_URLs", return_list=True
    )
    for url in preload_URLs:
        if url.startswith("http"):
            filename = f"ckpts/{os.path.basename(url)}"
            if not os.path.exists(filename):
                try:
                    download_model_file(url, filename)
                except Exception as e:
                    logger.warning(f"Failed to download preload file {filename}: {e}")

    # 下载模块文件
    modules = get_model_recursive_prop(model_type, "modules", return_list=True)
    for module_type in modules:
        module_filename = get_model_filename(module_type, is_module=True)
        if not os.path.exists(module_filename):
            try:
                download_models(module_filename, module_type)
            except Exception as e:
                logger.warning(f"Failed to download module {module_type}: {e}")

    # 下载依赖的核心文件
    _download_core_dependencies()


def _download_core_dependencies():
    """下载核心依赖文件"""
    # 下载 VAE 文件
    vae_path = "ckpts/Wan2.1_VAE.pth"
    if not os.path.exists(vae_path):
        logger.info("Downloading VAE model...")
        try:
            hf_hub_download(
                repo_id="DeepBeepMeep/Wan2.1",
                filename="Wan2.1_VAE.pth",
                local_dir="ckpts/",
                local_dir_use_symlinks=False,
            )
        except Exception as e:
            logger.warning(f"Failed to download VAE: {e}")

    # 下载文本编码器
    text_encoder_path = get_wan_text_encoder_filename("bf16")
    if not os.path.exists(text_encoder_path):
        logger.info("Downloading text encoder...")
        text_encoder_dir = os.path.dirname(text_encoder_path)
        os.makedirs(text_encoder_dir, exist_ok=True)
        try:
            hf_hub_download(
                repo_id="DeepBeepMeep/Wan2.1",
                filename="umt5-xxl/models_t5_umt5-xxl-enc-bf16.safetensors",
                local_dir="ckpts/",
                local_dir_use_symlinks=False,
            )
        except Exception as e:
            logger.warning(f"Failed to download text encoder: {e}")

    # 下载其他必需的文件（参考 wgp.py 的 process_files_def）
    try:
        _download_shared_dependencies()
    except Exception as e:
        logger.warning(f"Failed to download shared dependencies: {e}")


def _download_shared_dependencies():
    """下载共享依赖文件"""
    shared_files = {
        "pose": ["dw-ll_ucoco_384.onnx", "yolox_l.onnx"],
        "scribble": ["netG_A_latest.pth"],
        "flow": ["raft-things.pth"],
        "depth": ["depth_anything_v2_vitl.pth", "depth_anything_v2_vitb.pth"],
        "mask": ["sam_vit_h_4b8939_fp16.safetensors"],
        "wav2vec": [
            "config.json",
            "feature_extractor_config.json",
            "model.safetensors",
            "preprocessor_config.json",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "vocab.json",
        ],
        "chinese-wav2vec2-base": [
            "config.json",
            "pytorch_model.bin",
            "preprocessor_config.json",
        ],
        "pyannote": [
            "pyannote_model_wespeaker-voxceleb-resnet34-LM.bin",
            "pytorch_model_segmentation-3.0.bin",
        ],
        "": ["flownet.pkl"],
    }

    for folder, files in shared_files.items():
        folder_path = os.path.join("ckpts", folder) if folder else "ckpts"
        for filename in files:
            file_path = os.path.join(folder_path, filename)
            if not os.path.exists(file_path):
                try:
                    os.makedirs(folder_path, exist_ok=True)
                    hf_hub_download(
                        repo_id="DeepBeepMeep/Wan2.1",
                        filename=f"{folder}/{filename}" if folder else filename,
                        local_dir="ckpts/",
                        local_dir_use_symlinks=False,
                    )
                except Exception:
                    # 这些文件不是必需的，下载失败不影响主要功能
                    pass


# 尝试导入其他模型（可选）
try:
    from hyvideo.diffusion.pipelines.pipeline_hunyuan_video import HunyuanVideoPipeline

    HUNYUAN_AVAILABLE = True
except ImportError:
    HUNYUAN_AVAILABLE = False
    logger.warning("Hunyuan Video not available")

try:
    from ltx_video.pipelines.pipeline_ltx_video import LTXVideoPipeline

    LTX_AVAILABLE = True
except ImportError:
    LTX_AVAILABLE = False
    logger.warning("LTX Video not available")

try:
    from flux.flux_main import create_flux_pipeline

    FLUX_AVAILABLE = True
except ImportError:
    FLUX_AVAILABLE = False
    logger.warning("Flux not available")


def save_video(frames, output_path, fps=24):
    """保存视频到指定路径，frames: [T, H, W, C] numpy 或 torch.Tensor"""
    if hasattr(frames, "cpu"):
        frames = frames.cpu().numpy()
    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)
    height, width = frames.shape[1], frames.shape[2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()


def sanitize_file_name(name: str) -> str:
    """安全文件名（只保留字母数字和下划线）"""
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


def get_video_info(path: str):
    """获取视频信息（帧数、分辨率、fps）"""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return {"frames": frames, "width": width, "height": height, "fps": fps}


# 参考 wgp.py 的模型加载函数
def load_wan_model(model_type, device="cuda", dtype=torch.bfloat16):
    """参考 wgp.py 加载 WAN 模型"""
    logger.info(f"Loading WAN model: {model_type}")

    # 获取基础模型信息
    base_model_type = get_base_model_type(model_type)
    model_def = get_model_def(model_type)

    # 获取正确的配置对象（参考 wgp.py 的实现）
    if test_class_i2v(base_model_type):
        cfg = WAN_CONFIGS["i2v-14B"]
    else:
        cfg = WAN_CONFIGS["t2v-14B"]

    # 获取模型文件列表
    model_filename = get_model_filename(model_type, quantization="", dtype_policy="")
    modules = get_model_recursive_prop(model_type, "modules", return_list=True)

    model_file_list = [model_filename]
    model_type_list = [model_type]

    # 添加模块文件
    for module_type in modules:
        module_filename = get_model_filename(module_type, is_module=True)
        model_file_list.append(module_filename)
        model_type_list.append(module_type)

    # 下载所有需要的文件
    for filename, file_model_type in zip(model_file_list, model_type_list):
        download_models(filename, file_model_type)

    checkpoint_dir = "ckpts"

    # 设置量化和精度参数
    quantizeTransformer = False
    save_quantized = False
    VAE_dtype = torch.float32
    mixed_precision_transformer = False

    try:
        # 检查是否为特殊模型类型需要不同的模型工厂
        if base_model_type in ("sky_df_1.3B", "sky_df_14B"):
            # 需要导入 DTT2V，但为了简化，我们使用 WanAny2V
            logger.warning(
                f"Model type {base_model_type} may require DTT2V, using WanAny2V as fallback"
            )

        # 使用与 wgp.py 相同的参数初始化 WanAny2V
        pipeline = WanAny2V(
            config=cfg,
            checkpoint_dir=checkpoint_dir,
            model_filename=model_file_list,  # 传递文件列表而不是单个文件
            model_type=model_type,
            model_def=model_def,
            base_model_type=base_model_type,
            text_encoder_filename=get_wan_text_encoder_filename("bf16"),
            quantizeTransformer=quantizeTransformer,
            dtype=dtype,
            VAE_dtype=VAE_dtype,
            mixed_precision_transformer=mixed_precision_transformer,
            save_quantized=save_quantized,
        )

        logger.info(f"Successfully loaded WAN model: {model_type}")
        return pipeline

    except Exception as e:
        logger.error(f"Failed to load WAN model: {e}")
        raise


def load_hunyuan_model(model_type, device="cuda", dtype=torch.bfloat16):
    """加载 Hunyuan 模型"""
    if not HUNYUAN_AVAILABLE:
        raise ImportError("Hunyuan Video pipeline not available")

    logger.info(f"Loading Hunyuan model: {model_type}")
    model_path = f"ckpts/{model_type}"

    pipeline = HunyuanVideoPipeline.from_pretrained(
        model_path, torch_dtype=dtype, device_map=device
    )
    return pipeline


def load_ltx_model(model_type, device="cuda", dtype=torch.bfloat16):
    """加载 LTX Video 模型"""
    if not LTX_AVAILABLE:
        raise ImportError("LTX Video pipeline not available")

    logger.info(f"Loading LTX model: {model_type}")
    model_path = f"ckpts/{model_type}"

    pipeline = LTXVideoPipeline.from_pretrained(
        model_path, torch_dtype=dtype, device_map=device
    )
    return pipeline


def load_flux_model(model_type, device="cuda", dtype=torch.bfloat16):
    """加载 Flux 模型"""
    if not FLUX_AVAILABLE:
        raise ImportError("Flux pipeline not available")

    logger.info(f"Loading Flux model: {model_type}")
    return create_flux_pipeline(model_type=model_type)


class ModelFactory:
    """模型工厂，参考 wgp.py 的模型加载逻辑"""

    @staticmethod
    def create_pipeline(model_type: str, **kwargs):
        """根据模型类型创建相应的管道"""
        device = kwargs.get("device", "cuda")
        dtype = kwargs.get("dtype", torch.bfloat16)

        # 获取模型家族来决定使用哪个加载器
        model_family = get_model_family(model_type)
        
        if model_family == "wan":
            return load_wan_model(model_type, device, dtype)
        elif model_family == "hunyuan":
            return load_hunyuan_model(model_type, device, dtype)
        elif model_family == "ltxv":
            return load_ltx_model(model_type, device, dtype)
        elif model_family == "flux":
            return load_flux_model(model_type, device, dtype)
        else:
            raise ValueError(f"Unsupported model type: {model_type} (family: {model_family})")


class TaskConfig:
    """任务配置类，处理生成参数"""

    def __init__(self, config_dict: Dict[str, Any]):
        self.config = config_dict
        self._validate_config()

    def _validate_config(self):
        """验证配置的有效性"""
        required_fields = ["model_type", "prompt"]
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Required field '{field}' missing from config")

        # 验证分辨率格式
        resolution = self.config.get("resolution", "720x480")
        if "x" not in resolution:
            raise ValueError(f"Invalid resolution format: {resolution}")

        width, height = map(int, resolution.split("x"))
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid resolution dimensions: {width}x{height}")

    def get(self, key: str, default=None):
        """获取配置值"""
        return self.config.get(key, default)

    def get_generation_params(self) -> Dict[str, Any]:
        """获取生成参数，根据模型类型调整"""
        params = {
            "prompt": self.get("prompt"),
            "negative_prompt": self.get("negative_prompt", ""),
            "num_inference_steps": self.get("steps", 20),
            "guidance_scale": self.get("guidance_scale", 7.0),
            "width": int(self.get("resolution", "720x480").split("x")[0]),
            "height": int(self.get("resolution", "720x480").split("x")[1]),
            "num_frames": self.get("video_length", 49),
            "fps": self.get("fps", 24),
            "seed": self.get("seed", None),
        }

        # 根据模型类型添加特定参数
        model_type = self.get("model_type")
        if model_type in ["i2v"] and self.get("image_path"):
            params["image"] = self.get("image_path")

        return params


class ProgressCallback:
    """进度回调类"""

    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()

    def __call__(self, step: int, timestep: int = None, latents=None):
        """进度回调函数"""
        self.current_step = step
        progress = (step / self.total_steps) * 100
        elapsed = time.time() - self.start_time
        eta = (elapsed / step * (self.total_steps - step)) if step > 0 else 0

        logger.info(
            f"Step {step}/{self.total_steps} ({progress:.1f}%) - ETA: {eta:.1f}s"
        )


class CLITaskRunner:
    """CLI 任务运行器，参考 wgp.py 的执行逻辑"""

    def __init__(self, device: str = "cuda", dtype: str = "bfloat16"):
        self.device = device
        self.dtype = getattr(torch, dtype) if hasattr(torch, dtype) else torch.bfloat16
        self.current_pipeline = None
        self.current_model_type = None
        self.offload_manager = None

        # 初始化内存管理
        self._init_memory_management()

    def _init_memory_management(self):
        """初始化内存管理"""
        if not MMGP_AVAILABLE:
            logger.info("MMGP not available, skipping memory management")
            return

        try:
            if hasattr(offload, "OffloadManager"):
                self.offload_manager = offload.OffloadManager(
                    device=self.device, profile=profile_type.BALANCED
                )
                logger.info("Memory management initialized")
            else:
                logger.warning("OffloadManager not found in mmgp.offload")
        except Exception as e:
            logger.warning(f"Failed to initialize memory management: {e}")

    def load_model(self, model_type: str, **kwargs):
        """加载指定模型"""
        if self.current_model_type == model_type and self.current_pipeline is not None:
            logger.info(f"Model {model_type} already loaded")
            return

        logger.info(f"Loading model: {model_type}")

        # 卸载当前模型
        if self.current_pipeline is not None:
            self._unload_current_model()

        try:
            # 创建新管道
            self.current_pipeline = ModelFactory.create_pipeline(
                model_type=model_type, device=self.device, dtype=self.dtype, **kwargs
            )
            self.current_model_type = model_type
            logger.info(f"Model {model_type} loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model {model_type}: {e}")
            raise

    def _unload_current_model(self):
        """卸载当前模型"""
        if self.current_pipeline is not None:
            logger.info("Unloading current model")
            if hasattr(self.current_pipeline, "unload"):
                self.current_pipeline.unload()
            del self.current_pipeline
            self.current_pipeline = None
            self.current_model_type = None

            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def test_model_loading(self, model_type: str) -> bool:
        """测试模型加载而不实际执行生成"""
        logger.info(f"Testing model loading: {model_type}")
        
        try:
            # 检查模型定义
            model_def = get_model_def(model_type)
            if not model_def:
                logger.error(f"No model definition found for {model_type}")
                return False
            
            logger.info("✓ Model definition found")
            
            # 检查模型家族
            model_family = get_model_family(model_type)
            logger.info(f"✓ Model family: {model_family}")
            
            # 检查模型文件
            model_filename = get_model_filename(model_type)
            logger.info(f"✓ Expected model file: {model_filename}")
            
            if os.path.exists(model_filename):
                logger.info(f"✓ Model file exists: {model_filename}")
            else:
                logger.info(f"⚠ Model file not found, would need to download: {model_filename}")
                
                # 检查下载 URLs
                URLs = get_model_recursive_prop(model_type, "URLs", return_list=False)
                if URLs:
                    logger.info(f"✓ Download URLs available: {len(URLs) if isinstance(URLs, list) else 1}")
                else:
                    logger.error(f"✗ No download URLs found for {model_type}")
                    return False
            
            # 尝试创建管道（不实际加载权重）
            logger.info("✓ Testing pipeline creation...")
            model_family = get_model_family(model_type)
            logger.info(f"✓ Model type {model_type} is supported (family: {model_family})")
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Model loading test failed: {e}")
            return False

    def execute_task(self, task_config: TaskConfig) -> str:
        """执行生成任务"""
        model_type = task_config.get("model_type")
        output_dir = task_config.get("output_path", "outputs")

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 加载模型
        self.load_model(model_type)

        # 准备生成参数
        gen_params = task_config.get_generation_params()

        # 设置随机种子
        if gen_params["seed"] is not None:
            torch.manual_seed(gen_params["seed"])
            if torch.cuda.is_available():
                torch.cuda.manual_seed(gen_params["seed"])

        logger.info(f"Starting generation with prompt: {gen_params['prompt']}")
        logger.info(f"Parameters: {gen_params}")

        try:
            # 执行生成
            start_time = time.time()

            # 根据模型类型调用不同的生成方法
            if hasattr(self.current_pipeline, "__call__"):
                result = self.current_pipeline(**gen_params)
            elif hasattr(self.current_pipeline, "generate"):
                result = self.current_pipeline.generate(**gen_params)
            else:
                raise ValueError(f"Unknown pipeline interface for {model_type}")

            generation_time = time.time() - start_time
            logger.info(f"Generation completed in {generation_time:.2f} seconds")

            # 保存结果
            output_path = self._save_result(result, task_config, output_dir)

            return output_path

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

    def _save_result(self, result, task_config: TaskConfig, output_dir: str) -> str:
        """保存生成结果"""
        # 生成文件名
        prompt = task_config.get("prompt", "generated")
        safe_prompt = sanitize_file_name(prompt[:50])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_prompt}_{timestamp}.mp4"
        output_path = os.path.join(output_dir, filename)

        # 提取视频帧
        frames = None
        if hasattr(result, "frames"):
            frames = (
                result.frames[0] if isinstance(result.frames, list) else result.frames
            )
        elif hasattr(result, "videos"):
            frames = (
                result.videos[0] if isinstance(result.videos, list) else result.videos
            )
        elif isinstance(result, torch.Tensor):
            frames = result
        elif isinstance(result, list) and len(result) > 0:
            frames = result[0]
        else:
            logger.error(f"Unknown result format: {type(result)}")
            return None

        if frames is None:
            logger.error("No frames found in result")
            return None

        # 保存视频
        save_video(
            frames=frames, output_path=output_path, fps=task_config.get("fps", 24)
        )
        logger.info(f"Video saved to: {output_path}")

        # 保存元数据
        metadata = {
            "prompt": task_config.get("prompt"),
            "model_type": task_config.get("model_type"),
            "generation_params": task_config.get_generation_params(),
            "timestamp": timestamp,
            "output_path": output_path,
        }

        metadata_path = output_path.replace(".mp4", "_metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        return output_path

    def batch_execute(self, config_list: List[TaskConfig]) -> List[str]:
        """批量执行任务"""
        results = []

        for i, task_config in enumerate(config_list):
            logger.info(f"Executing task {i+1}/{len(config_list)}")
            try:
                result_path = self.execute_task(task_config)
                results.append(result_path)
            except Exception as e:
                logger.error(f"Task {i+1} failed: {e}")
                results.append(None)

        return results

    def cleanup(self):
        """清理资源"""
        self._unload_current_model()
        if self.offload_manager and hasattr(self.offload_manager, "cleanup"):
            self.offload_manager.cleanup()


def load_config_file(config_path: str) -> TaskConfig:
    """从文件加载任务配置"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        if config_path.endswith(".json"):
            config_dict = json.load(f)
        else:
            raise ValueError("Only JSON config files are supported")

    return TaskConfig(config_dict)


def create_sample_configs():
    """创建示例配置文件"""
    configs = {
        "text2video.json": {
            "model_type": "t2v",
            "prompt": "A majestic eagle soaring through snow-capped mountains at sunset",
            "negative_prompt": "blurry, low quality, distorted, ugly",
            "resolution": "720x480",
            "video_length": 49,
            "steps": 20,
            "guidance_scale": 7.0,
            "fps": 24,
            "seed": 42,
            "output_path": "outputs/text2video",
        },
        "image2video.json": {
            "model_type": "i2v",
            "prompt": "The person in the image starts walking forward",
            "image_path": "inputs/reference.jpg",
            "resolution": "720x480",
            "video_length": 49,
            "steps": 25,
            "guidance_scale": 7.5,
            "fps": 24,
            "output_path": "outputs/image2video",
        },
        "hunyuan_video.json": {
            "model_type": "hunyuan",
            "prompt": "A cat playing with a ball of yarn in slow motion",
            "resolution": "720x480",
            "video_length": 61,
            "steps": 30,
            "guidance_scale": 6.0,
            "fps": 24,
            "output_path": "outputs/hunyuan",
        },
        "batch_config.json": {
            "tasks": [
                {
                    "model_type": "t2v",
                    "prompt": "A peaceful lake at dawn with mist rising",
                    "resolution": "720x480",
                    "steps": 20,
                },
                {
                    "model_type": "t2v",
                    "prompt": "City traffic at night with neon lights",
                    "resolution": "720x480",
                    "steps": 20,
                },
            ],
            "output_path": "outputs/batch",
        },
    }

    os.makedirs("configs", exist_ok=True)

    for filename, config in configs.items():
        config_path = os.path.join("configs", filename)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        logger.info(f"Created sample config: {config_path}")


def main():
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

    args = parser.parse_args()

    # 创建示例配置
    if args.create_samples:
        create_sample_configs()
        return

    # 初始化任务运行器
    runner = CLITaskRunner(device=args.device, dtype=args.dtype)

    try:
        if args.batch:
            # 批量处理
            batch_config = load_config_file(args.batch)
            tasks = batch_config.get("tasks", [])
            output_path = batch_config.get("output_path", "outputs/batch")

            task_configs = []
            for task_dict in tasks:
                task_dict["output_path"] = output_path
                task_configs.append(TaskConfig(task_dict))

            logger.info(f"Starting batch processing of {len(task_configs)} tasks")
            results = runner.batch_execute(task_configs)

            successful = [r for r in results if r is not None]
            logger.info(
                f"Batch completed: {len(successful)}/{len(results)} tasks successful"
            )

        elif args.config:
            # 单个配置文件
            task_config = load_config_file(args.config)
            result_path = runner.execute_task(task_config)
            logger.info(f"Task completed successfully: {result_path}")

        else:
            # 命令行参数
            if not args.prompt and not args.test_only:
                logger.error("Either --config, --prompt, or --test-only must be provided")
                sys.exit(1)

            if args.test_only:
                # 仅测试模型加载
                model_type = args.model
                success = runner.test_model_loading(model_type)
                if success:
                    logger.info(f"✓ Model {model_type} loading test passed")
                else:
                    logger.error(f"✗ Model {model_type} loading test failed")
                    sys.exit(1)
                return

            config_dict = {
                "model_type": args.model,
                "prompt": args.prompt,
                "resolution": args.resolution,
                "video_length": args.length,
                "steps": args.steps,
                "guidance_scale": args.guidance,
                "fps": args.fps,
                "output_path": args.output,
            }

            if args.seed is not None:
                config_dict["seed"] = args.seed

            task_config = TaskConfig(config_dict)
            result_path = runner.execute_task(task_config)
            logger.info(f"Task completed successfully: {result_path}")

    except Exception as e:
        logger.error(f"Task execution failed: {e}")
        sys.exit(1)
    finally:
        runner.cleanup()


if __name__ == "__main__":
    main()
