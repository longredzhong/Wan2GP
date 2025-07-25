"""
WanGP 模型加载和管理模块
"""

import sys
import torch
import logging
import types
from pathlib import Path
from typing import Any, Optional

# 尝试导入 MMGP，容错处理
try:
    from mmgp import offload, profile_type
    MMGP_AVAILABLE = True
except ImportError:
    MMGP_AVAILABLE = False

# 导入 WanGP 核心组件
try:
    from wan.configs import WAN_CONFIGS
    from wan.any2video import WanAny2V
    from huggingface_hub import hf_hub_download
except ImportError as e:
    logging.getLogger(__name__).error(f"无法导入 WAN 组件: {e}")
    sys.exit(1)

# 尝试导入其他模型管道
try:
    from hyvideo.diffusion.pipelines.pipeline_hunyuan_video import HunyuanVideoPipeline
    HUNYUAN_AVAILABLE = True
except ImportError:
    HUNYUAN_AVAILABLE = False

try:
    from ltx_video.pipelines.pipeline_ltx_video import LTXVideoPipeline
    LTX_AVAILABLE = True
except ImportError:
    LTX_AVAILABLE = False

try:
    from flux.flux_main import create_flux_pipeline
    FLUX_AVAILABLE = True
except ImportError:
    FLUX_AVAILABLE = False

from .config_manager import ConfigManager

logger = logging.getLogger(__name__)


class ModelLoader:
    """模型文件加载器，负责下载和管理模型文件"""

    def __init__(self, config_manager: ConfigManager, ckpts_dir: str = "ckpts"):
        self.config_manager = config_manager
        self.ckpts_dir = Path(ckpts_dir)
        self.ckpts_dir.mkdir(exist_ok=True)
        self._core_deps_downloaded = False

    def get_model_filename(self, model_type: str, quantization: str = "", dtype_policy: str = "", is_module: bool = False) -> str:
        """获取模型文件名"""
        if is_module:
            return str(self.ckpts_dir / f"{model_type}_module.safetensors")

        definition = self.config_manager.get_model_definition(model_type)
        if not definition:
            # 默认文件名格式
            return str(self.ckpts_dir / f"wan2.1_{model_type}_14B_bf16.safetensors")

        urls = self.config_manager.resolve_model_urls(model_type)
        if not urls:
            return str(self.ckpts_dir / f"wan2.1_{model_type}_14B_bf16.safetensors")

        choices = [(self.ckpts_dir / Path(url).name) for url in urls]

        if not quantization:
            quantization = "bf16"

        if len(choices) <= 1:
            return str(choices[0])
        
        # 根据量化和精度选择最佳文件
        if quantization in ("int8", "fp8"):
            sub_choices = [p for p in choices if quantization in p.name or quantization.upper() in p.name]
        else:
            sub_choices = [p for p in choices if "quanto" not in p.name]

        if not sub_choices:
            sub_choices = choices

        # 根据数据类型进一步筛选
        dtype = self._get_transformer_dtype(dtype_policy)
        dtype_str = "mfp16" if dtype == torch.float16 else "mbf16"
        
        final_choices = [p for p in sub_choices if dtype_str in p.name or dtype_str.upper() in p.name]
        
        return str(final_choices[0] if final_choices else sub_choices[0])

    def download_model(self, model_type: str):
        """下载指定模型及其所有依赖"""
        logger.info(f"为模型 {model_type} 检查并下载文件...")
        
        definition = self.config_manager.get_model_definition(model_type)
        if not definition:
            raise ValueError(f"未找到模型定义: {model_type}")

        # 下载主模型文件
        urls = self.config_manager.resolve_model_urls(model_type)
        for url in urls:
            self._download_file_from_url(url)

        # 下载预加载文件
        for url in definition.preload_urls:
            self._download_file_from_url(url)

        # 递归下载模块
        for module_type in definition.modules:
            self.download_model(module_type)

        # 下载核心依赖（只执行一次）
        if not self._core_deps_downloaded:
            self._download_core_dependencies()
            self._core_deps_downloaded = True

    def _download_file_from_url(self, url: str):
        """从HuggingFace Hub下载文件"""
        if not url.startswith("https://huggingface.co/"):
            logger.warning(f"不支持的URL格式，跳过下载: {url}")
            return

        try:
            url_path = url.replace("https://huggingface.co/", "")
            repo_id, file_path = url_path.split("/resolve/main/", 1)
            
            destination = self.ckpts_dir / file_path
            if destination.exists():
                return

            logger.info(f"正在下载 {file_path} 从仓库 {repo_id}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            
            hf_hub_download(
                repo_id=repo_id,
                filename=file_path,
                local_dir=self.ckpts_dir,
                local_dir_use_symlinks=False,
            )
            logger.info(f"成功下载 {destination.name}")
        except Exception as e:
            logger.error(f"从 {url} 下载失败: {e}", exc_info=True)
    
    def _download_core_dependencies(self):
        """下载所有模型共享的核心依赖"""
        logger.info("检查核心依赖文件...")
        
        core_deps = {
            "DeepBeepMeep/Wan2.1": [
                "Wan2.1_VAE.pth",
                "Wan2.1_VAE.safetensors",
                "umt5-xxl/models_t5_umt5-xxl-enc-bf16.safetensors",
                "pose/dw-ll_ucoco_384.onnx",
                "pose/yolox_l.onnx",
                "scribble/netG_A_latest.pth",
                "flow/raft-things.pth",
                "depth/depth_anything_v2_vitl.pth",
                "depth/depth_anything_v2_vitb.pth",
                "mask/sam_vit_h_4b8939_fp16.safetensors",
                "wav2vec/config.json", "wav2vec/model.safetensors", "wav2vec/vocab.json",
                "chinese-wav2vec2-base/config.json", "chinese-wav2vec2-base/pytorch_model.bin",
                "pyannote/pyannote_model_wespeaker-voxceleb-resnet34-LM.bin", 
                "pyannote/pytorch_model_segmentation-3.0.bin",
                "flownet.pkl",
            ]
        }

        for repo_id, files in core_deps.items():
            for file_path in files:
                self._download_file_from_url(f"https://huggingface.co/{repo_id}/resolve/main/{file_path}")

    def get_wan_text_encoder_filename(self, quantization="bf16") -> str:
        """获取WAN文本编码器文件名"""
        base_name = "umt5-xxl/models_t5_umt5-xxl-enc-bf16.safetensors"
        if quantization == "int8":
            base_name = base_name.replace("bf16", "quanto_int8")
        return str(self.ckpts_dir / base_name)

    def _get_transformer_dtype(self, dtype_policy: str) -> torch.dtype:
        """获取Transformer的数据类型"""
        if not isinstance(dtype_policy, str) or not dtype_policy:
            return torch.bfloat16
        
        policy_map = {
            "fp16": torch.float16, "float16": torch.float16,
            "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
            "fp32": torch.float32, "float32": torch.float32,
        }
        return policy_map.get(dtype_policy.lower(), torch.bfloat16)


class ModelFactory:
    """模型工厂，用于创建和管理模型管道实例"""

    def __init__(self, config_manager: ConfigManager, model_loader: ModelLoader):
        self.config_manager = config_manager
        self.model_loader = model_loader
        self.current_pipeline: Optional[Any] = None
        self.current_model_type: Optional[str] = None
        self.offload_manager: Optional[Any] = None
        self._init_memory_management()

    def _init_memory_management(self):
        """初始化内存管理器"""
        if not MMGP_AVAILABLE:
            logger.info("MMGP 不可用，跳过内存管理")
            return
        try:
            if hasattr(offload, "OffloadManager"):
                self.offload_manager = offload.OffloadManager(profile=profile_type.BALANCED)
                logger.info("内存管理器初始化成功")
            else:
                logger.warning("mmgp.offload 中未找到 OffloadManager")
        except Exception as e:
            logger.warning(f"初始化内存管理器失败: {e}")

    def create_pipeline(self, model_type: str, device: str = "cuda", dtype: str = "bfloat16", **kwargs):
        """根据模型类型创建管道"""
        if self.current_model_type == model_type and self.current_pipeline:
            logger.info(f"模型 {model_type} 已加载")
            return self.current_pipeline

        self.unload_current_model()
        
        logger.info(f"正在加载模型: {model_type}")
        self.model_loader.download_model(model_type)

        model_family = self.config_manager.get_model_family(
            self.config_manager.get_base_model_type(model_type)
        )
        
        torch_dtype = self.model_loader._get_transformer_dtype(dtype)

        try:
            if model_family == "wan":
                pipeline = self._load_wan_model(model_type, device, torch_dtype)
            elif model_family == "hunyuan":
                pipeline = self._load_hunyuan_model(model_type, device, torch_dtype)
            elif model_family == "ltxv":
                pipeline = self._load_ltx_model(model_type, device, torch_dtype)
            elif model_family == "flux":
                pipeline = self._load_flux_model(model_type, device, torch_dtype)
            else:
                raise ValueError(f"不支持的模型家族: {model_family}")
            
            self.current_pipeline = pipeline
            self.current_model_type = model_type
            logger.info(f"模型 {model_type} 加载成功")
            return pipeline
        except Exception as e:
            logger.error(f"加载模型 {model_type} 失败: {e}", exc_info=True)
            raise

    def _load_wan_model(self, model_type: str, device: str, dtype: torch.dtype):
        """加载WAN系列模型"""
        base_model_type = self.config_manager.get_base_model_type(model_type)
        model_def = self.config_manager.get_model_definition(model_type)
        
        cfg = WAN_CONFIGS["i2v-14B"] if self.config_manager.is_i2v_model(model_type) else WAN_CONFIGS["t2v-14B"]
        
        model_filename = self.model_loader.get_model_filename(model_type)
        modules = model_def.modules if model_def else []
        
        model_file_list = [model_filename]
        for module in modules:
            model_file_list.append(self.model_loader.get_model_filename(module, is_module=True))

        model_def_dict = model_def.__dict__ if model_def else {}
        
        pipeline = WanAny2V(
            config=cfg,
            checkpoint_dir=str(self.model_loader.ckpts_dir),
            model_filename=model_file_list,
            model_type=model_type,
            model_def=model_def_dict,
            base_model_type=base_model_type,
            text_encoder_filename=self.model_loader.get_wan_text_encoder_filename(),
            quantizeTransformer=False,
            dtype=dtype,
            VAE_dtype=torch.float32,
            mixed_precision_transformer=False,
            save_quantized=False,
        )

        # 猴子补丁来修复设备不匹配问题
        if hasattr(pipeline, 'text_encoder'):
            pipeline.text_encoder.__class__.__call__ = _patched_text_encoder_call
            logger.info("已应用T5文本编码器的设备修复补丁。")

        return pipeline

    def _load_hunyuan_model(self, model_type: str, device: str, dtype: torch.dtype):
        """加载Hunyuan模型"""
        if not HUNYUAN_AVAILABLE:
            raise ImportError("Hunyuan Video pipeline 不可用")
        model_path = self.model_loader.ckpts_dir / model_type
        return HunyuanVideoPipeline.from_pretrained(str(model_path), torch_dtype=dtype, device_map=device)

    def _load_ltx_model(self, model_type: str, device: str, dtype: torch.dtype):
        """加载LTX Video模型"""
        if not LTX_AVAILABLE:
            raise ImportError("LTX Video pipeline 不可用")
        model_path = self.model_loader.ckpts_dir / model_type
        return LTXVideoPipeline.from_pretrained(str(model_path), torch_dtype=dtype, device_map=device)

    def _load_flux_model(self, model_type: str, device: str, dtype: torch.dtype):
        """加载Flux模型"""
        if not FLUX_AVAILABLE:
            raise ImportError("Flux pipeline 不可用")
        return create_flux_pipeline(model_type=model_type, dtype=dtype, device=device)

    def unload_current_model(self):
        """卸载当前模型以释放内存"""
        if self.current_pipeline:
            logger.info(f"正在卸载模型: {self.current_model_type}")
            if hasattr(self.current_pipeline, "unload"):
                self.current_pipeline.unload()
            del self.current_pipeline
            self.current_pipeline = None
            self.current_model_type = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def cleanup(self):
        """清理所有资源"""
        self.unload_current_model()
        if self.offload_manager and hasattr(self.offload_manager, "cleanup"):
            self.offload_manager.cleanup()
            logger.info("内存管理器已清理")


def _patched_text_encoder_call(self, texts, device):
    """
    对 T5EncoderModel.__call__ 的猴子补丁版本。
    确保输入张量在模型所在的设备上（CPU），并将输出张量移动到目标设备上（GPU）。
    """
    # self 在这里是 T5EncoderModel 的实例
    # self.device 是 'cpu', device 是 'cuda'
    ids, mask = self.tokenizer(
        texts, return_mask=True, add_special_tokens=True)
    
    # 将输入移动到模型所在的CPU设备
    ids = ids.to(self.device)
    mask = mask.to(self.device)
    
    seq_lens = mask.gt(0).sum(dim=1).long()
    
    # 在CPU上执行模型
    context = self.model(ids, mask)
    
    # 将输出移动到原始请求的GPU设备
    return [u[:v].to(device) for u, v in zip(context, seq_lens)]
