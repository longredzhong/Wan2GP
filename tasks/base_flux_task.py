from pydantic import BaseModel, Field, field_validator, model_validator
import os
import time
import glob
from flux.flux_main import model_factory  # Flux 模型工厂
from wan.utils.utils import convert_tensor_to_image
from PIL import Image
from typing import List, Optional, Literal, Union
from enum import Enum

def find_flux_model_file(model_name: str, quantization: str = "int8") -> List[str]:
    """
    查找 Flux 模型文件路径
    Args:
        model_name: 模型名称 (flux-dev, flux-schnell, flux-dev-kontext)
        quantization: 量化类型 (int8, bf16)
    Returns:
        模型文件路径列表
    """
    # 常见的模型文件名模式
    model_patterns = {
        "flux-dev": [
            "ckpts/flux1-dev_quanto_bf16_int8.safetensors",
            "ckpts/flux1-dev_bf16.safetensors",
            "ckpts/*flux*dev*int8*.safetensors",
            "ckpts/*flux*dev*bf16*.safetensors",
            "ckpts/*flux*dev*.safetensors"
        ],
        "flux-schnell": [
            "ckpts/flux1-schnell_quanto_bf16_int8.safetensors", 
            "ckpts/flux1-schnell_bf16.safetensors",
            "ckpts/*flux*schnell*int8*.safetensors",
            "ckpts/*flux*schnell*bf16*.safetensors", 
            "ckpts/*flux*schnell*.safetensors"
        ],
        "flux-dev-kontext": [
            "ckpts/flux1_kontext_dev_quanto_bf16_int8.safetensors",
            "ckpts/flux1_kontext_dev_bf16.safetensors",
            "ckpts/*flux*kontext*int8*.safetensors",
            "ckpts/*flux*kontext*bf16*.safetensors",
            "ckpts/*flux*kontext*.safetensors"
        ]
    }
    
    patterns = model_patterns.get(model_name, model_patterns["flux-dev"])
    
    # 根据量化类型调整优先级
    if quantization == "int8":
        # 优先选择量化模型
        patterns = [p for p in patterns if "int8" in p or "quanto" in p] + \
                  [p for p in patterns if "int8" not in p and "quanto" not in p]
    else:
        # 优先选择非量化模型  
        patterns = [p for p in patterns if "int8" not in p and "quanto" not in p] + \
                  [p for p in patterns if "int8" in p or "quanto" in p]
    
    # 查找存在的文件
    for pattern in patterns:
        if "*" in pattern:
            # 通配符搜索
            files = glob.glob(pattern)
            if files:
                return files[:1]  # 返回第一个匹配的文件
        else:
            # 精确匹配
            if os.path.exists(pattern):
                return [pattern]
    
    # 如果找不到特定模型，尝试查找通用 flux 模型
    fallback_patterns = [
        "ckpts/*flux*.safetensors",
        "ckpts/flux*.safetensors"
    ]
    
    for pattern in fallback_patterns:
        files = glob.glob(pattern)
        if files:
            print(f"Warning: Using fallback model {files[0]} for {model_name}")
            return files[:1]
    
    raise FileNotFoundError(f"No Flux model files found for {model_name}. Please ensure model files are in the ckpts/ directory.")

class FluxModelType(str, Enum):
    """Flux 模型类型枚举"""
    FLUX_DEV = "flux-dev"
    FLUX_SCHNELL = "flux-schnell"
    FLUX_DEV_KONTEXT = "flux-dev-kontext"

class ControlType(str, Enum):
    """控制类型枚举"""
    NONE = "none"
    DEPTH = "depth"
    CANNY = "canny"
    REDUX = "redux"

class BaseFluxParams(BaseModel):
    """Flux 基础参数"""
    model_name: FluxModelType = Field(FluxModelType.FLUX_DEV, description="Flux 模型名称")
    prompt: str = Field(..., description="Text prompt for image generation")
    width: int = Field(1024, ge=64, le=2048, description="Output image width")
    height: int = Field(1024, ge=64, le=2048, description="Output image height")  
    seed: int = Field(-1, ge=-1, description="Random seed (-1 表示随机)")
    steps: int = Field(50, ge=1, le=150, description="Number of inference steps")
    guidance_scale: float = Field(3.5, ge=0, le=20, description="Guidance scale (0 for schnell model)")
    output_dir: str = Field("outputs", description="Directory to save the generated image")
    loras_slists: Optional[List[str]] = Field(default=None, description="要加载的 LoRA 列表")
    batch_size: int = Field(1, gt=0, le=4, description="Batch size for generation")

    @field_validator("prompt")
    @classmethod
    def non_empty_prompt(cls, v):
        if not v.strip():
            raise ValueError("Prompt cannot be empty")
        return v

    @field_validator("output_dir")
    @classmethod
    def ensure_output_dir(cls, v):
        os.makedirs(v, exist_ok=True)
        return v
        
    @model_validator(mode='after')
    def validate_model_specific_params(self):
        # flux-schnell 通常使用较少步数和零引导
        if self.model_name == FluxModelType.FLUX_SCHNELL:
            if self.steps > 20:
                self.steps = 4  # 默认 schnell 步数
            if self.guidance_scale > 0:
                self.guidance_scale = 0.0  # schnell 不使用 guidance
        return self

class Text2ImageParams(BaseFluxParams):
    """文本生图参数"""
    pass

def generate_text2image(params: Text2ImageParams) -> str:
    """
    基于 Flux 模型的文本生图任务，生成并保存一张图片，返回图片文件路径
    """
    # 1. 设置随机种子
    if params.seed >= 0:
        import random, numpy as np, torch
        random.seed(params.seed)
        np.random.seed(params.seed)
        torch.manual_seed(params.seed)

    # 2. 查找模型文件路径
    model_filename = find_flux_model_file(params.model_name.value, quantization="int8")
    
    # 3. 加载指定名称的 Flux 模型
    flux_model = model_factory(
        checkpoint_dir="ckpts",
        model_filename=model_filename,
        model_type="flux",
        model_def={"flux-model": params.model_name.value},
        base_model_type="flux"
    )

    # 4. 调用模型生成图像张量，并转换为 PIL.Image
    image_tensor = flux_model.generate(
        seed=params.seed if params.seed >= 0 else None,
        input_prompt=params.prompt,
        sampling_steps=params.steps,
        width=params.width,
        height=params.height,
        embedded_guidance_scale=params.guidance_scale,
        loras_slists=params.loras_slists,
        batch_size=params.batch_size
    )
    
    if image_tensor is None:
        raise RuntimeError("Generation was interrupted or failed")
        
    image = convert_tensor_to_image(image_tensor)

    # 5. 保存图片并返回路径
    ts = int(time.time() * 1000)
    filename = f"flux_{params.model_name.value}_{ts}.png"
    out_path = os.path.join(params.output_dir, filename)
    image.save(out_path)
    return out_path

class Image2ImageParams(BaseFluxParams):
    """图生图参数 - 仅适用于 flux-dev-kontext"""
    model_name: FluxModelType = Field(FluxModelType.FLUX_DEV_KONTEXT, description="图生图建议使用 kontext 模型")
    input_image_path: str = Field(..., description="输入参考图像路径")
    fit_into_canvas: bool = Field(True, description="是否调整图像尺寸适应画布")

    @field_validator("input_image_path")
    @classmethod
    def valid_image_path(cls, v):
        if not os.path.isfile(v):
            raise ValueError(f"Input image not found: {v}")
        return v

    @field_validator("model_name")
    @classmethod
    def validate_kontext_for_img2img(cls, v):
        if v != FluxModelType.FLUX_DEV_KONTEXT:
            raise ValueError("Image-to-image generation requires flux-dev-kontext model")
        return v

class ControlNetParams(BaseFluxParams):
    """ControlNet 控制生成参数"""
    control_type: ControlType = Field(ControlType.DEPTH, description="控制类型")
    control_image_path: str = Field(..., description="控制图像路径")
    control_strength: float = Field(1.0, ge=0, le=2, description="控制强度")

    @field_validator("control_image_path")
    @classmethod
    def valid_control_image_path(cls, v):
        if not os.path.isfile(v):
            raise ValueError(f"Control image not found: {v}")
        return v

class InpaintParams(BaseFluxParams):
    """填充（修复）参数"""
    input_image_path: str = Field(..., description="输入图像路径")
    mask_image_path: str = Field(..., description="遮罩图像路径")
    
    @field_validator("input_image_path", "mask_image_path")
    @classmethod
    def valid_paths(cls, v):
        if not os.path.isfile(v):
            raise ValueError(f"File not found: {v}")
        return v

def generate_image2image(params: Image2ImageParams) -> str:
    """
    基于 Flux Kontext 模型的图生图任务，使用输入图像和编辑指令生成新图像
    """
    # 1. 设置随机种子
    if params.seed >= 0:
        import random, numpy as np, torch
        random.seed(params.seed)
        np.random.seed(params.seed)
        torch.manual_seed(params.seed)

    # 2. 查找模型文件路径
    model_filename = find_flux_model_file(params.model_name.value, quantization="int8")

    # 3. 加载 Flux Kontext 模型
    flux_model = model_factory(
        checkpoint_dir="ckpts",
        model_filename=model_filename,
        model_type="flux",
        model_def={"flux-model": params.model_name.value},
        base_model_type="flux"
    )

    # 4. 打开参考图像
    ref_img = Image.open(params.input_image_path).convert("RGB")
    
    # 5. 调用模型生成图像张量
    image_tensor = flux_model.generate(
        seed=params.seed if params.seed >= 0 else None,
        input_prompt=params.prompt,
        sampling_steps=params.steps,
        input_ref_images=[ref_img],
        width=params.width,
        height=params.height,
        embedded_guidance_scale=params.guidance_scale,
        fit_into_canvas=params.fit_into_canvas,
        loras_slists=params.loras_slists,
        batch_size=params.batch_size
    )
    
    if image_tensor is None:
        raise RuntimeError("Generation was interrupted or failed")
        
    image = convert_tensor_to_image(image_tensor)

    # 6. 保存图片并返回路径
    ts = int(time.time() * 1000)
    filename = f"flux_kontext_{ts}.png"
    out_path = os.path.join(params.output_dir, filename)
    image.save(out_path)
    return out_path

def generate_controlnet(params: ControlNetParams) -> str:
    """
    基于 ControlNet 控制的图像生成任务
    """
    # 1. 设置随机种子
    if params.seed >= 0:
        import random, numpy as np, torch
        random.seed(params.seed)
        np.random.seed(params.seed)
        torch.manual_seed(params.seed)

    # 2. 查找模型文件路径
    model_filename = find_flux_model_file(params.model_name.value, quantization="int8")

    # 3. 加载 Flux 模型
    flux_model = model_factory(
        checkpoint_dir="ckpts",
        model_filename=model_filename,
        model_type="flux",
        model_def={"flux-model": params.model_name.value},
        base_model_type="flux"
    )

    # 4. 准备控制参数
    control_kwargs = {}
    if params.control_type != ControlType.NONE:
        control_kwargs["control_image_path"] = params.control_image_path
        control_kwargs["control_type"] = params.control_type.value
        control_kwargs["control_strength"] = params.control_strength

    # 5. 生成图像
    image_tensor = flux_model.generate(
        seed=params.seed if params.seed >= 0 else None,
        input_prompt=params.prompt,
        sampling_steps=params.steps,
        width=params.width,
        height=params.height,
        embedded_guidance_scale=params.guidance_scale,
        loras_slists=params.loras_slists,
        batch_size=params.batch_size,
        **control_kwargs
    )
    
    if image_tensor is None:
        raise RuntimeError("Generation was interrupted or failed")
        
    image = convert_tensor_to_image(image_tensor)

    # 6. 保存图片
    ts = int(time.time() * 1000)
    filename = f"flux_control_{params.control_type.value}_{ts}.png"
    out_path = os.path.join(params.output_dir, filename)
    image.save(out_path)
    return out_path

def generate_inpaint(params: InpaintParams) -> str:
    """
    基于 Flux 的图像修复任务
    """
    # 1. 设置随机种子
    if params.seed >= 0:
        import random, numpy as np, torch
        random.seed(params.seed)
        np.random.seed(params.seed)
        torch.manual_seed(params.seed)

    # 2. 查找模型文件路径
    model_filename = find_flux_model_file(params.model_name.value, quantization="int8")

    # 3. 加载 Flux 模型
    flux_model = model_factory(
        checkpoint_dir="ckpts",
        model_filename=model_filename,
        model_type="flux",
        model_def={"flux-model": params.model_name.value},
        base_model_type="flux"
    )

    # 4. 生成图像
    image_tensor = flux_model.generate(
        seed=params.seed if params.seed >= 0 else None,
        input_prompt=params.prompt,
        sampling_steps=params.steps,
        width=params.width,
        height=params.height,
        embedded_guidance_scale=params.guidance_scale,
        inpaint_image_path=params.input_image_path,
        inpaint_mask_path=params.mask_image_path,
        loras_slists=params.loras_slists,
        batch_size=params.batch_size
    )
    
    if image_tensor is None:
        raise RuntimeError("Generation was interrupted or failed")
        
    image = convert_tensor_to_image(image_tensor)

    # 5. 保存图片
    ts = int(time.time() * 1000)
    filename = f"flux_inpaint_{ts}.png"
    out_path = os.path.join(params.output_dir, filename)
    image.save(out_path)
    return out_path

# 使用示例函数
def example_usage():
    """使用示例"""
    # 1. 文本生图
    t2i_params = Text2ImageParams(
        model_name=FluxModelType.FLUX_DEV,
        prompt="A beautiful landscape with mountains and lake",
        width=1024,
        height=768,
        seed=42,
        steps=50
    )
    result_path = generate_text2image(t2i_params)
    print(f"Generated image: {result_path}")
    
    # 2. 图像编辑 (Kontext)
    i2i_params = Image2ImageParams(
        input_image_path="input.jpg",
        prompt="add a rainbow in the sky",
        width=1024,
        height=768,
        seed=42
    )
    result_path = generate_image2image(i2i_params)
    print(f"Edited image: {result_path}")
    
    # 3. 深度控制生成
    control_params = ControlNetParams(
        control_type=ControlType.DEPTH,
        control_image_path="depth_map.jpg",
        prompt="A futuristic city with neon lights",
        width=1024,
        height=768
    )
    result_path = generate_controlnet(control_params)
    print(f"Controlled image: {result_path}")