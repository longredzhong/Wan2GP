from pydantic import BaseModel, Field, field_validator, model_validator
import os
import time
import glob
from flux.flux_main import model_factory
from wan.utils.utils import convert_tensor_to_image
from typing import List
from enum import Enum

MODEL_ROOT_PATH = os.environ.get("MODEL_ROOT_PATH", "ckpts")

def find_flux_model_file(model_name: str = "flux-dev", quantization: str = "int8") -> List[str]:
    """查找 Flux 模型文件路径"""
    model_patterns = {
        "flux-dev": [
            f"{MODEL_ROOT_PATH}/flux1-dev_quanto_bf16_int8.safetensors",
            f"{MODEL_ROOT_PATH}/flux1-dev_bf16.safetensors",
            f"{MODEL_ROOT_PATH}/*flux*dev*int8*.safetensors",
            f"{MODEL_ROOT_PATH}/*flux*dev*bf16*.safetensors",
            f"{MODEL_ROOT_PATH}/*flux*dev*.safetensors",
        ],
        "flux-schnell": [
            f"{MODEL_ROOT_PATH}/flux1-schnell_quanto_bf16_int8.safetensors",
            f"{MODEL_ROOT_PATH}/flux1-schnell_bf16.safetensors",
            f"{MODEL_ROOT_PATH}/*flux*schnell*int8*.safetensors",
            f"{MODEL_ROOT_PATH}/*flux*schnell*bf16*.safetensors",
            f"{MODEL_ROOT_PATH}/*flux*schnell*.safetensors",
        ],
        "flux-dev-kontext": [
            f"{MODEL_ROOT_PATH}/flux1_kontext_dev_quanto_bf16_int8.safetensors",
            f"{MODEL_ROOT_PATH}/flux1_kontext_dev_bf16.safetensors",
            f"{MODEL_ROOT_PATH}/*flux*kontext*int8*.safetensors",
            f"{MODEL_ROOT_PATH}/*flux*kontext*bf16*.safetensors",
            f"{MODEL_ROOT_PATH}/*flux*kontext*.safetensors",
        ],
    }
    
    patterns = model_patterns.get(model_name, model_patterns["flux-dev"])
    
    # 查找存在的文件
    for pattern in patterns:
        if "*" in pattern:
            files = glob.glob(pattern)
            if files:
                return files[:1]
        else:
            if os.path.exists(pattern):
                return [pattern]
    
    # 备选方案
    fallback_patterns = [f"{MODEL_ROOT_PATH}/*flux*.safetensors"]
    for pattern in fallback_patterns:
        files = glob.glob(pattern)
        if files:
            print(f"Warning: Using fallback model {files[0]} for {model_name}")
            return files[:1]
    
    raise FileNotFoundError(f"No Flux model files found for {model_name}")

class FluxModelType(str, Enum):
    """Flux 模型类型"""
    FLUX_DEV = "flux-dev"
    FLUX_SCHNELL = "flux-schnell"
    FLUX_DEV_KONTEXT = "flux-dev-kontext"

class Text2ImageParams(BaseModel):
    """文本生图参数"""
    model_name: FluxModelType = Field(FluxModelType.FLUX_DEV, description="Flux 模型名称")
    prompt: str = Field(..., description="文本提示词")
    width: int = Field(1024, ge=64, le=2048, description="图像宽度")
    height: int = Field(1024, ge=64, le=2048, description="图像高度")
    seed: int = Field(-1, ge=-1, description="随机种子 (-1表示随机)")
    steps: int = Field(50, ge=1, le=150, description="推理步数")
    guidance_scale: float = Field(3.5, ge=0, le=20, description="引导系数")
    output_dir: str = Field("outputs", description="输出目录")

    @field_validator("prompt")
    @classmethod
    def non_empty_prompt(cls, v):
        if not v.strip():
            raise ValueError("提示词不能为空")
        return v

    @field_validator("output_dir")
    @classmethod
    def ensure_output_dir(cls, v):
        os.makedirs(v, exist_ok=True)
        return v

    @model_validator(mode="after")
    def validate_model_params(self):
        # flux-schnell 使用较少步数和零引导
        if self.model_name == FluxModelType.FLUX_SCHNELL:
            if self.steps > 20:
                self.steps = 4
            if self.guidance_scale > 0:
                self.guidance_scale = 0.0
        return self

def generate_text2image(params: Text2ImageParams) -> str:
    """
    文本生成图像
    
    Args:
        params: 生成参数
        
    Returns:
        生成图像的文件路径
    """
    # 设置随机种子
    if params.seed >= 0:
        import random
        import numpy as np
        import torch
        random.seed(params.seed)
        np.random.seed(params.seed)
        torch.manual_seed(params.seed)

    # 查找模型文件
    model_filename = find_flux_model_file(params.model_name.value, quantization="int8")
    print(f"使用模型文件: {model_filename}")
    # 加载模型
    flux_model = model_factory(
        checkpoint_dir="ckpts",
        model_filename=model_filename,
        model_type="flux",
        model_def={"flux-model": params.model_name.value},
        base_model_type="flux",
    )
    print(flux_model)
    # 生成图像
    image_tensor = flux_model.generate(
        seed=params.seed if params.seed >= 0 else None,
        input_prompt=params.prompt,
        sampling_steps=params.steps,
        width=params.width,
        height=params.height,
        embedded_guidance_scale=params.guidance_scale,
    )

    if image_tensor is None:
        raise RuntimeError("生成失败或被中断")

    image = convert_tensor_to_image(image_tensor)

    # 保存图像
    ts = int(time.time() * 1000)
    filename = f"flux_{params.model_name.value}_{ts}.png"
    out_path = os.path.join(params.output_dir, filename)
    image.save(out_path)
    return out_path

# 使用示例
def example_usage():
    """使用示例"""
    params = Text2ImageParams(
        model_name=FluxModelType.FLUX_DEV,
        prompt="A beautiful mountain landscape with clear lake reflection",
        width=1024,
        height=768,
        seed=42,
        steps=50,
        guidance_scale=3.5,
        output_dir="outputs"
    )
    
    result_path = generate_text2image(params)
    print(f"生成的图像保存在: {result_path}")

if __name__ == "__main__":
    example_usage()
