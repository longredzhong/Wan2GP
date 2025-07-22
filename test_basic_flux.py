#!/usr/bin/env python3
"""
简化版 Flux 文本生图测试
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tasks.base_flux_task import Text2ImageParams, FluxModelType, generate_text2image

def test_basic_generation():
    """测试基础文本生图功能"""
    print("开始测试 Flux 基础文本生图功能...")
    
    # 创建参数
    params = Text2ImageParams(
        model_name=FluxModelType.FLUX_DEV,
        prompt="A serene mountain lake with morning mist",
        width=512,  # 较小尺寸用于测试
        height=512,
        seed=42,
        steps=20,  # 较少步数用于测试
        guidance_scale=3.5,
        output_dir="outputs/test"
    )
    
    print(f"参数验证: {params}")
    
    try:
        # 生成图像
        result_path = generate_text2image(params)
        print(f"✅ 生成成功! 图像保存在: {result_path}")
        return True
        
    except FileNotFoundError as e:
        print(f"❌ 模型文件未找到: {e}")
        return False
    except ImportError as e:
        print(f"❌ 依赖缺失: {e}")
        return False
    except Exception as e:
        print(f"❌ 生成失败: {e}")
        return False

def test_parameter_validation():
    """测试参数验证"""
    print("\n测试参数验证...")
    
    # 测试空提示词
    try:
        params = Text2ImageParams(
            prompt="", 
            model_name=FluxModelType.FLUX_DEV,
            width=512,
            height=512,
            seed=42,
            steps=20,
            guidance_scale=3.5,
            output_dir="outputs"
        )
        print("❌ 空提示词验证失败")
        return False
    except ValueError:
        print("✅ 空提示词验证通过")
    
    # 测试正常参数
    try:
        params = Text2ImageParams(
            prompt="test prompt",
            model_name=FluxModelType.FLUX_DEV,
            width=1024,
            height=768,
            seed=42,
            steps=20,
            guidance_scale=3.5,
            output_dir="outputs"
        )
        print("✅ 正常参数验证通过")
        print(f"模型: {params.model_name}, 尺寸: {params.width}x{params.height}")
        return True
    except Exception as e:
        print(f"❌ 正常参数验证失败: {e}")
        return False

if __name__ == "__main__":
    print("简化版 Flux 测试")
    print("=" * 50)
    
    # 参数验证测试
    if not test_parameter_validation():
        sys.exit(1)
    
    # 基础功能测试
    if not test_basic_generation():
        print("\n提示: 确保模型文件存在于 ckpts/ 目录中")
        sys.exit(1)
    
    print("\n🎉 所有测试通过!")
