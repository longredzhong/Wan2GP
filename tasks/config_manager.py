"""
WanGP 配置管理模块
处理模型定义、任务配置等
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
import logging

from .utils import parse_resolution, sanitize_file_name

logger = logging.getLogger(__name__)


@dataclass
class ModelDefinition:
    """模型定义数据类"""
    name: str
    architecture: str
    description: str = ""
    urls: List[str] = field(default_factory=list)
    preload_urls: List[str] = field(default_factory=list)
    modules: List[str] = field(default_factory=list)
    image_outputs: bool = False
    flux_model: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelDefinition":
        """从字典创建模型定义"""
        model_info = data.get("model", {})
        
        urls = model_info.get("URLs", [])
        if isinstance(urls, str):
            urls = [urls]
        elif urls is None:
            urls = []
            
        preload_urls = model_info.get("preload_URLs", [])
        if isinstance(preload_urls, str):
            preload_urls = [preload_urls]
        elif preload_urls is None:
            preload_urls = []
            
        modules = model_info.get("modules", [])
        if isinstance(modules, str):
            modules = [modules]
        elif modules is None:
            modules = []
        
        return cls(
            name=model_info.get("name", ""),
            architecture=model_info.get("architecture", ""),
            description=model_info.get("description", ""),
            urls=urls,
            preload_urls=preload_urls,
            modules=modules,
            image_outputs=model_info.get("image_outputs", False),
            flux_model=model_info.get("flux-model")
        )


@dataclass 
class TaskConfig:
    """任务配置数据类"""
    model_type: str
    prompt: str
    negative_prompt: str = ""
    resolution: str = "720x480"
    video_length: int = 49
    steps: int = 20
    guidance_scale: float = 7.0
    fps: int = 24
    seed: Optional[int] = None
    output_path: str = "outputs"
    image_path: Optional[str] = None
    
    def __post_init__(self):
        """验证配置"""
        self._validate()
    
    def _validate(self):
        """验证配置有效性"""
        if not self.model_type:
            raise ValueError("model_type 不能为空")
        if not self.prompt:
            raise ValueError("prompt 不能为空")
        
        # 验证分辨率
        try:
            parse_resolution(self.resolution)
        except ValueError as e:
            raise ValueError(f"分辨率格式错误: {e}")
        
        # 验证数值范围
        if self.video_length <= 0:
            raise ValueError("video_length 必须大于0")
        if self.steps <= 0:
            raise ValueError("steps 必须大于0")
        if self.guidance_scale < 0:
            raise ValueError("guidance_scale 不能小于0")
        if self.fps <= 0:
            raise ValueError("fps 必须大于0")
    
    @property
    def width(self) -> int:
        """获取宽度"""
        return parse_resolution(self.resolution)[0]
    
    @property
    def height(self) -> int:
        """获取高度"""
        return parse_resolution(self.resolution)[1]
    
    def get_generation_params(self) -> Dict[str, Any]:
        """获取生成参数字典"""
        params = {
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "num_inference_steps": self.steps,
            "guidance_scale": self.guidance_scale,
            "width": self.width,
            "height": self.height,
            "num_frames": self.video_length,
            "fps": self.fps,
        }
        
        if self.seed is not None:
            params["seed"] = self.seed
            
        if self.image_path:
            params["image"] = self.image_path
            
        return params
    
    def get_output_filename(self, extension: str = "mp4") -> str:
        """生成输出文件名"""
        safe_prompt = sanitize_file_name(self.prompt[:50])
        safe_model = sanitize_file_name(self.model_type)
        
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        return f"{safe_prompt}_{safe_model}_{timestamp}.{extension}"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskConfig":
        """从字典创建任务配置"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, defaults_dir: str = "defaults", configs_dir: str = "configs"):
        self.defaults_dir = Path(defaults_dir)
        self.configs_dir = Path(configs_dir) 
        self._model_definitions: Dict[str, ModelDefinition] = {}
        self._load_all_definitions()
    
    def _load_all_definitions(self):
        """加载所有模型定义"""
        logger.info(f"加载模型定义从: {self.defaults_dir}")
        
        if not self.defaults_dir.exists():
            logger.warning(f"默认配置目录不存在: {self.defaults_dir}")
            return
            
        # 加载所有 JSON 文件
        json_files = list(self.defaults_dir.glob("*.json"))
        logger.info(f"发现 {len(json_files)} 个配置文件")
        
        for json_file in json_files:
            try:
                model_type = json_file.stem
                definition = self._load_definition(json_file)
                if definition:
                    self._model_definitions[model_type] = definition
                    logger.debug(f"加载模型定义: {model_type}")
            except Exception as e:
                logger.error(f"加载 {json_file} 失败: {e}")
        
        logger.info(f"成功加载 {len(self._model_definitions)} 个模型定义")
    
    def _load_definition(self, json_file: Path) -> Optional[ModelDefinition]:
        """加载单个模型定义文件"""
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # 跳过 ReadMe.txt 等非模型配置文件
            if "model" not in data:
                return None
                
            return ModelDefinition.from_dict(data)
            
        except Exception as e:
            logger.error(f"解析 {json_file} 失败: {e}")
            return None
    
    def get_model_definition(self, model_type: str) -> Optional[ModelDefinition]:
        """获取模型定义"""
        return self._model_definitions.get(model_type)
    
    def get_all_model_types(self) -> List[str]:
        """获取所有可用的模型类型"""
        return list(self._model_definitions.keys())
    
    def get_models_by_family(self, family: str) -> List[str]:
        """按模型家族获取模型列表"""
        models = []
        for model_type, definition in self._model_definitions.items():
            model_family = self.get_model_family(definition.architecture)
            if model_family == family:
                models.append(model_type)
        return models
    
    @staticmethod
    def get_model_family(architecture: str) -> str:
        """根据架构确定模型家族"""
        if "hunyuan" in architecture.lower():
            return "hunyuan"
        elif "ltxv" in architecture.lower():
            return "ltxv" 
        elif "flux" in architecture.lower():
            return "flux"
        else:
            return "wan"
    
    def get_base_model_type(self, model_type: str) -> str:
        """获取基础模型类型"""
        definition = self.get_model_definition(model_type)
        if definition and definition.architecture:
            return definition.architecture
        
        # 如果没有定义，检查是否是已知的模型类型
        known_types = {
            "multitalk", "fantasy", "vace_14B", "vace_multitalk_14B",
            "t2v_1.3B", "t2v", "vace_1.3B", "phantom_1.3B", "phantom_14B",
            "recam_1.3B", "sky_df_1.3B", "sky_df_14B", "i2v", "flf2v_720p",
            "fun_inp_1.3B", "fun_inp", "ltxv_13B", "hunyuan", "hunyuan_i2v",
            "hunyuan_custom", "hunyuan_custom_audio", "hunyuan_custom_edit",
            "hunyuan_avatar", "flux"
        }
        
        return model_type if model_type in known_types else model_type
    
    def is_i2v_model(self, model_type: str) -> bool:
        """检查是否为图像到视频模型"""
        base_type = self.get_base_model_type(model_type)
        i2v_types = {
            "i2v", "fun_inp_1.3B", "fun_inp", "flf2v_720p", 
            "fantasy", "multitalk", "hunyuan_i2v"
        }
        return base_type in i2v_types
    
    def resolve_model_urls(self, model_type: str) -> List[str]:
        """解析模型的下载URLs，处理引用"""
        definition = self.get_model_definition(model_type)
        if not definition:
            return []
        
        urls = definition.urls
        
        # 如果URLs是引用其他模型类型
        if len(urls) == 1 and not urls[0].startswith("http"):
            referenced_type = urls[0]
            return self.resolve_model_urls(referenced_type)
        
        return urls
    
    def load_task_config(self, config_path: str) -> Union[TaskConfig, List[TaskConfig]]:
        """从文件加载任务配置"""
        config_file = Path(config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(config_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # 检查是否是批处理配置
        if "tasks" in data:
            # 批处理配置
            tasks = []
            base_output = data.get("output_path", "outputs")
            
            for task_data in data["tasks"]:
                if "output_path" not in task_data:
                    task_data["output_path"] = base_output
                tasks.append(TaskConfig.from_dict(task_data))
            
            return tasks
        else:
            # 单个任务配置
            return TaskConfig.from_dict(data)
    
    def save_task_config(self, config: TaskConfig, config_path: str):
        """保存任务配置到文件"""
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 转换为字典
        config_dict = {
            field.name: getattr(config, field.name)
            for field in config.__dataclass_fields__.values()
        }
        
        # 移除None值
        config_dict = {k: v for k, v in config_dict.items() if v is not None}
        
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    def create_sample_configs(self, output_dir: str = "configs"):
        """创建示例配置文件"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 示例配置
        configs = {
            "text2video.json": TaskConfig(
                model_type="t2v",
                prompt="A majestic eagle soaring through snow-capped mountains at sunset",
                negative_prompt="blurry, low quality, distorted, ugly",
                resolution="720x480",
                video_length=49,
                steps=20,
                guidance_scale=7.0,
                fps=24,
                seed=42,
                output_path="outputs/text2video"
            ),
            "image2video.json": TaskConfig(
                model_type="i2v", 
                prompt="The person in the image starts walking forward",
                image_path="inputs/reference.jpg",
                resolution="720x480",
                video_length=49,
                steps=25,
                guidance_scale=7.5,
                fps=24,
                output_path="outputs/image2video"
            ),
            "hunyuan_video.json": TaskConfig(
                model_type="hunyuan",
                prompt="A cat playing with a ball of yarn in slow motion",
                resolution="720x480", 
                video_length=61,
                steps=30,
                guidance_scale=6.0,
                fps=24,
                output_path="outputs/hunyuan"
            )
        }
        
        # 批处理配置
        batch_config = {
            "tasks": [
                {
                    "model_type": "t2v",
                    "prompt": "A peaceful lake at dawn with mist rising", 
                    "resolution": "720x480",
                    "steps": 20
                },
                {
                    "model_type": "t2v",
                    "prompt": "City traffic at night with neon lights",
                    "resolution": "720x480", 
                    "steps": 20
                }
            ],
            "output_path": "outputs/batch"
        }
        
        # 保存示例配置
        for filename, config in configs.items():
            self.save_task_config(config, output_path / filename)
            logger.info(f"创建示例配置: {output_path / filename}")
        
        # 保存批处理配置
        batch_file = output_path / "batch_config.json"
        with open(batch_file, "w", encoding="utf-8") as f:
            json.dump(batch_config, f, indent=2, ensure_ascii=False)
        logger.info(f"创建批处理配置: {batch_file}")
    
    def validate_model_type(self, model_type: str) -> bool:
        """验证模型类型是否有效"""
        return model_type in self._model_definitions
    
    def get_model_info(self, model_type: str) -> Dict[str, Any]:
        """获取模型详细信息"""
        definition = self.get_model_definition(model_type)
        if not definition:
            return {}
        
        return {
            "name": definition.name,
            "architecture": definition.architecture,
            "description": definition.description,
            "family": self.get_model_family(definition.architecture),
            "urls_count": len(definition.urls),
            "has_modules": len(definition.modules) > 0,
            "modules": definition.modules,
            "is_i2v": self.is_i2v_model(model_type),
            "image_outputs": definition.image_outputs,
            "flux_model": definition.flux_model
        }
