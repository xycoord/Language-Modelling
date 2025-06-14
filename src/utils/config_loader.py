from dataclasses import dataclass, asdict
from typing import Dict, Any, Union, Optional
from datetime import datetime
import uuid
import yaml
from functools import reduce

from models import BigramConfig, TransformerConfig

ModelConfig = Union[BigramConfig, TransformerConfig]

@dataclass
class Config:
    """Central configuration manager for training experiments.
    
    Supports YAML-based config inheritance to avoid duplication across experiments.
    Configs can inherit from multiple parent configs, with later parents and local
    values overriding earlier ones. This allows sharing common settings (e.g., base
    model architecture) while customising specific parameters.
    
    The config system separates model-specific parameters into typed dataclasses
    (BigramConfig, TransformerConfig) for validation and type safety, while keeping
    all other parameters flat for easy logging and experiment tracking.
    """

    # Experiment metadata
    experiment_name: str
    run_id: str
    
    # Model type and config
    model_type: str 
    model_config: Dict[str, Any]
    
    # Training
    seed: int
    learning_rate: float
    batch_size: int
    block_size: int
    vocab_size: int
    epochs: int
    eval_interval: int
    example_interval: int
    train_split: float
    
    # System
    data_dir: str
    output_dir: str

    # Optional (with defaults)
    max_train_steps: Optional[int] = None
    compile_model: bool = False
    mixed_precision: bool = False

    @classmethod
    def from_file(cls, config_path: str, overrides: Optional[Dict[str, Any]] = None) -> 'Config':
        """Load config with inheritance support
        - apply each inherited config in order
        - later configs override earlier configs
        - local config overrides inherited configs
        - apply overrides last
        - generate run_id if not provided
        """
        config = cls._load_config_dict_with_inheritance(config_path)
        
        if overrides:
            config = cls._merge_config_dicts(config, overrides)
        
        if 'run_id' not in config:
            config['run_id'] = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"

        return cls(**config)
    
    def __post_init__(self):
        self.model_config_typed = self._create_model_config()
        self.flat_config = self._create_flat_config()
    
    def _create_model_config(self) -> ModelConfig:
        """Create the appropriate model config object"""
        model_params = self.model_config.copy()
        
        if self.model_type == "transformer":
            model_params['vocab_size'] = self.vocab_size
            model_params['block_size'] = self.block_size
            return TransformerConfig(**model_params)
        elif self.model_type == "bigram":
            model_params['vocab_size'] = self.vocab_size
            return BigramConfig(**model_params)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _create_flat_config(self) -> Dict[str, Any]:
        """Flatten config for logging"""
        flat_config = asdict(self)

        del flat_config['model_config']
        for key, value in asdict(self.model_config_typed).items():
            flat_config[f"model_{key}"] = value
            
        return flat_config

    @staticmethod
    def _merge_config_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge configs, handling nested model_config properly"""
        result = base.copy()
        
        for key, value in override.items():
            if key == 'model_config' and key in result and isinstance(result[key], dict):
                # model_config needs deep merge to preserve nested params from parent configs
                result[key] = {**result[key], **value}
            else:
                result[key] = value
        
        return result

    @staticmethod
    def _load_config_dict_with_inheritance(config_path: str) -> Dict[str, Any]:
        """Load config as dict with inheritance support (helper function)"""
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {config_path}: {e}")
        
        if 'inherits' not in config:
            return config
        
        inherited_configs = [Config._load_config_dict_with_inheritance(path) 
                             for path in config['inherits']]
        
        local_config = config.copy()
        local_config.pop('inherits', None)

        merged_config = reduce(Config._merge_config_dicts, inherited_configs + [local_config], {})
        
        return merged_config