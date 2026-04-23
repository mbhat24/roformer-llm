"""
Configuration management for RoFormer LLM
"""

import yaml
import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    vocab_size: int = 5000
    embed_dim: int = 512
    num_heads: int = 8
    num_layers: int = 8
    ff_dim: int = 2048
    max_position_embeddings: int = 512
    dropout: float = 0.1
    pad_token_id: int = 0
    use_yarn: bool = False  # Enable YaRN scaling for better extrapolation
    yarn_alpha: float = 1.0  # YaRN alpha parameter
    yarn_beta: float = 0.1  # YaRN beta parameter
    
    def __post_init__(self):
        """Ensure numeric types are correct"""
        if isinstance(self.dropout, str):
            self.dropout = float(self.dropout)
        if isinstance(self.yarn_alpha, str):
            self.yarn_alpha = float(self.yarn_alpha)
        if isinstance(self.yarn_beta, str):
            self.yarn_beta = float(self.yarn_beta)


@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 16
    num_epochs: int = 50
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    fp16: bool = False
    bf16: bool = False
    
    def __post_init__(self):
        """Ensure numeric types are correct"""
        if isinstance(self.learning_rate, str):
            self.learning_rate = float(self.learning_rate)
        if isinstance(self.weight_decay, str):
            self.weight_decay = float(self.weight_decay)
        if isinstance(self.max_grad_norm, str):
            self.max_grad_norm = float(self.max_grad_norm)
        if isinstance(self.fp16, str):
            self.fp16 = self.fp16.lower() in ('true', '1', 'yes')
        if isinstance(self.bf16, str):
            self.bf16 = self.bf16.lower() in ('true', '1', 'yes')


@dataclass
class DataConfig:
    """Data configuration"""
    data_path: str = "data/raw/shakespeare.txt"
    max_length: int = 256
    train_split: float = 0.9
    val_split: float = 0.05
    test_split: float = 0.05
    shuffle: bool = True
    num_workers: int = 4
    
    def __post_init__(self):
        """Ensure numeric types are correct"""
        if isinstance(self.train_split, str):
            self.train_split = float(self.train_split)
        if isinstance(self.val_split, str):
            self.val_split = float(self.val_split)
        if isinstance(self.test_split, str):
            self.test_split = float(self.test_split)


@dataclass
class TokenizerConfig:
    """Tokenizer configuration"""
    vocab_size: int = 5000
    special_tokens: Dict[str, str] = field(default_factory=lambda: {
        "pad": "<pad>",
        "unk": "<unk>",
        "bos": "<bos>",
        "eos": "<eos>"
    })


@dataclass
class LoggingConfig:
    """Logging configuration"""
    log_interval: int = 10
    eval_interval: int = 500
    save_interval: int = 1000
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    tensorboard: bool = True
    wandb: bool = False


@dataclass
class HardwareConfig:
    """Hardware configuration"""
    device: str = "auto"  # auto, cuda, mps, cpu
    distributed: bool = False
    num_gpus: int = 1


@dataclass
class OptimizerConfig:
    """Optimizer configuration"""
    type: str = "adamw"
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8
    
    def __post_init__(self):
        """Ensure numeric types are correct"""
        if isinstance(self.eps, str):
            self.eps = float(self.eps)
        if isinstance(self.betas, list):
            self.betas = [float(b) if isinstance(b, str) else b for b in self.betas]


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration"""
    type: str = "cosine"  # cosine, linear, constant
    warmup_ratio: float = 0.1
    min_lr: float = 1e-6
    
    def __post_init__(self):
        """Ensure numeric types are correct"""
        if isinstance(self.warmup_ratio, str):
            self.warmup_ratio = float(self.warmup_ratio)
        if isinstance(self.min_lr, str):
            self.min_lr = float(self.min_lr)


@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    metrics: List[str] = field(default_factory=lambda: ["perplexity", "generation_quality", "coherence"])
    generation: Dict[str, Any] = field(default_factory=lambda: {
        "max_new_tokens": 50,
        "temperature": 0.8,
        "top_k": 50,
        "num_samples": 5
    })


@dataclass
class Config:
    """Main configuration class"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create configuration from dictionary"""
        config = cls()
        
        if "model" in config_dict:
            config.model = ModelConfig(**config_dict["model"])
        if "training" in config_dict:
            config.training = TrainingConfig(**config_dict["training"])
        if "data" in config_dict:
            config.data = DataConfig(**config_dict["data"])
        if "tokenizer" in config_dict:
            config.tokenizer = TokenizerConfig(**config_dict["tokenizer"])
        if "logging" in config_dict:
            config.logging = LoggingConfig(**config_dict["logging"])
        if "hardware" in config_dict:
            config.hardware = HardwareConfig(**config_dict["hardware"])
        if "optimizer" in config_dict:
            config.optimizer = OptimizerConfig(**config_dict["optimizer"])
        if "scheduler" in config_dict:
            config.scheduler = SchedulerConfig(**config_dict["scheduler"])
        if "evaluation" in config_dict:
            config.evaluation = EvaluationConfig(**config_dict["evaluation"])
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "data": self.data.__dict__,
            "tokenizer": self.tokenizer.__dict__,
            "logging": self.logging.__dict__,
            "hardware": self.hardware.__dict__,
            "optimizer": self.optimizer.__dict__,
            "scheduler": self.scheduler.__dict__,
            "evaluation": self.evaluation.__dict__
        }
    
    def save_yaml(self, save_path: str):
        """Save configuration to YAML file"""
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        with open(save_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
