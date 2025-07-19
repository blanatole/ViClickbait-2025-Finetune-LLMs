"""
Configuration file for clickbait detection experiments
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class ModelConfig:
    """Configuration for model settings"""
    name: str
    max_length: int = 512
    max_new_tokens: int = 10
    temperature: float = 0.1
    do_sample: bool = False
    torch_dtype: str = "float16"
    device_map: str = "auto"
    trust_remote_code: bool = True

@dataclass
class LoRAConfig:
    """Configuration for LoRA fine-tuning"""
    r: int = 16
    lora_alpha: int = 32
    target_modules: List[str] = None
    lora_dropout: float = 0.1
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

@dataclass
class TrainingConfig:
    """Configuration for training"""
    output_dir: str = "./results/models"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-4
    weight_decay: float = 0.001
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    warmup_steps: int = 100
    max_grad_norm: float = 0.3
    group_by_length: bool = True
    lr_scheduler_type: str = "cosine"
    optim: str = "paged_adamw_32bit"

@dataclass
class DataConfig:
    """Configuration for data"""
    data_dir: str = "data/simple_dataset"
    train_file: str = "train/train.json"
    val_file: str = "val/val.json"
    test_file: str = "test/test.json"
    max_samples: int = None  # None for all samples

@dataclass
class ExperimentConfig:
    """Main experiment configuration"""
    # Paths
    project_root: str = os.getcwd()
    results_dir: str = "results/experiments"
    logs_dir: str = "results/logs"
    
    # Models to test
    models: List[str] = None
    
    # Experiment settings
    random_seed: int = 42
    few_shot_examples: int = 3
    run_prompting: bool = True
    run_finetuning: bool = True
    
    # Hardware
    use_cuda: bool = True
    mixed_precision: str = "fp16"
    
    def __post_init__(self):
        if self.models is None:
            self.models = [
                "meta-llama/Llama-3.1-8B-Instruct",
                "google/gemma-7b-it"
            ]
        
        # Create directories if they don't exist
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

# Default configurations
DEFAULT_MODEL_CONFIG = ModelConfig(name="")
DEFAULT_LORA_CONFIG = LoRAConfig(target_modules=["q_proj", "v_proj"])
DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_DATA_CONFIG = DataConfig()
DEFAULT_EXPERIMENT_CONFIG = ExperimentConfig()

# Model-specific configurations
MODEL_CONFIGS = {
    "meta-llama/Llama-3.1-8B-Instruct": ModelConfig(
        name="meta-llama/Llama-3.1-8B-Instruct",
        max_length=512,
        max_new_tokens=10,
        temperature=0.1
    ),
    "google/gemma-7b-it": ModelConfig(
        name="google/gemma-7b-it",
        max_length=512,
        max_new_tokens=10,
        temperature=0.1
    )
}

def get_model_config(model_name: str) -> ModelConfig:
    """Get configuration for a specific model"""
    return MODEL_CONFIGS.get(model_name, DEFAULT_MODEL_CONFIG)
