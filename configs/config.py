"""
Centralized configuration for LoRA fine-tuning pipeline
"""
import torch
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ModelConfig:
    """Model configuration"""
    base_model: str = "google/gemma-2b"
    max_tokens: int = 128
    device: str = "auto"  # auto, cpu, cuda, mps
    
    def get_device(self):
        """Auto-detect best available device"""
        if self.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(self.device)

@dataclass
class DataConfig:
    """Dataset configuration"""
    dataset_name: str = "elenigkove/Email_Intent_Classification"
    intent_labels: List[str] = None
    test_emails: List[str] = None
    
    def __post_init__(self):
        if self.intent_labels is None:
            self.intent_labels = ["Request", "Informational", "Transaction", "Feedback"]
        
        if self.test_emails is None:
            self.test_emails = [
                "Can you please confirm the meeting time for tomorrow?",
                "We have processed your payment successfully.",
                "This is a gentle reminder to submit your feedback.",
                "Thank you for your email. I will get back to you shortly."
            ]

@dataclass
class LoRAConfig:
    """LoRA training configuration"""
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: List[str] = None
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj"]

@dataclass
class TrainingConfig:
    """Training configuration"""
    output_dir: str = "outputs"
    adapter_name: str = "lora_adapter"
    num_epochs: int = 2
    batch_size: int = 1
    learning_rate: float = 2e-4
    eval_steps: int = 100
    save_steps: int = 100
    logging_steps: int = 50

@dataclass
class GenerationConfig:
    """Generation configuration for inference"""
    max_new_tokens: int = 5
    temperature: float = 0.3
    top_k: int = 20
    do_sample: bool = True

# Global configuration instances
MODEL_CONFIG = ModelConfig()
DATA_CONFIG = DataConfig()
LORA_CONFIG = LoRAConfig()
TRAINING_CONFIG = TrainingConfig()
GENERATION_CONFIG = GenerationConfig()

# Legacy compatibility
BASE_MODEL = MODEL_CONFIG.base_model
MAX_TOKENS = MODEL_CONFIG.max_tokens
LABELS = DATA_CONFIG.intent_labels
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}
dummy_emails = DATA_CONFIG.test_emails