"""
Model handler for LoRA fine-tuning pipeline
Handles model loading, LoRA operations, and model saving
"""
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from typing import Tuple, Optional

from configs import MODEL_CONFIG, LORA_CONFIG, TRAINING_CONFIG, GENERATION_CONFIG


class ModelHandler:
    """Centralized model operations"""
    
    def __init__(self):
        self.device = MODEL_CONFIG.get_device()
        self.model = None
        self.tokenizer = None
        self.is_lora_model = False
    
    def load_base_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load the base model and tokenizer"""
        print(f"[INFO] Loading base model: {MODEL_CONFIG.base_model}")
        print(f"[INFO] Using device: {self.device}")
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_CONFIG.base_model,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            torch_dtype=torch.float32
        ).to(self.device)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_CONFIG.base_model, 
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"[INFO] Base model loaded successfully")
        return self.model, self.tokenizer
    
    def prepare_for_lora_training(self) -> AutoModelForCausalLM:
        """Prepare base model for LoRA training"""
        if self.model is None:
            self.load_base_model()
        
        print("[INFO] Preparing model for LoRA training...")
        
        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Create LoRA config
        lora_config = LoraConfig(
            r=LORA_CONFIG.rank,
            lora_alpha=LORA_CONFIG.alpha,
            target_modules=LORA_CONFIG.target_modules,
            lora_dropout=LORA_CONFIG.dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        self.is_lora_model = True
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
        
        return self.model
    
    def load_lora_adapter(self, adapter_path: Optional[str] = None) -> Tuple[PeftModel, AutoTokenizer]:
        """Load a trained LoRA adapter"""
        if adapter_path is None:
            adapter_path = os.path.join(TRAINING_CONFIG.output_dir, TRAINING_CONFIG.adapter_name)
        
        print(f"[INFO] Loading LoRA adapter from: {adapter_path}")
        
        # Load base model first
        if self.model is None:
            self.load_base_model()
        
        # Load LoRA adapter
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.is_lora_model = True
        
        print("[INFO] LoRA adapter loaded successfully")
        return self.model, self.tokenizer
    
    def save_lora_adapter(self, save_path: Optional[str] = None):
        """Save the trained LoRA adapter"""
        if not self.is_lora_model or self.model is None:
            raise ValueError("No LoRA model to save")
        
        if save_path is None:
            save_path = os.path.join(TRAINING_CONFIG.output_dir, TRAINING_CONFIG.adapter_name)
        
        print(f"[INFO] Saving LoRA adapter to: {save_path}")
        self.model.save_pretrained(save_path)
        print("[INFO] LoRA adapter saved successfully")
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response using the loaded model"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded")
        
        # Use generation config as defaults, allow override
        gen_config = {
            'max_new_tokens': GENERATION_CONFIG.max_new_tokens,
            'temperature': GENERATION_CONFIG.temperature,
            'top_k': GENERATION_CONFIG.top_k,
            'do_sample': GENERATION_CONFIG.do_sample,
            'eos_token_id': self.tokenizer.eos_token_id,
            'pad_token_id': self.tokenizer.eos_token_id,
        }
        gen_config.update(kwargs)
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_config)
        
        # Decode
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the new tokens (model's response)
        input_length = len(self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True))
        response = full_response[input_length:].strip()
        
        return response
    
    def get_model_info(self) -> dict:
        """Get information about the current model"""
        if self.model is None:
            return {"status": "No model loaded"}
        
        info = {
            "base_model": MODEL_CONFIG.base_model,
            "device": str(self.device),
            "is_lora_model": self.is_lora_model,
            "parameters": self.model.num_parameters() if hasattr(self.model, 'num_parameters') else "Unknown"
        }
        
        if self.is_lora_model and hasattr(self.model, 'print_trainable_parameters'):
            # Capture trainable parameters info
            import io
            import sys
            old_stdout = sys.stdout
            sys.stdout = buffer = io.StringIO()
            self.model.print_trainable_parameters()
            sys.stdout = old_stdout
            info["trainable_params"] = buffer.getvalue().strip()
        
        return info
