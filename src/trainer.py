"""
Trainer for LoRA fine-tuning pipeline
Handles the complete training process using the modular components
"""
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from .config import TRAINING_CONFIG
from .model_handler import ModelHandler
from .data_handler import DataHandler


class LoRATrainer:
    """Unified LoRA training pipeline"""
    
    def __init__(self):
        self.model_handler = ModelHandler()
        self.data_handler = DataHandler()
    
    def train(self, custom_config: dict = None) -> str:
        """
        Complete LoRA fine-tuning pipeline
        Returns path to saved adapter
        """
        print("="*60)
        print("STARTING LORA FINE-TUNING PIPELINE")
        print("="*60)
        
        # 1. Load base model and tokenizer
        print("\n[STEP 1] Loading base model and tokenizer...")
        model, tokenizer = self.model_handler.load_base_model()
        self.data_handler.set_tokenizer(tokenizer)
        
        # 2. Load and preprocess dataset
        print("\n[STEP 2] Loading and preprocessing dataset...")
        self.data_handler.load_dataset()
        train_dataset = self.data_handler.preprocess_for_training("train")
        eval_dataset = self.data_handler.preprocess_for_training("test")
        
        # 3. Prepare model for LoRA training
        print("\n[STEP 3] Preparing model for LoRA training...")
        model = self.model_handler.prepare_for_lora_training()
        
        # 4. Setup training configuration
        print("\n[STEP 4] Setting up training configuration...")
        config = {
            'output_dir': TRAINING_CONFIG.output_dir,
            'per_device_train_batch_size': TRAINING_CONFIG.batch_size,
            'per_device_eval_batch_size': TRAINING_CONFIG.batch_size,
            'eval_strategy': "steps",
            'save_strategy': "steps",
            'eval_steps': TRAINING_CONFIG.eval_steps,
            'save_steps': TRAINING_CONFIG.save_steps,
            'logging_steps': TRAINING_CONFIG.logging_steps,
            'num_train_epochs': TRAINING_CONFIG.num_epochs,
            'learning_rate': TRAINING_CONFIG.learning_rate,
            'fp16': False,
            'bf16': False,
            'save_total_limit': 2,
            'load_best_model_at_end': True,
            'report_to': "none",
        }
        
        # Apply custom configuration if provided
        if custom_config:
            config.update(custom_config)
            print(f"[INFO] Applied custom configuration: {custom_config}")
        
        training_args = TrainingArguments(**config)
        
        # 5. Setup data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, 
            mlm=False
        )
        
        # 6. Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
        
        # 7. Start training
        print("\n[STEP 5] Starting LoRA training...")
        print("-" * 40)
        
        import time
        start_time = time.time()
        
        trainer.train()
        
        training_time = time.time() - start_time
        print("-" * 40)
        print(f"[INFO] Training completed in {training_time:.2f} seconds")
        
        # 8. Save LoRA adapter
        print("\n[STEP 6] Saving LoRA adapter...")
        adapter_path = f"{TRAINING_CONFIG.output_dir}/{TRAINING_CONFIG.adapter_name}"
        self.model_handler.save_lora_adapter(adapter_path)
        
        print("\n" + "="*60)
        print("LORA FINE-TUNING COMPLETED SUCCESSFULLY!")
        print(f"Adapter saved to: {adapter_path}")
        print("="*60)
        
        return adapter_path