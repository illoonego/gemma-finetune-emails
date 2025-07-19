#!/usr/bin/env python3
"""
Main entry point for LoRA Fine-tuning Pipeline
A unified CLI for the complete email intent classification workflow
"""
import argparse
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model_handler import ModelHandler
from src.data_handler import DataHandler
from src.evaluator import Evaluator
from src.trainer import LoRATrainer
from configs import MODEL_CONFIG, DATA_CONFIG, TRAINING_CONFIG


class LoRAPipeline:
    """Complete LoRA fine-tuning pipeline"""
    
    def __init__(self):
        self.model_handler = ModelHandler()
        self.data_handler = DataHandler()
        self.evaluator = None
        self.trainer = LoRATrainer()
    
    def setup_evaluator(self):
        """Setup evaluator with current handlers"""
        if self.evaluator is None:
            self.evaluator = Evaluator(self.model_handler, self.data_handler)
    
    def show_info(self):
        """Show pipeline configuration and dataset information"""
        print("="*60)
        print("LORA FINE-TUNING PIPELINE - INFORMATION")
        print("="*60)
        
        print("\n📋 CONFIGURATION:")
        print(f"  Base Model: {MODEL_CONFIG.base_model}")
        print(f"  Max Tokens: {MODEL_CONFIG.max_tokens}")
        print(f"  Device: {MODEL_CONFIG.get_device()}")
        print(f"  Dataset: {DATA_CONFIG.dataset_name}")
        print(f"  Intent Labels: {', '.join(DATA_CONFIG.intent_labels)}")
        print(f"  Training Epochs: {TRAINING_CONFIG.num_epochs}")
        print(f"  Learning Rate: {TRAINING_CONFIG.learning_rate}")
        print(f"  Output Directory: {TRAINING_CONFIG.output_dir}")
        
        # Load and show dataset info
        self.data_handler.load_dataset()
        dataset_info = self.data_handler.get_dataset_info()
        
        print(f"\n📊 DATASET INFO:")
        print(f"  Training Samples: {dataset_info['train_size']}")
        print(f"  Test Samples: {dataset_info['test_size']}")
        print(f"  Unique Intents: {', '.join(dataset_info['unique_intents'])}")
        
        print(f"\n📝 SAMPLE DATA:")
        for i, sample in enumerate(dataset_info['sample_examples']):
            print(f"  Example {i+1}: \"{sample['email']}\" → {sample['intent']}")
    
    def test_base_model(self):
        """Test the base model performance"""
        print("="*60)
        print("TESTING BASE MODEL PERFORMANCE")
        print("="*60)
        
        # Load base model
        self.model_handler.load_base_model()
        self.data_handler.set_tokenizer(self.model_handler.tokenizer)
        self.setup_evaluator()
        
        # Evaluate
        results = self.evaluator.evaluate_on_test_emails()
        return results
    
    def train_model(self, custom_config=None):
        """Train the LoRA model"""
        print("="*60)
        print("STARTING LORA TRAINING")
        print("="*60)
        
        adapter_path = self.trainer.train(custom_config)
        return adapter_path
    
    def test_lora_model(self, adapter_path=None):
        """Test the LoRA fine-tuned model"""
        print("="*60)
        print("TESTING LORA FINE-TUNED MODEL")
        print("="*60)
        
        # Load LoRA model
        self.model_handler.load_lora_adapter(adapter_path)
        self.data_handler.set_tokenizer(self.model_handler.tokenizer)
        self.setup_evaluator()
        
        # Evaluate
        results = self.evaluator.evaluate_on_test_emails()
        return results
    
    def compare_models(self, adapter_path=None):
        """Compare base model vs LoRA model"""
        print("="*60)
        print("COMPARING BASE MODEL vs LORA MODEL")
        print("="*60)
        
        self.data_handler.set_tokenizer(None)  # Will be set by model_handler
        self.setup_evaluator()
        
        comparison = self.evaluator.compare_models(adapter_path)
        return comparison
    
    def interactive_test(self, adapter_path=None):
        """Interactive testing with custom emails"""
        print("="*60)
        print("INTERACTIVE EMAIL INTENT CLASSIFICATION")
        print("="*60)
        
        # Load model (LoRA if available, base otherwise)
        try:
            if adapter_path and os.path.exists(adapter_path):
                print("[INFO] Loading LoRA fine-tuned model...")
                self.model_handler.load_lora_adapter(adapter_path)
            else:
                print("[INFO] Loading base model...")
                self.model_handler.load_base_model()
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            return
        
        self.data_handler.set_tokenizer(self.model_handler.tokenizer)
        
        print(f"\nAvailable intent categories: {', '.join(DATA_CONFIG.intent_labels)}")
        print("Enter emails to classify (type 'quit' to exit):")
        
        while True:
            print("\n" + "-"*40)
            email = input("📧 Enter email text: ").strip()
            
            if email.lower() in ['quit', 'exit', 'q']:
                break
            
            if not email:
                continue
            
            try:
                prompt = self.data_handler.create_evaluation_prompt(email)
                raw_response = self.model_handler.generate_response(prompt)
                predicted_intent = self.data_handler.extract_intent_from_response(raw_response)
                
                print(f"🎯 Predicted Intent: {predicted_intent}")
                print(f"🔧 Raw Response: '{raw_response}'")
                
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def full_pipeline(self):
        """Run the complete pipeline: info -> base test -> train -> lora test -> compare"""
        print("🚀 RUNNING COMPLETE LORA FINE-TUNING PIPELINE")
        print("="*60)
        
        # 1. Show info
        self.show_info()
        input("\n⏳ Press Enter to test base model performance...")
        
        # 2. Test base model
        base_results = self.test_base_model()
        input(f"\n⏳ Base model accuracy: {base_results['accuracy']:.1f}%. Press Enter to start training...")
        
        # 3. Train model
        adapter_path = self.train_model()
        input(f"\n⏳ Training complete! Adapter saved to {adapter_path}. Press Enter to test LoRA model...")
        
        # 4. Test LoRA model
        lora_results = self.test_lora_model(adapter_path)
        input(f"\n⏳ LoRA model accuracy: {lora_results['accuracy']:.1f}%. Press Enter for comparison...")
        
        # 5. Compare models
        comparison = self.compare_models(adapter_path)
        
        print("\n🎉 PIPELINE COMPLETED!")
        print(f"Final improvement: {comparison['improvement']['accuracy_delta']:+.1f}% accuracy")
        
        # 6. Optional interactive test
        test_interactive = input("\n🤖 Test with custom emails? (y/N): ").strip().lower()
        if test_interactive in ['y', 'yes']:
            self.interactive_test(adapter_path)


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning Pipeline for Email Intent Classification")
    
    parser.add_argument("command", choices=[
        "info", "base-test", "train", "lora-test", "compare", "interactive", "full"
    ], help="Pipeline command to run")
    
    parser.add_argument("--adapter-path", type=str, default=None,
                       help="Path to LoRA adapter (default: outputs/lora_adapter)")
    
    parser.add_argument("--epochs", type=int, default=None,
                       help="Number of training epochs")
    
    parser.add_argument("--learning-rate", type=float, default=None,
                       help="Learning rate for training")
    
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Batch size for training")
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = LoRAPipeline()
    
    # Prepare custom training config if provided
    custom_config = {}
    if args.epochs is not None:
        custom_config['num_train_epochs'] = args.epochs
    if args.learning_rate is not None:
        custom_config['learning_rate'] = args.learning_rate
    if args.batch_size is not None:
        custom_config['per_device_train_batch_size'] = args.batch_size
        custom_config['per_device_eval_batch_size'] = args.batch_size
    
    # Execute command
    try:
        if args.command == "info":
            pipeline.show_info()
        
        elif args.command == "base-test":
            pipeline.test_base_model()
        
        elif args.command == "train":
            pipeline.train_model(custom_config if custom_config else None)
        
        elif args.command == "lora-test":
            pipeline.test_lora_model(args.adapter_path)
        
        elif args.command == "compare":
            pipeline.compare_models(args.adapter_path)
        
        elif args.command == "interactive":
            pipeline.interactive_test(args.adapter_path)
        
        elif args.command == "full":
            pipeline.full_pipeline()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
