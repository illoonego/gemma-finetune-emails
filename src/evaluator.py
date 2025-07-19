"""
Evaluator for LoRA fine-tuning pipeline
Handles model evaluation, metrics calculation, and performance analysis
"""
from typing import List, Dict, Tuple
import time
from .model_handler import ModelHandler
from .data_handler import DataHandler


class Evaluator:
    """Centralized evaluation operations"""
    
    def __init__(self, model_handler: ModelHandler, data_handler: DataHandler):
        self.model_handler = model_handler
        self.data_handler = data_handler
    
    def evaluate_on_test_emails(self, verbose: bool = True) -> Dict:
        """Evaluate model on predefined test emails"""
        test_emails = self.data_handler.get_test_emails()
        expected_intents = self.data_handler.get_expected_intents()
        
        results = []
        correct_predictions = 0
        total_predictions = len(test_emails)
        
        if verbose:
            print("[INFO] Running evaluation on test emails...")
            print("="*60)
        
        for i, (email, expected) in enumerate(zip(test_emails, expected_intents)):
            if verbose:
                print(f"\n--- Test Case #{i+1} ---")
                print(f"Email: {email}")
                print(f"Expected: {expected}")
            
            # Create prompt and get prediction
            prompt = self.data_handler.create_evaluation_prompt(email)
            
            start_time = time.time()
            try:
                raw_response = self.model_handler.generate_response(prompt)
                predicted_intent = self.data_handler.extract_intent_from_response(raw_response)
                inference_time = time.time() - start_time
                
                # Check if correct
                is_correct = predicted_intent == expected
                if is_correct:
                    correct_predictions += 1
                
                result = {
                    "email": email,
                    "expected": expected,
                    "predicted": predicted_intent,
                    "raw_response": raw_response,
                    "correct": is_correct,
                    "inference_time": inference_time
                }
                results.append(result)
                
                if verbose:
                    print(f"Predicted: {predicted_intent}")
                    print(f"Raw response: '{raw_response}'")
                    print(f"Result: {'✅ CORRECT' if is_correct else '❌ INCORRECT'}")
                    print(f"Time: {inference_time:.2f}s")
                    
            except Exception as e:
                result = {
                    "email": email,
                    "expected": expected,
                    "predicted": "ERROR",
                    "raw_response": str(e),
                    "correct": False,
                    "inference_time": 0
                }
                results.append(result)
                
                if verbose:
                    print(f"❌ ERROR: {e}")
        
        # Calculate metrics
        accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
        avg_inference_time = sum(r["inference_time"] for r in results) / len(results) if results else 0
        
        summary = {
            "accuracy": accuracy,
            "correct": correct_predictions,
            "total": total_predictions,
            "avg_inference_time": avg_inference_time,
            "results": results
        }
        
        if verbose:
            print("\n" + "="*60)
            print("EVALUATION SUMMARY:")
            print(f"Accuracy: {accuracy:.1f}% ({correct_predictions}/{total_predictions})")
            print(f"Average Inference Time: {avg_inference_time:.2f}s")
            print("="*60)
        
        return summary
    
    def compare_models(self, adapter_path: str = None) -> Dict:
        """Compare base model vs fine-tuned model performance"""
        print("[INFO] Comparing base model vs LoRA fine-tuned model...")
        
        # Test base model
        print("\n[INFO] Testing BASE MODEL...")
        self.model_handler.load_base_model()
        base_results = self.evaluate_on_test_emails(verbose=False)
        
        # Test LoRA model
        print("\n[INFO] Testing LORA FINE-TUNED MODEL...")
        self.model_handler.load_lora_adapter(adapter_path)
        lora_results = self.evaluate_on_test_emails(verbose=False)
        
        # Compare results
        comparison = {
            "base_model": {
                "accuracy": base_results["accuracy"],
                "avg_time": base_results["avg_inference_time"]
            },
            "lora_model": {
                "accuracy": lora_results["accuracy"],
                "avg_time": lora_results["avg_inference_time"]
            },
            "improvement": {
                "accuracy_delta": lora_results["accuracy"] - base_results["accuracy"],
                "time_delta": lora_results["avg_inference_time"] - base_results["avg_inference_time"]
            }
        }
        
        # Print comparison
        print("\n" + "="*60)
        print("MODEL COMPARISON:")
        print(f"Base Model     - Accuracy: {base_results['accuracy']:.1f}%, Time: {base_results['avg_inference_time']:.2f}s")
        print(f"LoRA Model     - Accuracy: {lora_results['accuracy']:.1f}%, Time: {lora_results['avg_inference_time']:.2f}s")
        print(f"Improvement    - Accuracy: {comparison['improvement']['accuracy_delta']:+.1f}%, Time: {comparison['improvement']['time_delta']:+.2f}s")
        print("="*60)
        
        return comparison
    
    def evaluate_per_intent(self, results: List[Dict] = None) -> Dict:
        """Calculate per-intent accuracy"""
        if results is None:
            evaluation = self.evaluate_on_test_emails(verbose=False)
            results = evaluation["results"]
        
        # Group by intent
        intent_stats = {}
        for result in results:
            expected = result["expected"]
            if expected not in intent_stats:
                intent_stats[expected] = {"correct": 0, "total": 0}
            
            intent_stats[expected]["total"] += 1
            if result["correct"]:
                intent_stats[expected]["correct"] += 1
        
        # Calculate per-intent accuracy
        for intent in intent_stats:
            stats = intent_stats[intent]
            stats["accuracy"] = (stats["correct"] / stats["total"]) * 100 if stats["total"] > 0 else 0
        
        return intent_stats
    
    def generate_evaluation_report(self, save_path: str = None) -> str:
        """Generate a comprehensive evaluation report"""
        print("[INFO] Generating comprehensive evaluation report...")
        
        # Get model info
        model_info = self.model_handler.get_model_info()
        
        # Get dataset info  
        dataset_info = self.data_handler.get_dataset_info()
        
        # Run evaluation
        evaluation_results = self.evaluate_on_test_emails(verbose=False)
        
        # Calculate per-intent stats
        intent_stats = self.evaluate_per_intent(evaluation_results["results"])
        
        # Generate report
        report = f"""
# LoRA Fine-Tuning Evaluation Report

## Model Configuration
- Base Model: {model_info.get('base_model', 'Unknown')}
- Device: {model_info.get('device', 'Unknown')}
- Is LoRA Model: {model_info.get('is_lora_model', False)}

## Dataset Information
- Dataset: {dataset_info.get('dataset_name', 'Unknown')}
- Training Samples: {dataset_info.get('train_size', 0)}
- Test Samples: {dataset_info.get('test_size', 0)}
- Intent Categories: {', '.join(dataset_info.get('unique_intents', []))}

## Overall Performance
- Overall Accuracy: {evaluation_results['accuracy']:.1f}%
- Correct Predictions: {evaluation_results['correct']}/{evaluation_results['total']}
- Average Inference Time: {evaluation_results['avg_inference_time']:.3f} seconds

## Per-Intent Performance
"""
        
        for intent, stats in intent_stats.items():
            report += f"- {intent}: {stats['accuracy']:.1f}% ({stats['correct']}/{stats['total']})\n"
        
        report += f"""
## Detailed Results
"""
        
        for i, result in enumerate(evaluation_results['results']):
            status = "✅" if result['correct'] else "❌"
            report += f"""
### Test Case {i+1} {status}
- **Email:** {result['email'][:100]}{'...' if len(result['email']) > 100 else ''}
- **Expected:** {result['expected']}
- **Predicted:** {result['predicted']}
- **Raw Response:** "{result['raw_response']}"
- **Inference Time:** {result['inference_time']:.3f}s
"""
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"[INFO] Report saved to: {save_path}")
        
        return report
