"""
Data handler for LoRA fine-tuning pipeline
Handles dataset loading, preprocessing, and sample data
"""
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import List, Dict, Any

from configs import DATA_CONFIG, MODEL_CONFIG


class DataHandler:
    """Centralized data operations"""
    
    def __init__(self, tokenizer: AutoTokenizer = None):
        self.tokenizer = tokenizer
        self.dataset = None
    
    def set_tokenizer(self, tokenizer: AutoTokenizer):
        """Set the tokenizer for preprocessing"""
        self.tokenizer = tokenizer
    
    def load_dataset(self, dataset_name: str = None) -> dict:
        """Load the training dataset"""
        if dataset_name is None:
            dataset_name = DATA_CONFIG.dataset_name
        
        print(f"[INFO] Loading dataset: {dataset_name}")
        self.dataset = load_dataset(dataset_name)
        
        print(f"[INFO] Dataset loaded:")
        print(f"  - Train samples: {len(self.dataset['train'])}")
        print(f"  - Test samples: {len(self.dataset['test'])}")
        
        return self.dataset
    
    def get_dataset_info(self) -> dict:
        """Get information about the dataset"""
        if self.dataset is None:
            return {"status": "No dataset loaded"}
        
        train_data = self.dataset['train']
        
        # Get unique intents
        unique_intents = list(set(train_data['Intent']))
        
        # Sample examples
        samples = []
        for i in range(min(3, len(train_data))):
            samples.append({
                "email": train_data[i]['Email'][:100] + "..." if len(train_data[i]['Email']) > 100 else train_data[i]['Email'],
                "intent": train_data[i]['Intent']
            })
        
        return {
            "dataset_name": DATA_CONFIG.dataset_name,
            "train_size": len(train_data),
            "test_size": len(self.dataset['test']) if 'test' in self.dataset else 0,
            "unique_intents": sorted(unique_intents),
            "sample_examples": samples
        }
    
    def preprocess_for_training(self, split: str = "train", max_length: int = None) -> Any:
        """Preprocess dataset for training"""
        if self.dataset is None:
            self.load_dataset()
        
        if self.tokenizer is None:
            raise ValueError("Tokenizer not set. Call set_tokenizer() first.")
        
        if max_length is None:
            max_length = MODEL_CONFIG.max_tokens
        
        print(f"[INFO] Preprocessing {split} split for training...")
        
        def format_example(example):
            """Format example in instruction-response format"""
            prompt = (
                "Classify the intent of the following email:\n\n"
                f"{example['Email']}\n\n"
                f"Intent: {example['Intent']}"
            )
            
            tokenized = self.tokenizer(
                prompt,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors=None
            )
            
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
        
        # Process the dataset
        processed_dataset = self.dataset[split].map(
            format_example,
            remove_columns=["Email", "Intent"]
        )
        
        print(f"[INFO] {split.title()} dataset preprocessed: {len(processed_dataset)} samples")
        return processed_dataset
    
    def get_test_emails(self) -> List[str]:
        """Get test emails for evaluation"""
        return DATA_CONFIG.test_emails
    
    def get_expected_intents(self) -> List[str]:
        """Get expected intents for test emails (manually defined)"""
        # These correspond to the test emails in DATA_CONFIG
        return ["Request", "Transaction", "Request", "Informational"]
    
    def create_evaluation_prompt(self, email: str) -> str:
        """Create a properly formatted prompt for evaluation"""
        return (
            f"Classify the intent of the following email:\n\n"
            f"{email}\n\n"
            f"Choose from: {', '.join(DATA_CONFIG.intent_labels)}\n\n"
            f"Intent:"
        )
    
    def extract_intent_from_response(self, response: str) -> str:
        """Extract intent category from model response"""
        import re
        
        # Clean the response
        response = re.sub(r'<[^>]+>', '', response)  # Remove HTML tags
        response = response.strip()
        
        # Look for exact matches with known intents
        for intent in DATA_CONFIG.intent_labels:
            if intent.lower() in response.lower():
                return intent
        
        # If no exact match, return the first meaningful word
        words = response.split()
        for word in words[:3]:
            cleaned_word = re.sub(r'[^a-zA-Z]', '', word)
            if len(cleaned_word) > 2:
                return cleaned_word.title()
        
        return "Unknown"
