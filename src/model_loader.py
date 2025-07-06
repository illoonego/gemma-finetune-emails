from src.config import DEVICE
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def load_model_and_tokenizer(base_model_name):
    """
    Loads the pre-trained LLaMA 3 7B model and tokenizer from Hugging Face.
    Adds error handling for download/load failures.
    """
    print("\n[INFO] Loading model and tokenizer...")
    try:
        # Load the tokenizer, which converts text to token IDs and vice versa
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name, trust_remote_code=True
        )
    except Exception as e:
        print(f"[ERROR] Failed to load tokenizer: {e}")
        raise

    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=DEVICE,  # Use device from config
        )
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        raise

    print("[INFO] Model and tokenizer loaded.")
    return model, tokenizer  # Return both for use in inference
