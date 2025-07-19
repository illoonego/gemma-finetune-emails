import transformers
transformers.logging.set_verbosity_error() # Suppress HF warnings

import time
from tqdm import tqdm
from src.config import BASE_MODEL, MAX_TOKENS
from src.model_loader import load_model_and_tokenizer
from src.data_loader import load_sample_inputs
from src.evaluate import run_inference


def main():
    try:
        # Load model and tokenizer (e.g., Gemma 2B)
        model, tokenizer = load_model_and_tokenizer(BASE_MODEL)
    except Exception as e:
        print(f"[ERROR] Failed to load model or tokenizer: {e}")
        return

    try:
        # Load sample inputs to test basic inference
        sample_inputs = load_sample_inputs()
    except Exception as e:
        print(f"[ERROR] Failed to load sample inputs: {e}")
        return

    # Run inference with progress bar and timing
    for i, email in enumerate(tqdm(sample_inputs, desc="\nClassifying Emails")):
        tqdm.write(f"\n\n=== Email #{i+1} ===\n{email}")

        start_time = time.time()
        try:
            response = run_inference(
                model, tokenizer, email, max_new_tokens=MAX_TOKENS
            )
        except Exception as e:
            tqdm.write(f"[ERROR] Inference failed: {e}")
            continue
        end_time = time.time()

        elapsed = end_time - start_time
        tqdm.write(f"\nPredicted Intent: {response}")
        tqdm.write(f"\nTime Taken: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
