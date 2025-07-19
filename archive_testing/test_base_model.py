import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.config import BASE_MODEL, dummy_emails

# Choose device
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

def load_base_model():
    """
    Loads the base model without LoRA adapter for comparison.
    """
    print("[INFO] Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    ).to(DEVICE)
    model.eval()

    print("[INFO] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def run_inference(model, tokenizer, email):
    """
    Runs inference on a single email string and returns the model's response.
    """
    try:
        # Build the prompt using the same format as training data
        prompt = (
            "<|user|>\n"
            f"Classify the intent of the following email:\n\n{email}\n\nIntent:"
            "<|eot_id|>\n<|assistant|>\n"
        )

        # Tokenize and send to the correct device
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode the output
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the new tokens (model's response)
        input_length = len(tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True))
        response = decoded[input_length:].strip()
        
        return response if response else "No response"

    except Exception as e:
        return f"[ERROR] Inference failed: {e}"

def main():
    print("[DEBUG] Testing BASE MODEL (without LoRA)...")
    model, tokenizer = load_base_model()

    print("[DEBUG] Running inference on dummy emails...")

    for i, email in enumerate(dummy_emails):
        print(f"\n--- Email #{i+1} ---")
        print(f"Email:\n{email}")
        
        prediction = run_inference(model, tokenizer, email)
        print(f"Base Model Output: {prediction}")

if __name__ == "__main__":
    main()
