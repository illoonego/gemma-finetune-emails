import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from src.config import BASE_MODEL, dummy_emails
import re

# Choose device
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

def load_adapter_model():
    """
    Loads the base model and applies the fine-tuned LoRA adapter.
    """
    print("[INFO] Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    ).to(DEVICE)

    print("[INFO] Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, "outputs/lora_adapter")
    model = model.to(DEVICE)
    model.eval()

    print("[INFO] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def clean_intent(text):
    """
    Clean the model output to extract just the intent category.
    """
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Extract first word that matches known intents
    known_intents = ['Request', 'Informational', 'Transaction', 'Feedback']
    
    words = text.split()
    for word in words:
        cleaned_word = re.sub(r'[^a-zA-Z]', '', word)
        if cleaned_word in known_intents:
            return cleaned_word
    
    # If no match found, return the first meaningful word
    for word in words[:3]:  # Check first 3 words
        cleaned = re.sub(r'[^a-zA-Z]', '', word)
        if len(cleaned) > 2:
            return cleaned
    
    return text.strip()

def run_inference(model, tokenizer, email):
    """
    Runs inference on a single email string and returns the model's response.
    """
    try:
        # Build a simple and direct prompt
        prompt = f"Email Intent Classification:\n\nEmail: {email}\n\nClassify this email into one of these categories: Request, Informational, Transaction, Feedback\n\nIntent:"

        # Tokenize and send to the correct device
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=3,  # Very short for just the intent word
                do_sample=True,
                temperature=0.3,
                top_k=20,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode the output
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the new tokens (model's response)
        input_length = len(tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True))
        response = decoded[input_length:].strip()
        
        # Clean the response
        cleaned_response = clean_intent(response)
        
        return cleaned_response if cleaned_response else "Unknown"

    except Exception as e:
        return f"[ERROR] {e}"

def evaluate_classifications():
    """
    Test the model and provide evaluation feedback.
    """
    print("[DEBUG] Loading model and tokenizer...")
    model, tokenizer = load_adapter_model()

    test_cases = [
        ("Can you please confirm the meeting time for tomorrow?", "Request"),
        ("We have processed your payment successfully.", "Transaction"), 
        ("This is a gentle reminder to submit your feedback.", "Request"),
        ("Thank you for your email. I will get back to you shortly.", "Informational"),
    ]

    print("[DEBUG] Running email intent classification test...")
    correct = 0
    total = len(test_cases)

    for i, (email, expected) in enumerate(test_cases):
        print(f"\n--- Test Case #{i+1} ---")
        print(f"Email: {email}")
        print(f"Expected Intent: {expected}")
        
        predicted = run_inference(model, tokenizer, email)
        print(f"Predicted Intent: {predicted}")
        
        if predicted == expected:
            print("✅ CORRECT")
            correct += 1
        else:
            print("❌ INCORRECT")

    accuracy = (correct / total) * 100
    print(f"\n{'='*50}")
    print(f"EVALUATION RESULTS:")
    print(f"Correct: {correct}/{total}")
    print(f"Accuracy: {accuracy:.1f}%")
    print(f"{'='*50}")

if __name__ == "__main__":
    evaluate_classifications()
