from src.config import MAX_TOKENS
import torch
from typing import List


def run_inference(model, tokenizer, email, max_new_tokens=MAX_TOKENS):
    """
    Runs inference on a single email string and returns the model's response.
    """
    try:
        # Build the prompt with instruction
        prompt = (
            "Classify the intent of the following email:\n\n"
            f"{email}\n\n"
            "Use one of these categories: Request, Informational, Transaction, Feedback."
        )

        # Tokenize and send to the correct device
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode the output
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract model response
        return decoded.split("\n")[-1].strip()

    except Exception as e:
        return f"[ERROR] Inference failed: {e}"


def run_batch_inference(model, tokenizer, emails: List[str], max_new_tokens=MAX_TOKENS):
    """
    Runs inference over a list of emails and returns a list of responses.
    """
    try:
        prompts = [
            f"Classify the intent of the following email:\n\n{email}\n\nUse one of these categories: Request, Informational, Transaction, Feedback."
            for email in emails
        ]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
            )

        results = []
        for output in outputs:
            try:
                decoded = tokenizer.decode(output, skip_special_tokens=True)
                results.append(decoded.split("\n")[-1].strip())
            except Exception as e:
                results.append(f"[ERROR] Decoding failed: {e}")
        return results

    except Exception as e:
        return [f"[ERROR] Batch inference failed: {e}"] * len(emails)