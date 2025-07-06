from src.config import MAX_NEW_TOKENS
import torch
from typing import List


def run_inference(model, tokenizer, email, max_new_tokens=MAX_NEW_TOKENS):
    """
    Runs inference on a single email string and returns the model's response.
    """
    try:
        # Format instruction for LLaMA 3 Instruct model
        prompt = (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
            "Classify the intent of the following email:\n\n"
            f"{email}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n"
        )

        # Tokenize the input text and move tensors to the same device as the model
        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        except Exception as e:
            return f"[ERROR] Tokenizer failed: {e}"

        # Disable gradient calculation (inference mode)
        with torch.no_grad():
            try:
                # Generate a response from the model
                outputs = model.generate(
                    **inputs,  # Unpack input tensors for the model
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # disables temperature and top_p sampling
                    eos_token_id=tokenizer.eos_token_id,  # Stop generation at the end-of-sequence token
                )
            except Exception as e:
                return f"[ERROR] Model generation failed: {e}"

        # Decode the generated tokens back to text
        try:
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            return f"[ERROR] Decoding failed: {e}"

        # Extract the model's predicted intent (last line)
        return response.split("\n")[-1].strip()
    except Exception as e:
        return f"[ERROR] Inference failed: {e}"


def run_batch_inference(
    model, tokenizer, emails: List[str], max_new_tokens=MAX_NEW_TOKENS
):
    """
    Runs inference on a batch of email strings and returns a list of model responses.
    Each email is processed in a batch for efficiency.
    """
    try:
        # Format prompts for all emails
        prompts = [
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\nClassify the intent of the following email:\n\n{email}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n"
            for email in emails
        ]
        try:
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(
                model.device
            )
        except Exception as e:
            return [f"[ERROR] Tokenizer failed: {e}"] * len(emails)

        with torch.no_grad():
            try:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    eos_token_id=tokenizer.eos_token_id,
                )
            except Exception as e:
                return [f"[ERROR] Model generation failed: {e}"] * len(emails)

        # Decode each output
        responses = []
        for output in outputs:
            try:
                response = tokenizer.decode(output, skip_special_tokens=True)
                responses.append(response.split("\n")[-1].strip())
            except Exception as e:
                responses.append(f"[ERROR] Decoding failed: {e}")
        return responses
    except Exception as e:
        return [f"[ERROR] Batch inference failed: {e}"] * len(emails)
