from datasets import load_dataset
from transformers import AutoTokenizer
from src.config import BASE_MODEL, MAX_TOKENS, dummy_emails


def load_sample_inputs():
    """
    Returns a small set of test emails for validation.
    """
    return dummy_emails


def load_instruction_dataset(split="train", tokenizer=None, max_length=MAX_TOKENS):
    """
    Loads and formats the dataset in instruction-response format.
    Output is a single sequence: <|user|> prompt <|eot_id|> <|assistant|> response <|eot_id|>
    Labels = input_ids.
    """
    print(f"\n[INFO] Loading dataset: elenigkove/Email_Intent_Classification ({split})")
    dataset = load_dataset("elenigkove/Email_Intent_Classification")[split]

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    def format_example(example):
        prompt = (
            "<|user|>\n"
            f"Classify the intent of the following email:\n\n{example['Email']}\n\nIntent:"
            "<|eot_id|>\n<|assistant|>\n"
            f"{example['Intent']}<|eot_id|>"
        )

        tokenized = tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    processed = dataset.map(format_example, remove_columns=["Email", "Intent"])
    print("[INFO] Dataset loaded and tokenized.")
    return processed