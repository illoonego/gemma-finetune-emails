from datasets import load_dataset
from transformers import AutoTokenizer
from src.config import BASE_MODEL, LABEL2ID, dummy_emails


def load_sample_inputs():
    """
    Returns a small set of test emails for validation.
    """
    return dummy_emails

def load_training_dataset(split="train", max_length=128):
    """
    Loads and tokenizes the elenigkove/Email_Intent_Classification dataset.
    """
    print("\n[INFO] Loading dataset: elenigkove/Email_Intent_Classification")
    dataset = load_dataset("elenigkove/Email_Intent_Classification")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    # Preprocessing function
    def preprocess(batch):
        prompt = [
            f"Classify the intent of this email using one of the following labels: {', '.join(LABEL2ID.keys())}.\n\nEmail:\n{email}"
            for email in batch["Email"]
        ]
        tokens = tokenizer(prompt, padding="max_length", truncation=True, max_length=max_length)
        tokens["labels"] = [LABEL2ID[label] for label in batch["Intent"]]
        return tokens

    tokenized_dataset = dataset.map(preprocess, batched=True, remove_columns=["Email", "Intent"])

    print("[INFO] Dataset loaded and tokenized.")
    return tokenized_dataset[split]
