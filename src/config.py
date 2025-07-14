import torch

# Base model (Gemma 2B for lightweight fine-tuning)
BASE_MODEL = "google/gemma-2b"

# Default max tokens for generation
MAX_TOKENS = 128

# Ensure compatibility with MPS (Apple Silicon)
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# Class labels for the dataset
LABELS = ["Request", "Informational", "Transaction", "Feedback"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

# Sample input list for validation
dummy_emails = [
    "Can you please confirm the meeting time for tomorrow?",
    "We have processed your payment successfully.",
    "This is a gentle reminder to submit your feedback.",
    "Thank you for your email. I will get back to you shortly.",
]