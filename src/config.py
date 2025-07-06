# Base LlaMA 3 model from Hugging Face
BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

# Default max tokens for generation
MAX_NEW_TOKENS = 64

# Default device (can be 'auto', 'cpu', or 'cuda')
DEVICE = "auto"

# Sample input list for validation
dummy_emails = [
    "Can you please confirm the meeting time for tomorrow?",
    "We have processed your payment successfully.",
    "This is a gentle reminder to submit your feedback.",
    "Thank you for your email. I will get back to you shortly.",
]

# Class labels for the dataset
LABELS = ["Request", "Informational", "Transaction", "Feedback"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}