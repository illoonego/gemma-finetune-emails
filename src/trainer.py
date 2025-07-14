import torch
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from src.config import BASE_MODEL, LABEL2ID
from src.data_loader import load_training_dataset

def build_lora_model():
    """
    Loads the base model with 4-bit quantization and applies LoRA adapter.
    Works on CPU (Apple Silicon compatible), but slower.
    """
    print("[INFO] Loading base model with 4-bit quantization...")

    # Define quantization config (4-bit mode)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # Load weights in 4-bit
        bnb_4bit_compute_dtype=torch.float32, # Use float32 for computation
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",  # Non-uniform quantization
    )

    # Load model with quantization config
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",  # Use CPU on Mac
        trust_remote_code=True,
    )

    # Prepare for LoRA
    print("[INFO] Applying LoRA config...")
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],  # Adjust if needed
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model

def fine_tune_lora():
    """
    Full training pipeline using Hugging Face Trainer + PEFT on 4-bit model.
    """
    from transformers import AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding

    print("[INFO] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print("[INFO] Loading datasets...")
    train_dataset = load_training_dataset("train")
    eval_dataset = load_training_dataset("test")

    print("[INFO] Building LoRA model...")
    model = build_lora_model()

    print("[INFO] Preparing training arguments...")
    args = TrainingArguments(
        output_dir="outputs",
        per_device_train_batch_size=1,      # small batch due to CPU and RAM
        per_device_eval_batch_size=1,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=100,
        save_steps=100,
        logging_steps=50,
        num_train_epochs=2,
        learning_rate=2e-4,
        fp16=False,                         # no GPU, use float32 on CPU
        bf16=False,                         # also disable bf16 (just in case)
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none",
    )

    print("[INFO] Preparing data collator...")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("[INFO] Starting training...")
    trainer.train()
    print("[INFO] Training complete!")

    print("[INFO] Saving fine-tuned LoRA adapter...")
    model.save_pretrained("outputs/lora_adapter")