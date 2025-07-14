import torch
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import BASE_MODEL, DEVICE, LABEL2ID
from src.data_loader import load_training_dataset

def build_lora_model():
    """
    Loads the base LLaMA model and applies a LoRA adapter.
    """
    print("[INFO] Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32, # Float32 fro MPS compatibility
        device_map=DEVICE, # Use MPS for Apple Silicon or cpu for others
    )

    print("[INFO] Applying LoRA config...")
    lora_config = LoraConfig(
        r=8,                       # Low-rank dimensionality
        lora_alpha=16,            # Scaling factor
        target_modules=["q_proj", "v_proj"],  # Common LLaMA layers
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model

def fine_tune_lora():
    """
    Full training pipeline using Hugging Face Trainer + PEFT.
    """
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = load_training_dataset("train")
    eval_dataset = load_training_dataset("test")

    model = build_lora_model()

    args = TrainingArguments(
        output_dir="outputs",
        per_device_train_batch_size=1,  # Reduced for memory
        per_device_eval_batch_size=1,  # Reduced for memory
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=100,
        save_steps=100,
        logging_steps=50,
        num_train_epochs=2,
        learning_rate=2e-4,
        fp16=False,  # Use float32 for MPS compatibility
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none",
    )

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

    # Save the fine-tuned LoRA adapter
    model.save_pretrained("outputs/lora_adapter")