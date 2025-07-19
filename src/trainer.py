import torch
from transformers import TrainingArguments, Trainer, AutoTokenizer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM
from src.config import BASE_MODEL
from src.data_loader import load_instruction_dataset


def build_lora_model():
    """
    Loads the base model without GPU support and applies LoRA adapter.
    Avoids meta tensor issues by disabling `device_map="auto"` and setting low memory usage.
    """
    print("[INFO] Loading base model for CPU-only environment...")

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        device_map=None,             # ⚠️ Force full CPU loading
        low_cpu_mem_usage=False,     # ⚠️ Don't offload to meta tensors
    )

    print("[INFO] Applying LoRA config...")
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


def fine_tune_lora():
    """
    Full instruction-tuning pipeline using Hugging Face Trainer + PEFT.
    """
    print("[INFO] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print("[INFO] Loading datasets...")
    train_dataset = load_instruction_dataset("train", tokenizer)
    eval_dataset = load_instruction_dataset("test", tokenizer)

    print("[INFO] Building LoRA model...")
    model = build_lora_model()

    print("[INFO] Preparing training arguments...")
    args = TrainingArguments(
        output_dir="outputs",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=100,
        save_steps=100,
        logging_steps=50,
        num_train_epochs=2,
        learning_rate=2e-4,
        fp16=False,
        bf16=False,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none",
    )

    print("[INFO] Preparing data collator...")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

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