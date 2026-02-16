# finetune.py
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

print("="*80)
print("This script is for Mistral 7b fine tuning")
print("="*80)

# 1. Load model and tokenizer
print("\n[Step 1] Loading Mistral-7B model and the tokeniezer...")
model_name = "mistralai/Mistral-7B-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

print(f"** Horray! Model loaded successfuly on: {model.device}")

# 2. Configure LoRA
print("\n[Step 2] Lora Configuratin ...")
lora_config = LoraConfig(
    r=16,                       
    lora_alpha=32,              
    target_modules=[            
        "q_proj",
        "k_proj", 
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# Print which parameters are trainable 
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
all_params = sum(p.numel() for p in model.parameters())
print(f"✓ Trainable params: {trainable_params:,} || All params: {all_params:,} || Trainable %: {100 * trainable_params / all_params:.2f}%")

# 3. Load and prepare dataset
print("\n[Step 3] Loading training dataset...")
# Using ready made predefined dataset called penassistant-guanaco on HF
dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")

# Take a subset for faster training 
dataset = dataset.select(range(1000))  # currently only using 1000 examples for quick and cheaper training

print(f"** Horray! Dataset was successfuly loaded: {len(dataset)} examples")
print(f"Printing few Examples to check: {dataset[0]['text'][:200]}...")

# 4. Tokenize dataset
print("\n[Step 4] Tokenizing dataset...")

def tokenize_function(examples):
    # Tokenize the text
    result = tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length",  
    )
    # Copy input_ids to labels for causal LM
    result["labels"] = result["input_ids"].copy()
    return result

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset.column_names,
    desc="Tokenizing"
)

print(f"** Horray! Dataset succesfuly tokenized")

# 5. Training arguments
print("\n[Step 5] Setting My trainign configurations...")
training_args = TrainingArguments(
    output_dir="./mistral-7b-lora",
    num_train_epochs=1,                    # limiting to 1 EPOCH for faster fine tuning
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,         
    learning_rate=2e-4,
    fp16=True,
    save_strategy="epoch",
    logging_steps=10,
    optim="paged_adamw_8bit",
    warmup_steps=50,
    max_grad_norm=0.3,
    group_by_length=True,                  # Group similar lengths for efficiency
    lr_scheduler_type="cosine",
)

print(f"** Horray!  Training for {training_args.num_train_epochs} epoch(s)")
print(f"** Horray!  Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")


from transformers import default_data_collator
data_collator = default_data_collator

# 7. Create trainer
print("\n[Step 6] Creating the new trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

print("** Horray! Trainer is now ready")

# 8. Train!
print("\n[Step 7] Starting training...")
print("="*80)
print("This will take about 1-2 hours.. using L4 GPU here...")
print("Training progress below:")
print("="*80)

trainer.train()

# 9. Save the model
print("\n" + "="*80)
print("** Horray! Training is now succsfuly complete")
print("="*80)
print("\nSaving model...")
model.save_pretrained("./mistral-7b-lora-final")
tokenizer.save_pretrained("./mistral-7b-lora-final")


print("\nNow lets compare the 2 models (base and fine tuned!)")

print("="*80)
