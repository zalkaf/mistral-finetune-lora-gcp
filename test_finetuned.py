# test_finetuned.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

print("="*80)
print("=== Test of Mistral Fine Tuned Model ===")
print("="*80)

# Load base model
print("\n[Step 1] Loading base Mistral 7b model...")
base_model_name = "mistralai/Mistral-7B-v0.1"
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    load_in_4bit=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load LoRA weights
print("[Step 2] Loading the new LoRA weights...")
model = PeftModel.from_pretrained(base_model, "./mistral-7b-lora-final")
tokenizer = AutoTokenizer.from_pretrained("./mistral-7b-lora-final")

print("[Step 3] Yaay! Our Model is ready!")
print("="*80)

# Test prompts
test_prompts = [
    "tell me about the history of artificial intellignece:",
    "write a simple javascript code to create a tic tac toe game:",
    "how is insulin produced in the human body?",
]

print("\nGenerating responses...\n")

for i, prompt in enumerate(test_prompts, 1):
    print(f"\n{'='*80}")
    print(f"PROMPT {i}: {prompt}")
    print('='*80)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(response)
    print()

print("="*80)
print("Testing complete!")
print("="*80)