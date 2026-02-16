# compare_models.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

print("="*80)
print("Lets compare the 2 models (base and fine tuned!)")
print("="*80)

# Load base model
print("\n[Step 1] Loading the base Mistral model...")
base_model_name = "mistralai/Mistral-7B-v0.1"
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    load_in_4bit=True,
    torch_dtype=torch.float16,
    device_map="auto"
)
base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Load fine-tuned model
print("[Step 2] Loading the new fine-tuned model...")
finetuned_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    load_in_4bit=True,
    torch_dtype=torch.float16,
    device_map="auto"
)
finetuned_model = PeftModel.from_pretrained(finetuned_model, "./mistral-7b-lora-final")
finetuned_tokenizer = AutoTokenizer.from_pretrained("./mistral-7b-lora-final")

print("[Step 3] Yaay! Models ready!")
print("="*80)

# Test prompts
test_prompts = [
    "tell me the history of artificial intelligence:",
    "write a javascript code to create a simple tic tac toe game:",
    "how is insulin produced in the human body naturally?",
]

def generate_response(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\nComparing responses...\n")

for i, prompt in enumerate(test_prompts, 1):
    print(f"\n{'='*80}")
    print(f"TEST {i}: {prompt}")
    print('='*80)
    
    print("\n📦 Response of the base model:")
    print("-" * 80)
    base_response = generate_response(base_model, base_tokenizer, prompt)
    print(base_response)
    
    print("\n✨ Response of the fine tuned model:")
    print("-" * 80)
    finetuned_response = generate_response(finetuned_model, finetuned_tokenizer, prompt)
    print(finetuned_response)
    
    print()

print("="*80)
print("Done comparison!!")
print("="*80)

