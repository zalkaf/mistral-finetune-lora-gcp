# test_inference.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "mistralai/Mistral-7B-v0.1"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

print("Loading model (this may take a few minutes)...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("Model loaded successfully!")
print(f"Model device: {model.device}")
print(f"Model dtype: {model.dtype}")

# Test generation
prompt = "Explain generative AI in simple terms:"
print(f"\nPrompt: {prompt}")
print("Generating response...\n")

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        repetition_penalty=1.1
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("=" * 80)
print(response)
print("=" * 80)
