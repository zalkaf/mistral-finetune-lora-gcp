# mistral-finetune-lora-gcp
Finetuned with LoRA Adapters Project on GCP


# Mistral-7B LoRA Fine-Tuning

Fine-tuning Mistral-7B language model using LoRA (Low-Rank Adaptation) on a GPU to create a custom AI model with minimal computational cost.

## Overview

This project demonstrates how to fine-tune the Mistral-7B open-source language model using LoRA, a parameter-efficient fine-tuning technique. Instead of updating all 7 billion parameters, LoRA adds small "adapter" layers that contain only ~4 million trainable parameters, making fine-tuning affordable on a single GPU.

## Features

- **Efficient Training**: Uses 4-bit quantization and LoRA for memory-efficient training
- **Single GPU**: Runs on a single GPU (16GB+ VRAM)
- **Cost-Effective**: Train for ~$1-5 on cloud GPUs
- **Production Ready**: Not ready for PROD! This is used only for testing purposed. The script includes comparison results

## Tech Stack

- **Base Model**: Mistral-7B-v0.1
- **Framework**: Hugging Face Transformers
- **Fine-tuning**: PEFT (LoRA)
- **Quantization**: bitsandbytes (4-bit)
- **Hardware**: NVIDIA L4 GPU with ubuntu image
- gcloud compute instances create llm-trainer \
    --zone=us-central1-a \
    --network=vpc-name \
    --subnet=subnet-name \
    --machine-type=g2-standard-4 \
    --accelerator=type=nvidia-l4,count=1 \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=200GB \
    --maintenance-policy=TERMINATE \
    --provisioning-model=SPOT

## Installation

### Prerequisites

- Python 3.8+
- NVIDIA GPU with 16GB+ VRAM
- CUDA 11.8+ installed

### Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/mistral-lora-finetuning.git
cd mistral-lora-finetuning

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

Create a `requirements.txt` file:
```
torch>=2.0.0
transformers>=4.40.0
accelerate>=0.28.0
peft>=0.10.0
bitsandbytes>=0.41.0
datasets>=2.15.0
trl>=0.7.4
huggingface-hub>=0.22.0
```

## 🚀 Quick Start

### 1. Test Base Model Inference

```bash
python test_inference.py
```

This loads the base Mistral-7B model and generates a sample response to verify your setup works.

### 2. Fine-Tune the Model

```bash
python finetune.py
```

**Training Configuration:**
- Dataset: OpenAssistant Guanaco (1000 examples)
- Epochs: 1
- Training time: ~1-2 hours on L4 GPU
- Output: `./mistral-7b-lora-final/`

**LoRA Configuration:**
```python
LoraConfig(
    r=16,                    # Rank of adaptation matrices
    lora_alpha=32,           # Scaling factor
    target_modules=[         # Layers to adapt
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

### 3. Test Fine-Tuned Model

```bash
python test_finetuned.py
```

### 4. Compare Base vs Fine-Tuned

```bash
python compare_models.py
```

This script generates responses from both the base model and fine-tuned model side-by-side for comparison.

## Project Structure

```
mistral-lora-finetuning/
├── README.md
├── requirements.txt
├── finetune.py              # Main training script
├── test_inference.py        # Test base model
├── test_finetuned.py        # Test fine-tuned model
├── compare_models.py        # Compare base vs fine-tuned
├── mistral-7b-lora/         # Training checkpoints (created during training)
└── mistral-7b-lora-final/   # Final fine-tuned model (created after training)
```

## Configuration

### Training Parameters

```python
# Dataset size
dataset = dataset.select(range(1000))  # I only used 1000 examples for faster and cheaper trainign 

# Training epochs
num_train_epochs=1  # I only used 1 EPOCH for this poc

# Batch size (reduce if OOM)
per_device_train_batch_size=4
gradient_accumulation_steps=4  # Effective batch size = 16
```

### LoRA Hyperparameters
Here I used rank 16 and alpha 32 to optimize the cost. 
```python
# For faster training (less memory):
r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"]
```

## Cost Estimation

I used L4 GPU in us-central-1 runing for 2-3 hours which costed me about 5 bucks

