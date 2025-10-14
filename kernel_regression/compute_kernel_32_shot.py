from transformers import AutoTokenizer, GPTNeoXForCausalLM
import torch
from datasets import load_dataset
import numpy as np

model_name = "EleutherAI/pythia-160m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = GPTNeoXForCausalLM.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

positive_token_id = 2762 
negative_token_id = 4016

# ============================================================
# Setup: Enable gradients for last 2 layers
# ============================================================

for param in model.parameters():
    param.requires_grad = False

params_to_track = []

for layer_idx in [-2, -1]:
    layer = model.gpt_neox.layers[layer_idx]
    for param in layer.parameters():
        param.requires_grad = True
        params_to_track.append(param)

for param in model.embed_out.parameters():
    param.requires_grad = True
    params_to_track.append(param)

# ============================================================
# Load training data (32 examples: 16 per class) ← CHANGED!
# ============================================================

sst2_train = load_dataset("glue", "sst2", split="train")

# Sample balanced dataset: 16 positive, 16 negative
positive_examples = [ex for ex in sst2_train if ex["label"] == 1][:16]  # ← Changed from 8
negative_examples = [ex for ex in sst2_train if ex["label"] == 0][:16]  # ← Changed from 8
train_data = positive_examples + negative_examples

N = len(train_data)
print(f"Computing kernel for {N} examples...")

# ============================================================
# Compute gradients for all training examples
# ============================================================

def compute_gradient(text):
    """Compute ∇f for one example"""
    prompt = f"Classify the sentiment:\n{text}\n\nSentiment:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    model.zero_grad()
    
    outputs = model(**inputs)
    logits = outputs.logits[0, -1, :]
    f_x = logits[positive_token_id] - logits[negative_token_id]
    
    f_x.backward()
    
    grad_list = []
    for param in params_to_track:
        if param.grad is not None:
            grad_list.append(param.grad.flatten().detach())
    
    return torch.cat(grad_list)

# Compute all gradients
print("Computing gradients...")
gradients = []

for i, example in enumerate(train_data):
    text = example["sentence"]
    grad = compute_gradient(text)
    gradients.append(grad.cpu())
    
    if (i + 1) % 8 == 0:
        print(f"  {i+1}/{N} done")

# Stack into matrix
gradients = torch.stack(gradients)
print(f"Gradient matrix shape: {gradients.shape}")

# ============================================================
# Compute kernel matrix
# ============================================================

print("\nComputing kernel matrix...")
K = gradients @ gradients.T

print(f"Kernel matrix shape: {K.shape}")

# Save for later
torch.save({
    'kernel': K,
    'labels': [ex['label'] for ex in train_data],
    'texts': [ex['sentence'] for ex in train_data]
}, 'entk_32shot.pt')

print("\nKernel saved to 'entk_32shot.pt'")