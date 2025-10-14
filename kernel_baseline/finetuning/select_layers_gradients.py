from transformers import AutoTokenizer, GPTNeoXForCausalLM
import torch
from datasets import load_dataset

model_name = "EleutherAI/pythia-160m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = GPTNeoXForCausalLM.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

positive_token_id = 2762 
negative_token_id = 4016

# Load one example
sst2 = load_dataset("glue", "sst2", split="train[:1]")
text = sst2[0]["sentence"]

# Use our best prompt
prompt = f"Classify the sentiment:\n{text}\n\nSentiment:"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# ============================================================
# Enable gradients for last 2 layers + output embedding
# ============================================================

# First, disable all gradients
for param in model.parameters():
    param.requires_grad = False

# Enable only for last 2 transformer layers + output embedding
params_to_track = []

# Last 2 transformer layers
for layer_idx in [-2, -1]:
    layer = model.gpt_neox.layers[layer_idx]
    for param in layer.parameters():
        param.requires_grad = True
        params_to_track.append(param)

# Output embedding
for param in model.embed_out.parameters():
    param.requires_grad = True
    params_to_track.append(param)

print(f"Tracking {len(params_to_track)} parameter tensors")

# ============================================================
# Forward pass and compute f(x)
# ============================================================

outputs = model(**inputs)
logits = outputs.logits[0, -1, :]  # Last position

# f(x) = logit(positive) - logit(negative)
f_x = logits[positive_token_id] - logits[negative_token_id]

print(f"\nExample: {text}")
print(f"f(x) = {f_x.item():.3f}")

# ============================================================
# Compute gradients
# ============================================================

f_x.backward()

# Extract and flatten all gradients
gradient_vector = []
total_params = 0

for param in params_to_track:
    if param.grad is not None:
        gradient_vector.append(param.grad.flatten())
        total_params += param.grad.numel()
    else:
        print(f"WARNING: param has no gradient!")

gradient_vector = torch.cat(gradient_vector)

print(f"\n" + "="*70)
print(f"Gradient vector shape: {gradient_vector.shape}")
print(f"Total parameters tracked: {total_params:,}")
print(f"Gradient norm: {gradient_vector.norm().item():.3f}")
print(f"Gradient mean: {gradient_vector.mean().item():.6f}")
print(f"Gradient std: {gradient_vector.std().item():.6f}")
print("="*70)