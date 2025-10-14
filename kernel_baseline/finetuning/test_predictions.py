from transformers import AutoModel, AutoTokenizer, GPTNeoXForCausalLM
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM
from datasets import load_dataset
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, f1_score
import functools
import random
from typing import Dict, List, Tuple, Optional
from torch.utils.data import DataLoader, Dataset
import logging


model_name = "EleutherAI/pythia-160m"

#loading model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = GPTNeoXForCausalLM.from_pretrained(model_name)
model.eval()

#set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Move model to device
model = model.to(device)

# Check our label tokens
positive_token_id = 2762 
negative_token_id = 4016


# Load SST-2 validation set (we'll use first 100 for speed)
sst2_val = load_dataset("glue", "sst2", split="validation[:100]")

# Define different prompt templates
prompt_templates = [
    ("Original", lambda text: f"Review: {text}\nSentiment:"),
    ("Simple completion", lambda text: f"{text}\n\nThis review is"),
    ("Question", lambda text: f"Is this review positive or negative?\n{text}\nAnswer:"),
    ("Direct", lambda text: f"{text}\nSentiment:"),
    ("Instruction", lambda text: f"Classify the sentiment:\n{text}\n\nSentiment:"),
    ("Natural ending", lambda text: f'"{text}"\n\nThe sentiment of this review is'),
]

print("Testing different prompt formats on SST-2 validation set (100 examples)")
print("="*80)

results = []

for prompt_name, prompt_fn in prompt_templates:
    correct = 0
    chi_values = []
    
    for example in sst2_val:
        text = example["sentence"]
        label = example["label"]  # 0=negative, 1=positive
        
        prompt = prompt_fn(text)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]
        
        # Get logits for our labels
        pos_logit = logits[positive_token_id]
        neg_logit = logits[negative_token_id]
        
        # Compute f(x) = pos_logit - neg_logit
        f_x = pos_logit - neg_logit
        
        # Check if correct
        prediction = 1 if f_x > 0 else 0
        if prediction == label:
            correct += 1
        
        # Compute χ
        target = 1.0 if label == 1 else -1.0
        chi = 2 * (f_x - target)
        chi_values.append(abs(chi.item()))
    
    accuracy = correct / len(sst2_val) * 100
    avg_chi = sum(chi_values) / len(chi_values)
    
    results.append((prompt_name, accuracy, avg_chi))
    
    print(f"\n{prompt_name:20s}")
    print(f"  Accuracy: {accuracy:.1f}%")
    print(f"  Avg |χ|:  {avg_chi:.3f}")
    print(f"  Example:  {prompt_fn('great movie')}")

print("\n" + "="*80)
print("SUMMARY:")
print("="*80)

# Sort by accuracy
results.sort(key=lambda x: x[1], reverse=True)

for prompt_name, accuracy, avg_chi in results:
    status = "✓✓ GREAT!" if accuracy >= 65 else "✓ OK" if accuracy >= 55 else "✗ Poor"
    print(f"{prompt_name:20s} | Acc: {accuracy:5.1f}% | χ: {avg_chi:5.2f} | {status}")

print("\nBest prompt:", results[0][0])