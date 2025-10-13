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


model_name = "EleutherAI/pythia-70m"

#loading model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = GPTNeoXForCausalLM.from_pretrained(model_name)
model.eval()

#set device
device = torch.device("cuda" if torch.cuda.is_available() else cpu)
#Move model to device
model = model.to(device)

#text example to be used
text = "this film is extraordinarily horrendous"
prompt = f"Review:{text} as positive or negative sentiment\nSentiment:" 


#Tokenize prompt and generate continuation
inputs = tokenizer(prompt, return_tensors="pt").to(device)


outputs = model.generate(
    **inputs,
    max_new_tokens = 20,
    temperature = 0.7,
    top_p = 0.9,
    do_sample = True
)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)



