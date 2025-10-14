import torch
from datasets import load_dataset
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score
import numpy as np
from transformers import AutoTokenizer, GPTNeoXForCausalLM

# ============================================================
# Load the kernel we computed
# ============================================================

data = torch.load('/home/acarbol1/scratchenalisn1/acarbol1/NTK_safety_finetuning/entk_32shot.pt')
K_train = data['kernel'].numpy()  # [16, 16]
y_train = np.array(data['labels'])  # [16]
train_texts = data['texts']

# Convert labels to {-1, +1} for regression
y_train_signed = 2 * y_train - 1  # 0→-1, 1→+1

print(f"Kernel shape: {K_train.shape}")
print(f"Labels: {y_train}")
print(f"Signed labels: {y_train_signed}")

# ============================================================
# Solve kernel ridge regression: min ||Kα - y||² + λ||α||²
# ============================================================

# Try different regularization strengths
lambdas = [0.001, 0.01, 0.1, 1.0, 10.0]

print("\n" + "="*70)
print("Training kernel regression...")
print("="*70)

best_alpha = None
best_lambda = None
best_train_acc = 0

for lam in lambdas:
    # Solve (K + λI)α = y
    ridge = Ridge(alpha=lam, fit_intercept=False)
    ridge.fit(K_train, y_train_signed)
    
    alpha = ridge.coef_
    
    # Predictions on training set
    y_pred_train = K_train @ alpha
    y_pred_train_binary = (y_pred_train > 0).astype(int)
    
    train_acc = accuracy_score(y_train, y_pred_train_binary)
    
    print(f"λ={lam:6.3f} | Train Acc: {train_acc*100:.1f}%")
    
    if train_acc > best_train_acc:
        best_train_acc = train_acc
        best_alpha = alpha
        best_lambda = lam

print(f"\nBest λ: {best_lambda} (Train Acc: {best_train_acc*100:.1f}%)")

# ============================================================
# Now evaluate on TEST set
# ============================================================

print("\n" + "="*70)
print("Computing test set kernel...")
print("="*70)

# Load model again for test evaluation
model_name = "EleutherAI/pythia-160m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = GPTNeoXForCausalLM.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Setup gradients same as before
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

positive_token_id = 2762
negative_token_id = 4016

def compute_gradient(text):
    """Compute gradient for one example"""
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
    
    return torch.cat(grad_list).cpu()

# Load test set
sst2_test = load_dataset("glue", "sst2", split="validation[:100]")

print(f"Computing kernel for {len(sst2_test)} test examples...")

# Compute K_test: [N_test, N_train]
# K_test[i,j] = <grad_test[i], grad_train[j]>

# Load training gradients
train_grads = []
for text in train_texts:
    grad = compute_gradient(text)
    train_grads.append(grad)
train_grads = torch.stack(train_grads)  # [16, num_params]

# Compute test gradients and kernel
test_labels = []
test_predictions = []

for i, example in enumerate(sst2_test):
    test_grad = compute_gradient(example['sentence'])
    
    # Compute kernel with all training examples
    k_test = (test_grad @ train_grads.T).numpy()  # [16]
    
    # Predict: f(x_test) ≈ Σᵢ αᵢ K(x_test, x_train_i)
    y_pred = k_test @ best_alpha
    
    test_labels.append(example['label'])
    test_predictions.append(1 if y_pred > 0 else 0)
    
    if (i + 1) % 20 == 0:
        print(f"  {i+1}/{len(sst2_test)} done")

# ============================================================
# Results
# ============================================================

test_acc = accuracy_score(test_labels, test_predictions)

print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"Train accuracy (kernel): {best_train_acc*100:.1f}%")
print(f"Test accuracy (kernel):  {test_acc*100:.1f}%")
print("\nThis is the eNTK performance!")
print("Next: Compare to actual fine-tuning to see if they match")