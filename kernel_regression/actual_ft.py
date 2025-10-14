from transformers import AutoTokenizer, GPTNeoXForCausalLM, Trainer, TrainingArguments
import torch
from datasets import load_dataset, Dataset
import numpy as np

model_name = "EleutherAI/pythia-160m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

positive_token_id = 2762
negative_token_id = 4016

# ============================================================
# Load same 16 examples as kernel
# ============================================================

data = torch.load('entk_16shot.pt', weights_only=False)
train_texts = data['texts']
train_labels = data['labels']

print(f"Loaded {len(train_texts)} training examples")

# Prepare datasets - use lists directly, not Dataset.from_dict
train_prompts = [f"Classify the sentiment:\n{text}\n\nSentiment:" for text in train_texts]

# Create simple list of dicts
train_data = [{'text': prompt, 'label': label} for prompt, label in zip(train_prompts, train_labels)]

sst2_test = load_dataset("glue", "sst2", split="validation[:100]")
test_prompts = [f"Classify the sentiment:\n{ex['sentence']}\n\nSentiment:" for ex in sst2_test]
test_data = [{'text': prompt, 'label': ex['label']} for prompt, ex in zip(test_prompts, sst2_test)]

# ============================================================
# Simple custom dataset class
# ============================================================

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

train_dataset = SimpleDataset(train_data)
test_dataset = SimpleDataset(test_data)

# Test the dataset
print(f"Train dataset sample: {train_dataset[0]}")

# ============================================================
# Data collator
# ============================================================

def collate_fn(batch):
    # Now batch is a list of dicts with 'text' and 'label'
    texts = [item['text'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch])
    
    encodings = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    
    return {
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'labels': labels
    }

# ============================================================
# Custom trainer with our loss function
# ============================================================

class SentimentTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop('labels')
        
        outputs = model(**inputs)
        logits = outputs.logits[:, -1, :]  # Last position
        
        pos_logits = logits[:, positive_token_id]
        neg_logits = logits[:, negative_token_id]
        
        # f(x) = pos - neg, target = +1 for positive, -1 for negative
        f_x = pos_logits - neg_logits
        target = 2 * labels.float() - 1
        
        # MSE loss
        loss = ((f_x - target) ** 2).mean()
        
        if return_outputs:
            predictions = torch.stack([neg_logits, pos_logits], dim=1)
            return loss, predictions
        
        return loss

# ============================================================
# Compute metrics
# ============================================================

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=-1)
    return {"accuracy": (preds == labels).mean()}

# ============================================================
# Training setup
# ============================================================

model = GPTNeoXForCausalLM.from_pretrained(model_name)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=50,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=16,
    learning_rate=1e-5,
    eval_strategy="epoch",
    save_strategy="no",
    logging_steps=5,
    report_to="none",
    seed=42,
    dataloader_num_workers=0  # Add this to avoid multiprocessing issues
)

trainer = SentimentTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=collate_fn,
    compute_metrics=compute_metrics
)

# ============================================================
# Train and evaluate
# ============================================================

print("="*70)
print("Fine-tuning on 16 examples...")
print("="*70)

trainer.train()

# Evaluate on both train and test
train_results = trainer.evaluate(train_dataset)
test_results = trainer.evaluate(test_dataset)

print("\n" + "="*70)
print("FINE-TUNING RESULTS (16-shot)")
print("="*70)
print(f"Train accuracy: {train_results['eval_accuracy']*100:.1f}%")
print(f"Test accuracy:  {test_results['eval_accuracy']*100:.1f}%")
print("\n" + "="*70)
print("COMPARISON TO KERNEL")
print("="*70)
print(f"Kernel (eNTK):  Train: 100.0% | Test: 48.0%")
print(f"Fine-tuning:    Train: {train_results['eval_accuracy']*100:.1f}% | Test: {test_results['eval_accuracy']*100:.1f}%")
print("\nIf these are similar, the kernel is working correctly!")