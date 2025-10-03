import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score
import random
from typing import List, Tuple, Optional
from torch.utils.data import Dataset
import logging
from tqdm import tqdm
import gc
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SST2Dataset(Dataset):
    """Dataset wrapper for SST-2 with prompting for autoregressive models."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Label words for SST-2 (matching paper's Table 3)
        self.label_words = ["terrible", "great"]  # 0: negative, 1: positive
        self.label_token_ids = [
            tokenizer.encode(word, add_special_tokens=False)[0] 
            for word in self.label_words
        ]
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        prompted_text = f"{text} It was"
        
        encoding = self.tokenizer(
            prompted_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        non_pad_positions = (input_ids != self.tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
        mask_pos = non_pad_positions[-1].item()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'mask_pos': mask_pos,
            'label': label,
        }

class StreamingKernelComputer:
    """
    Memory-efficient kernel computation that never stores full gradients.
    Computes kernel entries on-the-fly and saves to disk.
    """
    
    def __init__(self, model, tokenizer, label_token_ids, device='cuda', use_sign_kernel=True):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.label_token_ids = label_token_ids
        self.device = device
        self.use_sign_kernel = use_sign_kernel
        
        for param in self.model.parameters():
            param.requires_grad_(True)
    
    def compute_single_gradient(self, input_ids, attention_mask, position, token_id):
        """Compute gradient without storing - return as generator for memory efficiency."""
        self.model.zero_grad()
        
        outputs = self.model(input_ids=input_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0))
        logits = outputs.logits
        target_logit = logits[0, position, token_id]
        target_logit.backward()
        
        # Yield gradients one parameter at a time (never materialize full vector)
        for param in self.model.parameters():
            if param.grad is not None:
                grad = param.grad.view(-1)
                if self.use_sign_kernel:
                    yield torch.sign(grad)
                else:
                    yield grad
            else:
                yield torch.zeros(param.numel(), device=self.device)
    
    def compute_kernel_entry(self, example1, example2, class1_idx, class2_idx):
        """
        Compute single kernel entry K[i,j] without storing gradients.
        Returns scalar value.
        """
        token1 = self.label_token_ids[class1_idx]
        token2 = self.label_token_ids[class2_idx]
        
        # Compute first gradient
        grad1_gen = self.compute_single_gradient(
            example1['input_ids'].to(self.device),
            example1['attention_mask'].to(self.device),
            example1['mask_pos'],
            token1
        )
        grad1_chunks = [g.detach() for g in grad1_gen]
        
        # Compute second gradient
        grad2_gen = self.compute_single_gradient(
            example2['input_ids'].to(self.device),
            example2['attention_mask'].to(self.device),
            example2['mask_pos'],
            token2
        )
        
        # Compute dot product incrementally
        kernel_val = 0.0
        for g1, g2 in zip(grad1_chunks, grad2_gen):
            kernel_val += torch.dot(g1, g2.detach()).item()
        
        # Cleanup
        del grad1_chunks
        torch.cuda.empty_cache()
        
        return kernel_val
    
    def compute_kernel_matrix_streaming(self, dataset1: Dataset, dataset2: Dataset = None):
        """
        Compute kernel matrix by computing each entry individually.
        Returns kernel matrix without ever storing gradients.
        
        If dataset2 is None, computes K(dataset1, dataset1).
        Otherwise computes K(dataset1, dataset2) (cross-kernel).
        """
        if dataset2 is None:
            dataset2 = dataset1
            is_symmetric = True
        else:
            is_symmetric = False
        
        n1 = len(dataset1)
        n2 = len(dataset2)
        num_classes = len(self.label_token_ids)
        
        # Full kernel size
        size1 = n1 * num_classes
        size2 = n2 * num_classes
        
        kernel_matrix = np.zeros((size1, size2), dtype=np.float32)
        
        logger.info(f"Computing kernel matrix of size {size1} x {size2}")
        
        # Compute each block
        for i in tqdm(range(n1), desc="Computing kernel rows"):
            ex1 = dataset1[i]
            
            start_j = i if is_symmetric else 0
            for j in range(start_j, n2):
                ex2 = dataset2[j]
                
                # Compute 2x2 block for this example pair
                for c1 in range(num_classes):
                    for c2 in range(num_classes):
                        row_idx = i * num_classes + c1
                        col_idx = j * num_classes + c2
                        
                        kernel_val = self.compute_kernel_entry(ex1, ex2, c1, c2)
                        kernel_matrix[row_idx, col_idx] = kernel_val
                        
                        if is_symmetric and i != j:
                            # Fill symmetric entry
                            kernel_matrix[col_idx, row_idx] = kernel_val
            
            # Periodic cleanup
            if i % 5 == 0:
                gc.collect()
                torch.cuda.empty_cache()
        
        return kernel_matrix

class KernelRegressor:
    """Kernel regression solver."""
    
    def __init__(self, kernel_matrix: np.ndarray):
        self.kernel_matrix = kernel_matrix
        self.reg_params = [0.0, 0.001, 0.01, 0.1, 1.0]
        self.f0_scaling = [10, 100, 1000, 10000, np.inf]
    
    def fit_and_predict(
        self, 
        train_labels: np.ndarray,
        pretrain_outputs: np.ndarray,
        test_kernel: np.ndarray,
        test_pretrain_outputs: np.ndarray,
        val_kernel: Optional[np.ndarray] = None,
        val_labels: Optional[np.ndarray] = None,
        val_pretrain_outputs: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float, float]:
        num_train = len(train_labels)
        num_classes = pretrain_outputs.shape[1]
        
        # Create one-hot labels
        expanded_labels = []
        for label in train_labels:
            one_hot = np.zeros(num_classes)
            one_hot[label] = 1.0
            expanded_labels.extend(one_hot)
        expanded_labels = np.array(expanded_labels)
        
        best_reg = 0.0
        best_f0 = np.inf
        best_score = -float('inf')
        
        logger.info("Hyperparameter search...")
        for reg in self.reg_params:
            for f0 in self.f0_scaling:
                if f0 == np.inf:
                    scaled_labels = expanded_labels
                else:
                    scaled_labels = expanded_labels * f0
                
                reg_kernel = self.kernel_matrix + reg * np.eye(len(self.kernel_matrix))
                
                try:
                    alpha = np.linalg.solve(reg_kernel, scaled_labels)
                except np.linalg.LinAlgError:
                    continue
                
                if val_kernel is not None and val_labels is not None:
                    val_output = val_kernel @ alpha
                    
                    if f0 != np.inf:
                        val_pretrain_flat = val_pretrain_outputs.flatten()
                        val_output = val_output + val_pretrain_flat
                    
                    val_probs = val_output.reshape(-1, num_classes)
                    val_pred = np.argmax(val_probs, axis=1)
                    val_score = accuracy_score(val_labels, val_pred)
                    
                    if val_score > best_score:
                        best_score = val_score
                        best_reg = reg
                        best_f0 = f0
        
        logger.info(f"Best: reg={best_reg}, f0={best_f0}, val_acc={best_score:.3f}")
        
        # Fit with best hyperparameters
        if best_f0 == np.inf:
            scaled_labels = expanded_labels
        else:
            scaled_labels = expanded_labels * best_f0
        
        reg_kernel = self.kernel_matrix + best_reg * np.eye(len(self.kernel_matrix))
        alpha = np.linalg.solve(reg_kernel, scaled_labels)
        
        # Predict
        test_output = test_kernel @ alpha
        
        if best_f0 != np.inf:
            test_pretrain_flat = test_pretrain_outputs.flatten()
            test_output = test_output + test_pretrain_flat
        
        test_probs = test_output.reshape(-1, num_classes)
        test_pred = np.argmax(test_probs, axis=1)
        
        return test_pred, best_reg, best_f0

def get_pretrained_outputs(model, dataset, label_token_ids, device='cuda', batch_size=16):
    """Get pre-trained model outputs."""
    all_outputs = []
    model.eval()
    
    with torch.no_grad():
        for i in tqdm(range(0, len(dataset), batch_size), desc="Pre-trained outputs"):
            batch_outputs = []
            for j in range(i, min(i + batch_size, len(dataset))):
                ex = dataset[j]
                input_ids = ex['input_ids'].unsqueeze(0).to(device)
                attention_mask = ex['attention_mask'].unsqueeze(0).to(device)
                mask_pos = ex['mask_pos']
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                label_logits = logits[0, mask_pos, label_token_ids].cpu().numpy()
                batch_outputs.append(label_logits)
            
            all_outputs.extend(batch_outputs)
    
    return np.array(all_outputs)

def load_sst2_data(num_shots: int = 16, seed: int = 42):
    """Load SST-2 data."""
    random.seed(seed)
    np.random.seed(seed)
    
    dataset = load_dataset("sst2")
    train_data = dataset['train']
    
    pos_examples = [(ex['sentence'], ex['label']) for ex in train_data if ex['label'] == 1]
    neg_examples = [(ex['sentence'], ex['label']) for ex in train_data if ex['label'] == 0]
    
    sampled_pos = random.sample(pos_examples, num_shots)
    sampled_neg = random.sample(neg_examples, num_shots)
    
    train_texts = [ex[0] for ex in sampled_pos + sampled_neg]
    train_labels = [ex[1] for ex in sampled_pos + sampled_neg]
    
    remaining_pos = [ex for ex in pos_examples if ex not in sampled_pos]
    remaining_neg = [ex for ex in neg_examples if ex not in sampled_neg]
    
    val_pos = random.sample(remaining_pos, num_shots)
    val_neg = random.sample(remaining_neg, num_shots)
    
    val_texts = [ex[0] for ex in val_pos + val_neg]
    val_labels = [ex[1] for ex in val_pos + val_neg]
    
    test_data = dataset['validation']
    test_texts = [ex['sentence'] for ex in test_data][:872]
    test_labels = [ex['label'] for ex in test_data][:872]
    
    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels

def main():
    MODEL_NAME = "EleutherAI/pythia-70m"
    NUM_SHOTS = 16
    SEED = 42
    USE_SIGN_KERNEL = True
    
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    
    logger.info(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    
    logger.info("Loading SST-2 data...")
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = load_sst2_data(
        num_shots=NUM_SHOTS, seed=SEED
    )
    
    logger.info(f"Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")
    
    train_dataset = SST2Dataset(train_texts, train_labels, tokenizer)
    val_dataset = SST2Dataset(val_texts, val_labels, tokenizer)
    test_dataset = SST2Dataset(test_texts, test_labels, tokenizer)
    
    label_token_ids = train_dataset.label_token_ids
    logger.info(f"Label tokens: {train_dataset.label_words} -> {label_token_ids}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")
    logger.info(f"Kernel: {'SignGD' if USE_SIGN_KERNEL else 'SGD'}")
    
    kernel_computer = StreamingKernelComputer(
        model, tokenizer, label_token_ids, 
        device=device, use_sign_kernel=USE_SIGN_KERNEL
    )
    
    # Get pre-trained outputs
    logger.info("Getting pre-trained outputs...")
    train_pretrain = get_pretrained_outputs(model, train_dataset, label_token_ids, device)
    val_pretrain = get_pretrained_outputs(model, val_dataset, label_token_ids, device)
    test_pretrain = get_pretrained_outputs(model, test_dataset, label_token_ids, device)
    
    # Compute kernels (streaming - no gradient storage)
    logger.info("Computing train kernel...")
    train_kernel = kernel_computer.compute_kernel_matrix_streaming(train_dataset)
    
    logger.info("Computing val-train cross-kernel...")
    val_train_kernel = kernel_computer.compute_kernel_matrix_streaming(val_dataset, train_dataset)
    
    logger.info("Computing test-train cross-kernel...")
    test_train_kernel = kernel_computer.compute_kernel_matrix_streaming(test_dataset, train_dataset)
    
    # Regression
    logger.info("Kernel regression...")
    regressor = KernelRegressor(train_kernel)
    
    test_pred, best_reg, best_f0 = regressor.fit_and_predict(
        train_labels, train_pretrain,
        test_train_kernel, test_pretrain,
        val_train_kernel, val_labels, val_pretrain
    )
    
    test_accuracy = accuracy_score(test_labels, test_pred)
    
    logger.info("=" * 60)
    logger.info(f"SST-2 {NUM_SHOTS}-shot Results:")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Kernel: {'SignGD' if USE_SIGN_KERNEL else 'SGD'}")
    logger.info(f"Best reg: {best_reg}, f0: {best_f0}")
    logger.info(f"Test Accuracy: {test_accuracy:.3f}")
    logger.info("=" * 60)
    
    return {'test_accuracy': test_accuracy, 'best_reg': best_reg, 'best_f0': best_f0}

if __name__ == "__main__":
    results = main()