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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SST2Dataset(Dataset):
    """Dataset wrapper for SST-2 with prompting."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Label mapping for SST-2
        self.label_words = ["terrible", "great"]  # 0: negative, 1: positive
        self.label_token_ids = [tokenizer.encode(word, add_special_tokens=False)[0] for word in self.label_words]
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Create prompt: "<text> It was [MASK]." for masked LM or "<text> It was" for autoregressive
        if hasattr(self.tokenizer, 'mask_token') and self.tokenizer.mask_token is not None:
            # For masked LM (BERT-style)
            prompted_text = f"{text} It was {self.tokenizer.mask_token}."
            encoding = self.tokenizer(
                prompted_text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            # Find mask position
            mask_positions = (encoding['input_ids'] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
            mask_pos = mask_positions[0] if len(mask_positions) > 0 else torch.tensor(1)
        else:
            # For autoregressive models (Pythia-style)
            prompted_text = f"{text} It was"
            encoding = self.tokenizer(
                prompted_text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            # Use last non-pad position (where we want to predict next token)
            input_ids = encoding['input_ids'].squeeze()
            non_pad_length = (input_ids != self.tokenizer.pad_token_id).sum().item()
            mask_pos = torch.tensor(non_pad_length - 1)  # Position where we predict next token
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'mask_pos': mask_pos,
            'label': torch.tensor(label, dtype=torch.long),
            'label_token_ids': torch.tensor(self.label_token_ids)
        }

class KernelComputer:
    """Computes empirical Neural Tangent Kernel for fine-tuning."""
    
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model.to(device)
        self.model.train()  # Ensure model is in training mode
        self.tokenizer = tokenizer
        self.device = device
        
        # Ensure all parameters require gradients
        for param in self.model.parameters():
            param.requires_grad_(True)
        
    def compute_gradients(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute gradients of model output w.r.t. parameters."""        
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        mask_pos = batch['mask_pos'].to(self.device)
        label_token_ids = batch['label_token_ids'].to(self.device)
        
        # We need gradients for each example and class separately
        all_gradients = []
        batch_size = input_ids.shape[0]
        num_classes = label_token_ids.shape[1]
        
        for i in range(batch_size):
            for j in range(num_classes):  # for each class
                # Clear previous gradients
                self.model.zero_grad()
                
                # Forward pass for single example with gradient tracking
                single_input = input_ids[i:i+1]  # Keep batch dimension
                single_mask = attention_mask[i:i+1]
                single_pos = mask_pos[i:i+1]
                
                # Forward pass with explicit gradient computation
                with torch.enable_grad():
                    outputs = self.model(input_ids=single_input, attention_mask=single_mask)
                    logits = outputs.logits
                    
                    # Extract logit for specific label token at mask position
                    # Use more explicit indexing to ensure gradient flow
                    batch_idx = 0
                    pos_idx = single_pos[0].item()
                    token_idx = label_token_ids[0, j].item()
                    
                    target_logit = logits[batch_idx, pos_idx, token_idx]
                    
                    # Compute gradients using autograd
                    grads = torch.autograd.grad(
                        outputs=[target_logit],
                        inputs=list(self.model.parameters()),
                        create_graph=False,
                        retain_graph=False,
                        allow_unused=True  # Some parameters might not affect this specific output
                    )
                
                # Collect gradients (handle None gradients for unused parameters)
                grad_list = []
                for grad in grads:
                    if grad is not None:
                        grad_list.append(grad.view(-1).detach().cpu())
                    else:
                        # For parameters not used in this computation, use zeros
                        # Get parameter size from model
                        param_size = next(iter(self.model.parameters())).numel()
                        grad_list.append(torch.zeros(param_size, device='cpu'))
                
                # Handle case where gradients have different sizes
                total_params = sum(p.numel() for p in self.model.parameters())
                final_grad = torch.zeros(total_params)
                
                start_idx = 0
                for param, grad in zip(self.model.parameters(), grads):
                    param_size = param.numel()
                    if grad is not None:
                        final_grad[start_idx:start_idx + param_size] = grad.view(-1).detach().cpu()
                    # If grad is None, leave as zeros
                    start_idx += param_size
                
                all_gradients.append(final_grad)
        
        # Reshape: [batch_size, num_classes, num_params]
        gradients = torch.stack(all_gradients).view(batch_size, num_classes, -1)
        return gradients
    
    def compute_kernel_matrix(self, dataset: Dataset, batch_size: int = 8) -> np.ndarray:
        """Compute the full kernel matrix for the dataset."""
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        all_gradients = []
        
        logger.info("Computing gradients for kernel matrix...")
        for batch_idx, batch in enumerate(dataloader):
            logger.info(f"Processing batch {batch_idx + 1}/{len(dataloader)}")
            
            with torch.no_grad():
                gradients = self.compute_gradients(batch)  # [batch_size, num_classes, num_params]
                all_gradients.append(gradients)
        
        # Concatenate all gradients: [total_samples, num_classes, num_params]
        all_gradients = torch.cat(all_gradients, dim=0)
        
        # For binary classification, flatten to [total_samples * num_classes, num_params]
        num_samples, num_classes, num_params = all_gradients.shape
        flattened_grads = all_gradients.view(-1, num_params)  # [total_samples * 2, num_params]
        
        logger.info("Computing kernel matrix...")
        # Compute kernel matrix K[i,j] = <grad_i, grad_j>
        kernel_matrix = torch.mm(flattened_grads, flattened_grads.T).cpu().numpy()
        
        return kernel_matrix, all_gradients.cpu().numpy()

class KernelRegressor:
    """Kernel regression solver following the paper's methodology."""
    
    def __init__(self, kernel_matrix: np.ndarray, regularization_params: List[float] = None):
        self.kernel_matrix = kernel_matrix
        if regularization_params is None:
            self.regularization_params = [0.0, 0.001, 0.01, 0.1, 1.0]
        else:
            self.regularization_params = regularization_params
    
    def fit_and_predict(self, train_labels: np.ndarray, test_kernel: np.ndarray, 
                       val_kernel: Optional[np.ndarray] = None, 
                       val_labels: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
        """
        Fit kernel regression and make predictions.
        
        Args:
            train_labels: Training labels [num_train_samples * num_classes]
            test_kernel: Kernel between test and train [num_test_samples * num_classes, num_train_samples * num_classes]
            val_kernel: Optional validation kernel for hyperparameter selection
            val_labels: Optional validation labels
        """
        # Prepare training data
        # For binary classification, we treat it as C separate binary problems
        num_train = len(train_labels)
        num_classes = 2
        
        # Create expanded labels: [num_train * num_classes]
        expanded_labels = []
        for label in train_labels:
            # One-hot encoding
            class_labels = np.zeros(num_classes)
            class_labels[label] = 1.0
            expanded_labels.extend(class_labels)
        
        expanded_labels = np.array(expanded_labels)
        
        best_reg = 0.0
        best_score = -float('inf')
        
        # Hyperparameter search
        for reg in self.regularization_params:
            # Solve kernel regression: (K + reg*I) * alpha = y
            reg_kernel = self.kernel_matrix + reg * np.eye(len(self.kernel_matrix))
            
            try:
                alpha = np.linalg.solve(reg_kernel, expanded_labels)
                
                # Validate if validation set provided
                if val_kernel is not None and val_labels is not None:
                    val_pred_raw = val_kernel @ alpha
                    val_pred_probs = val_pred_raw.reshape(-1, num_classes)
                    val_pred = np.argmax(val_pred_probs, axis=1)
                    val_score = accuracy_score(val_labels, val_pred)
                    
                    if val_score > best_score:
                        best_score = val_score
                        best_reg = reg
                else:
                    # Use training performance if no validation set
                    train_pred_raw = self.kernel_matrix @ alpha
                    train_pred_probs = train_pred_raw.reshape(-1, num_classes)
                    train_pred = np.argmax(train_pred_probs, axis=1)
                    train_score = accuracy_score(expanded_labels.reshape(-1, num_classes).argmax(axis=1), train_pred)
                    
                    if train_score > best_score:
                        best_score = train_score
                        best_reg = reg
                        
            except np.linalg.LinAlgError:
                logger.warning(f"Singular matrix for regularization {reg}")
                continue
        
        # Fit with best regularization
        reg_kernel = self.kernel_matrix + best_reg * np.eye(len(self.kernel_matrix))
        alpha = np.linalg.solve(reg_kernel, expanded_labels)
        
        # Make test predictions
        test_pred_raw = test_kernel @ alpha
        test_pred_probs = test_pred_raw.reshape(-1, num_classes)
        test_pred = np.argmax(test_pred_probs, axis=1)
        
        return test_pred, best_reg

def load_sst2_data(num_shots: int = 16, seed: int = 42) -> Tuple[List[str], List[int], List[str], List[int], List[str], List[int]]:
    """Load and prepare SST-2 data in few-shot setting."""
    random.seed(seed)
    np.random.seed(seed)
    
    # Load dataset
    dataset = load_dataset("sst2")
    
    # Prepare train data (few-shot sampling)
    train_data = dataset['train']
    
    # Sample k examples per class
    pos_examples = [(ex['sentence'], ex['label']) for ex in train_data if ex['label'] == 1]
    neg_examples = [(ex['sentence'], ex['label']) for ex in train_data if ex['label'] == 0]
    
    sampled_pos = random.sample(pos_examples, min(num_shots, len(pos_examples)))
    sampled_neg = random.sample(neg_examples, min(num_shots, len(neg_examples)))
    
    train_texts = [ex[0] for ex in sampled_pos + sampled_neg]
    train_labels = [ex[1] for ex in sampled_pos + sampled_neg]
    
    # Prepare validation data (same size as training for hyperparameter selection)
    remaining_pos = [ex for ex in pos_examples if ex not in sampled_pos]
    remaining_neg = [ex for ex in neg_examples if ex not in sampled_neg]
    
    val_pos = random.sample(remaining_pos, min(num_shots, len(remaining_pos)))
    val_neg = random.sample(remaining_neg, min(num_shots, len(remaining_neg)))
    
    val_texts = [ex[0] for ex in val_pos + val_neg]
    val_labels = [ex[1] for ex in val_pos + val_neg]
    
    # Test data (limit to 1000 examples as in paper)
    test_data = dataset['validation']  # SST-2 uses 'validation' as test set
    test_examples = [(ex['sentence'], ex['label']) for ex in test_data]
    if len(test_examples) > 1000:
        test_examples = random.sample(test_examples, 1000)
    
    test_texts = [ex[0] for ex in test_examples]
    test_labels = [ex[1] for ex in test_examples]
    
    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels

def main():
    # Configuration
    MODEL_NAME = "EleutherAI/pythia-70m"  # Change to pythia-1b if desired
    NUM_SHOTS = 16
    SEED = 42
    BATCH_SIZE = 1  # Adjust based on GPU memory
    
    logger.info(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Add pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    if 'pythia' in MODEL_NAME.lower():
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    else:
        model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
    
    # Load data
    logger.info("Loading SST-2 data...")
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = load_sst2_data(
        num_shots=NUM_SHOTS, seed=SEED
    )
    
    logger.info(f"Data sizes - Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")
    
    # Create datasets
    train_dataset = SST2Dataset(train_texts, train_labels, tokenizer)
    val_dataset = SST2Dataset(val_texts, val_labels, tokenizer)
    test_dataset = SST2Dataset(test_texts, test_labels, tokenizer)
    
    # Initialize kernel computer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    kernel_computer = KernelComputer(model, tokenizer, device=device)
    
    # Compute kernel matrices
    logger.info("Computing training kernel matrix...")
    train_kernel, train_gradients = kernel_computer.compute_kernel_matrix(train_dataset, batch_size=BATCH_SIZE)
    
    logger.info("Computing validation kernel matrix...")
    val_kernel, val_gradients = kernel_computer.compute_kernel_matrix(val_dataset, batch_size=BATCH_SIZE)
    
    logger.info("Computing test kernel matrix...")
    test_kernel, test_gradients = kernel_computer.compute_kernel_matrix(test_dataset, batch_size=BATCH_SIZE)
    
    # Compute cross-kernel matrices (test vs train, val vs train)
    logger.info("Computing cross-kernel matrices...")
    test_train_kernel = np.dot(test_gradients.reshape(len(test_texts) * 2, -1), 
                               train_gradients.reshape(len(train_texts) * 2, -1).T)
    val_train_kernel = np.dot(val_gradients.reshape(len(val_texts) * 2, -1),
                              train_gradients.reshape(len(train_texts) * 2, -1).T)
    
    # Kernel regression
    logger.info("Performing kernel regression...")
    regressor = KernelRegressor(train_kernel)
    
    test_pred, best_reg = regressor.fit_and_predict(
        train_labels, test_train_kernel, val_train_kernel, val_labels
    )
    
    # Evaluate
    test_accuracy = accuracy_score(test_labels, test_pred)
    test_f1 = f1_score(test_labels, test_pred, average='macro')
    
    logger.info(f"Results:")
    logger.info(f"Best regularization: {best_reg}")
    logger.info(f"Test Accuracy: {test_accuracy:.3f}")
    logger.info(f"Test F1: {test_f1:.3f}")
    
    # Compare to random baseline
    random_pred = np.random.choice([0, 1], size=len(test_labels))
    random_accuracy = accuracy_score(test_labels, random_pred)
    logger.info(f"Random baseline accuracy: {random_accuracy:.3f}")
    
    return {
        'test_accuracy': test_accuracy,
        'test_f1': test_f1,
        'best_regularization': best_reg,
        'random_baseline': random_accuracy
    }

if __name__ == "__main__":
    results = main()