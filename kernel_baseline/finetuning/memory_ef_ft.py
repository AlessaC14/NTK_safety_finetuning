import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import random
from typing import Dict, List, Tuple, Optional
from torch.utils.data import DataLoader, Dataset
import logging
from tqdm import tqdm
import gc

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
        
        # Prompt format from paper: "<text> It was [MASK]."
        # For autoregressive: "<text> It was" and predict next token
        prompted_text = f"{text} It was"
        
        encoding = self.tokenizer(
            prompted_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Find position where we predict the label token (last non-pad token)
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        non_pad_positions = (input_ids != self.tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
        mask_pos = non_pad_positions[-1].item()  # Last non-pad position
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'mask_pos': mask_pos,
            'label': label,
        }

class KernelComputer:
    """
    Computes empirical Neural Tangent Kernel following the paper's methodology.
    Implements both SGD kernel (standard NTK) and SignGD kernel (for Adam).
    """
    
    def __init__(self, model, tokenizer, label_token_ids, device='cuda', use_sign_kernel=True):
        self.model = model.to(device)
        self.model.eval()  # Use eval mode to avoid dropout randomness
        self.tokenizer = tokenizer
        self.label_token_ids = label_token_ids
        self.device = device
        self.use_sign_kernel = use_sign_kernel  # True for Adam/SignGD, False for SGD
        
        # Enable gradients for all parameters
        for param in self.model.parameters():
            param.requires_grad_(True)
    
    def compute_single_gradient(self, input_ids, attention_mask, position, token_id):
        """
        Compute gradient of f(ξ; θ) where f is the logit for token_id at position.
        Returns flattened gradient vector.
        """
        self.model.zero_grad()
        
        # Forward pass
        outputs = self.model(input_ids=input_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0))
        logits = outputs.logits
        
        # Extract target logit: f(ξ) = logits[0, position, token_id]
        target_logit = logits[0, position, token_id]
        
        # Backward pass
        target_logit.backward()
        
        # Collect gradients
        grad_list = []
        for param in self.model.parameters():
            if param.grad is not None:
                grad_list.append(param.grad.view(-1).detach())
            else:
                grad_list.append(torch.zeros(param.numel(), device=self.device))
        
        gradient = torch.cat(grad_list)
        
        # Apply sign function for SignGD kernel (Theorem 4.3)
        if self.use_sign_kernel:
            gradient = torch.sign(gradient)
        
        return gradient
    
    def compute_example_gradients(self, batch):
        """
        Compute gradients for all classes for each example in batch.
        Returns: [batch_size, num_classes, num_params]
        """
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        mask_pos = batch['mask_pos']
        
        batch_size = input_ids.shape[0]
        num_classes = len(self.label_token_ids)
        
        all_gradients = []
        
        for i in range(batch_size):
            example_grads = []
            for token_id in self.label_token_ids:
                grad = self.compute_single_gradient(
                    input_ids[i], 
                    attention_mask[i], 
                    mask_pos[i].item(), 
                    token_id
                )
                example_grads.append(grad.cpu())
                
                # Clear cache to avoid memory buildup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            all_gradients.append(torch.stack(example_grads))
        
        # Shape: [batch_size, num_classes, num_params]
        return torch.stack(all_gradients)
    
    def compute_kernel_matrix(self, dataset: Dataset, batch_size: int = 1):
        """
        Compute kernel matrix K where K[i,j] is a num_classes x num_classes block.
        Following paper's Appendix A.3 for multi-class problems.
        """
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        all_gradients = []
        
        logger.info("Computing gradients for kernel matrix...")
        for batch in tqdm(dataloader):
            gradients = self.compute_example_gradients(batch)
            all_gradients.append(gradients)
            
            # Aggressive memory cleanup
            del gradients
            gc.collect()
        
        # Concatenate: [num_examples, num_classes, num_params]
        all_gradients = torch.cat(all_gradients, dim=0)
        num_examples, num_classes, num_params = all_gradients.shape
        
        logger.info(f"Computing kernel matrix for {num_examples} examples...")
        
        # Reshape for kernel computation: [num_examples * num_classes, num_params]
        flat_grads = all_gradients.view(-1, num_params)
        
        # Compute kernel in chunks to avoid OOM
        kernel_size = num_examples * num_classes
        kernel_matrix = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        
        chunk_size = 100  # Adjust based on memory
        for i in tqdm(range(0, kernel_size, chunk_size), desc="Computing kernel"):
            end_i = min(i + chunk_size, kernel_size)
            chunk_i = flat_grads[i:end_i].to(self.device)
            
            for j in range(0, kernel_size, chunk_size):
                end_j = min(j + chunk_size, kernel_size)
                chunk_j = flat_grads[j:end_j].to(self.device)
                
                # K[i,j] = <grad_i, grad_j>
                kernel_block = torch.mm(chunk_i, chunk_j.T).cpu().numpy()
                kernel_matrix[i:end_i, j:end_j] = kernel_block
                
                del chunk_j
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            del chunk_i
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return kernel_matrix, all_gradients.numpy()

class KernelRegressor:
    """
    Kernel regression solver following paper's Appendix A.3.
    Handles multi-class classification as described in the paper.
    """
    
    def __init__(self, kernel_matrix: np.ndarray, f0_scaling: List[float] = None):
        self.kernel_matrix = kernel_matrix
        
        # Regularization parameters from paper (Table 4)
        self.reg_params = [0.0, 0.001, 0.01, 0.1, 1.0]
        
        # f0 scaling parameters from paper (Table 4)
        if f0_scaling is None:
            self.f0_scaling = [10, 100, 1000, 10000, np.inf]
        else:
            self.f0_scaling = f0_scaling
    
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
        """
        Fit kernel regression with hyperparameter search.
        
        Args:
            train_labels: [num_train]
            pretrain_outputs: [num_train, num_classes] - pre-trained model logits
            test_kernel: [num_test * num_classes, num_train * num_classes]
            test_pretrain_outputs: [num_test, num_classes]
            val_kernel: optional validation kernel
            val_labels: optional validation labels
            val_pretrain_outputs: optional validation pre-trained outputs
        """
        num_train = len(train_labels)
        num_classes = pretrain_outputs.shape[1]
        
        # Create one-hot labels: [num_train * num_classes]
        expanded_labels = []
        for label in train_labels:
            one_hot = np.zeros(num_classes)
            one_hot[label] = 1.0
            expanded_labels.extend(one_hot)
        expanded_labels = np.array(expanded_labels)
        
        best_reg = 0.0
        best_f0 = np.inf
        best_score = -float('inf')
        
        # Grid search over regularization and f0 scaling
        logger.info("Performing hyperparameter search...")
        for reg in self.reg_params:
            for f0 in self.f0_scaling:
                # Scale labels by f0 (paper's Appendix A.3)
                if f0 == np.inf:
                    scaled_labels = expanded_labels
                else:
                    scaled_labels = expanded_labels * f0
                
                # Solve: (K + reg*I) α = y
                reg_kernel = self.kernel_matrix + reg * np.eye(len(self.kernel_matrix))
                
                try:
                    alpha = np.linalg.solve(reg_kernel, scaled_labels)
                except np.linalg.LinAlgError:
                    continue
                
                # Validate
                if val_kernel is not None and val_labels is not None:
                    val_output = val_kernel @ alpha
                    
                    # Add pre-trained model output (paper's Appendix A.3)
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
        
        logger.info(f"Best hyperparameters: reg={best_reg}, f0={best_f0}, val_acc={best_score:.3f}")
        
        # Fit with best hyperparameters
        if best_f0 == np.inf:
            scaled_labels = expanded_labels
        else:
            scaled_labels = expanded_labels * best_f0
        
        reg_kernel = self.kernel_matrix + best_reg * np.eye(len(self.kernel_matrix))
        alpha = np.linalg.solve(reg_kernel, scaled_labels)
        
        # Predict on test set
        test_output = test_kernel @ alpha
        
        if best_f0 != np.inf:
            test_pretrain_flat = test_pretrain_outputs.flatten()
            test_output = test_output + test_pretrain_flat
        
        test_probs = test_output.reshape(-1, num_classes)
        test_pred = np.argmax(test_probs, axis=1)
        
        return test_pred, best_reg, best_f0

def get_pretrained_outputs(model, dataset, label_token_ids, device='cuda', batch_size=8):
    """Get pre-trained model outputs (before fine-tuning)."""
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_outputs = []
    model.eval()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Getting pre-trained outputs"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            mask_pos = batch['mask_pos']
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            batch_outputs = []
            for i in range(input_ids.shape[0]):
                pos = mask_pos[i].item()
                # Extract logits for label tokens
                label_logits = logits[i, pos, label_token_ids].cpu().numpy()
                batch_outputs.append(label_logits)
            
            all_outputs.extend(batch_outputs)
    
    return np.array(all_outputs)

def load_sst2_data(num_shots: int = 16, seed: int = 42):
    """Load SST-2 data in few-shot setting following paper's methodology."""
    random.seed(seed)
    np.random.seed(seed)
    
    dataset = load_dataset("sst2")
    train_data = dataset['train']
    
    # Sample k examples per class
    pos_examples = [(ex['sentence'], ex['label']) for ex in train_data if ex['label'] == 1]
    neg_examples = [(ex['sentence'], ex['label']) for ex in train_data if ex['label'] == 0]
    
    sampled_pos = random.sample(pos_examples, num_shots)
    sampled_neg = random.sample(neg_examples, num_shots)
    
    train_texts = [ex[0] for ex in sampled_pos + sampled_neg]
    train_labels = [ex[1] for ex in sampled_pos + sampled_neg]
    
    # Validation set (same size as training)
    remaining_pos = [ex for ex in pos_examples if ex not in sampled_pos]
    remaining_neg = [ex for ex in neg_examples if ex not in sampled_neg]
    
    val_pos = random.sample(remaining_pos, num_shots)
    val_neg = random.sample(remaining_neg, num_shots)
    
    val_texts = [ex[0] for ex in val_pos + val_neg]
    val_labels = [ex[1] for ex in val_pos + val_neg]
    
    # Test set (limit to 872 examples as in paper's Table 3)
    test_data = dataset['validation']
    test_texts = [ex['sentence'] for ex in test_data][:872]
    test_labels = [ex['label'] for ex in test_data][:872]
    
    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels

def main():
    # Configuration
    MODEL_NAME = "EleutherAI/pythia-70m"
    NUM_SHOTS = 16  # Paper uses k ∈ {16, 64, 512}
    SEED = 42
    BATCH_SIZE = 1  # Process one at a time for gradient computation
    USE_SIGN_KERNEL = True  # SignGD kernel for Adam (Theorem 4.3)
    
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    
    # Load model and tokenizer
    logger.info(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    
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
    
    # Get label token IDs
    label_token_ids = train_dataset.label_token_ids
    logger.info(f"Label tokens: {train_dataset.label_words} -> IDs: {label_token_ids}")
    
    # Initialize kernel computer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    logger.info(f"Using {'SignGD' if USE_SIGN_KERNEL else 'SGD'} kernel")
    
    kernel_computer = KernelComputer(
        model, tokenizer, label_token_ids, 
        device=device, use_sign_kernel=USE_SIGN_KERNEL
    )
    
    # Get pre-trained model outputs (needed for paper's method)
    logger.info("Getting pre-trained model outputs...")
    train_pretrain_outputs = get_pretrained_outputs(model, train_dataset, label_token_ids, device)
    val_pretrain_outputs = get_pretrained_outputs(model, val_dataset, label_token_ids, device)
    test_pretrain_outputs = get_pretrained_outputs(model, test_dataset, label_token_ids, device)
    
    # Compute kernel matrices
    logger.info("Computing training kernel matrix...")
    train_kernel, train_gradients = kernel_computer.compute_kernel_matrix(
        train_dataset, batch_size=BATCH_SIZE
    )
    
    logger.info("Computing validation cross-kernel...")
    _, val_gradients = kernel_computer.compute_kernel_matrix(
        val_dataset, batch_size=BATCH_SIZE
    )
    val_train_kernel = np.dot(
        val_gradients.reshape(len(val_texts) * 2, -1),
        train_gradients.reshape(len(train_texts) * 2, -1).T
    )
    
    logger.info("Computing test cross-kernel...")
    _, test_gradients = kernel_computer.compute_kernel_matrix(
        test_dataset, batch_size=BATCH_SIZE
    )
    test_train_kernel = np.dot(
        test_gradients.reshape(len(test_texts) * 2, -1),
        train_gradients.reshape(len(train_texts) * 2, -1).T
    )
    
    # Kernel regression with hyperparameter search
    logger.info("Performing kernel regression...")
    regressor = KernelRegressor(train_kernel)
    
    test_pred, best_reg, best_f0 = regressor.fit_and_predict(
        train_labels, train_pretrain_outputs,
        test_train_kernel, test_pretrain_outputs,
        val_train_kernel, val_labels, val_pretrain_outputs
    )
    
    # Evaluate
    test_accuracy = accuracy_score(test_labels, test_pred)
    
    logger.info("=" * 60)
    logger.info(f"Results (SST-2, {NUM_SHOTS}-shot):")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Kernel type: {'SignGD (for Adam)' if USE_SIGN_KERNEL else 'SGD'}")
    logger.info(f"Best regularization: {best_reg}")
    logger.info(f"Best f0 scaling: {best_f0}")
    logger.info(f"Test Accuracy: {test_accuracy:.3f}")
    logger.info("=" * 60)
    
    # Compare to paper's results (Table 2a: RoBERTa-base gets ~88.3% for 16-shot)
    logger.info("\nNote: Paper reports ~88.3% for RoBERTa-base 16-shot with K^(SignGD)")
    logger.info("Decoder-only models may perform differently than masked LMs")
    
    return {
        'test_accuracy': test_accuracy,
        'best_regularization': best_reg,
        'best_f0_scaling': best_f0,
    }

if __name__ == "__main__":
    results = main()