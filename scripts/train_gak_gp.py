"""
Train GP classifier using precomputed GAK (Global Alignment Kernel) matrix.

This script trains a variational GP classifier on a precomputed GAK similarity matrix.
The kernel matrix is accessed via integer indices, making this approach memory-efficient
for large datasets where the full kernel matrix has already been computed.
"""

import torch
import gpytorch
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import argparse


def load_gak_data(k_path=None, y_path=None, device='cuda'):
    """
    Load precomputed GAK matrix and labels from disk.
    
    Parameters
    ----------
    k_path : str or Path, optional
        Path to GAK matrix file (default: "dataset/X_gak_matrix.npy")
    y_path : str or Path, optional
        Path to labels file (default: "dataset/y_labels.npy")
    device : str, optional
        Device to load tensors on (default: 'cuda')
    
    Returns
    -------
    tuple
        (K_tensor, y_tensor) where:
        - K_tensor: PyTorch tensor of shape (N, N) with GAK similarity matrix
        - y_tensor: PyTorch tensor of shape (N,) with binary labels
    """
    if k_path is None:
        k_path = Path("dataset") / "X_gak_matrix.npy"
    if y_path is None:
        y_path = Path("dataset") / "y_labels.npy"
    
    k_path = Path(k_path)
    y_path = Path(y_path)
    
    if not k_path.exists():
        raise FileNotFoundError(f"GAK matrix not found: {k_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"Labels not found: {y_path}")
    
    K = np.load(k_path)
    y = np.load(y_path)
    
    if K.ndim != 2:
        raise ValueError(f"GAK matrix must be 2D, got shape {K.shape}")
    if K.shape[0] != K.shape[1]:
        raise ValueError(f"GAK matrix must be square, got shape {K.shape}")
    if len(y) != K.shape[0]:
        raise ValueError(f"Labels length {len(y)} doesn't match matrix size {K.shape[0]}")
    
    if np.any(np.isnan(K)):
        print(f"Warning: GAK matrix contains NaN values")
    if np.any(np.isinf(K)):
        print(f"Warning: GAK matrix contains Inf values")
    
    K_tensor = torch.from_numpy(K).float().to(device)
    y_tensor = torch.from_numpy(y).long().to(device)
    
    print(f"Loaded GAK matrix: shape {K_tensor.shape}")
    print(f"Loaded labels: shape {y_tensor.shape}")
    print(f"  CP: {torch.sum(y_tensor == 1).item()}, FP/FA: {torch.sum(y_tensor == 0).item()}")
    
    return K_tensor, y_tensor


class PrecomputedGAKKernel(gpytorch.kernels.Kernel):
    """
    GPyTorch kernel that uses a precomputed GAK similarity matrix.
    
    This kernel expects integer indices as inputs and returns the corresponding
    values from the precomputed kernel matrix.
    """
    
    has_lengthscale = False
    
    def __init__(self, K_full, **kwargs):
        """
        Initialize the precomputed GAK kernel.
        
        Parameters
        ----------
        K_full : torch.Tensor
            Full precomputed kernel matrix, shape (N, N)
        """
        super(PrecomputedGAKKernel, self).__init__(**kwargs)
        self.register_buffer("K_full", K_full)
    
    def forward(self, x1, x2, diag=False, **params):
        """
        Compute kernel matrix for given indices.
        
        Parameters
        ----------
        x1 : torch.Tensor
            First set of indices (as floats from GPyTorch), shape (B1, 1) or (B1,)
        x2 : torch.Tensor
            Second set of indices (as floats from GPyTorch), shape (B2, 1) or (B2,)
        diag : bool, optional
            If True, return only diagonal elements (default: False)
        
        Returns
        -------
        torch.Tensor
            Kernel matrix of shape (B1, B2) if diag=False
            or diagonal vector of shape (B1,) if diag=True
        """
        if x1.dim() > 1:
            x1 = x1.squeeze(-1)
        if x2.dim() > 1:
            x2 = x2.squeeze(-1)
        
        x1_idx = x1.long()
        x2_idx = x2.long()
        
        N = self.K_full.shape[0]
        if torch.any(x1_idx < 0) or torch.any(x1_idx >= N):
            raise ValueError(f"x1 indices out of bounds [0, {N})")
        if torch.any(x2_idx < 0) or torch.any(x2_idx >= N):
            raise ValueError(f"x2 indices out of bounds [0, {N})")
        
        if diag:
            if x1_idx.shape != x2_idx.shape or not torch.equal(x1_idx, x2_idx):
                raise ValueError("For diag=True, x1 and x2 must be equal")
            return self.K_full[x1_idx, x1_idx]
        else:
            return self.K_full[x1_idx.unsqueeze(1), x2_idx.unsqueeze(0)]


class IndexDataset(Dataset):
    """
    Dataset that returns integer indices and labels.
    
    Used for training GP models on precomputed kernel matrices where
    inputs are indices into the kernel matrix rather than feature vectors.
    """
    
    def __init__(self, indices, labels):
        """
        Initialize the index dataset.
        
        Parameters
        ----------
        indices : array-like
            Integer indices into the kernel matrix
        labels : array-like
            Binary labels corresponding to each index
        """
        self.indices = np.array(indices, dtype=np.int64)
        self.labels = np.array(labels, dtype=np.int64)
        
        if len(self.indices) != len(self.labels):
            raise ValueError(f"Indices and labels must have same length: {len(self.indices)} vs {len(self.labels)}")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.indices[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )


class GAKVariationalGPModel(gpytorch.models.ApproximateGP):
    """
    Variational GP model for GAK kernel matrix classification.
    
    Uses integer indices as inputs to access a precomputed GAK kernel matrix.
    """
    
    def __init__(self, K_full, inducing_indices):
        """
        Initialize the variational GP model.
        
        Parameters
        ----------
        K_full : torch.Tensor
            Full precomputed kernel matrix, shape (N, N)
        inducing_indices : torch.Tensor
            Integer indices for inducing points, shape (M,)
        """
        # Validate inducing indices
        N = K_full.shape[0]
        if torch.any(inducing_indices < 0) or torch.any(inducing_indices >= N):
            raise ValueError(f"Inducing indices must be in range [0, {N})")
        
        M = inducing_indices.shape[0]
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(M)
        inducing_points = inducing_indices.unsqueeze(-1).float()
        
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=False
        )
        variational_strategy._jitter_val = 1e-4
        super(GAKVariationalGPModel, self).__init__(variational_strategy)
        
        self.register_buffer("K_full", K_full)
        self.register_buffer("inducing_indices", inducing_indices)
        self.mean_module = gpytorch.means.ConstantMean()
        base_kernel = PrecomputedGAKKernel(K_full)
        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)
    
    def forward(self, x):
        """
        Forward pass through the GP.
        
        Parameters
        ----------
        x : torch.Tensor
            Integer indices, shape (batch_size,) or (batch_size, 1)
        
        Returns
        -------
        gpytorch.distributions.MultivariateNormal
            GP distribution
        """
        if x.dim() > 1:
            x = x.squeeze(-1)
        x_float = x.float().unsqueeze(-1)
        
        mean_x = self.mean_module(x_float)
        covar_x = self.covar_module(x_float)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_single_fold(train_indices, val_indices, K_full, y_full, num_iterations=100,
                     num_inducing=50, device='cuda', verbose=False, log_dir=None,
                     batch_size=32):
    """
    Train a GP model on a single fold of data with validation monitoring.
    
    Parameters
    ----------
    train_indices : array-like
        Training set indices
    val_indices : array-like
        Validation set indices
    K_full : torch.Tensor
        Full precomputed kernel matrix, shape (N, N)
    y_full : torch.Tensor
        Full labels array, shape (N,)
    num_iterations : int, optional
        Number of training iterations (default: 100)
    num_inducing : int, optional
        Number of inducing points (default: 50)
    device : str, optional
        Device to use ('cuda' or 'cpu', default: 'cuda')
    verbose : bool, optional
        Whether to print training progress (default: False)
    log_dir : Path, optional
        Directory to save logs, plots, and tensorboard files (default: None)
    batch_size : int, optional
        Batch size for DataLoader (default: 32)
    
    Returns
    -------
    tuple
        (model, likelihood, best_val_accuracy) trained on the fold
    """
    train_indices_np = np.array(train_indices, dtype=np.int64)
    val_indices_np = np.array(val_indices, dtype=np.int64)
    
    train_idx_t = torch.from_numpy(train_indices_np).to(y_full.device)
    val_idx_t = torch.from_numpy(val_indices_np).to(y_full.device)
    
    train_labels_np = y_full[train_idx_t].cpu().numpy()
    val_labels_np = y_full[val_idx_t].cpu().numpy()
    
    num_inducing_actual = min(num_inducing, len(train_indices_np))
    inducing_candidates = torch.from_numpy(train_indices_np).to(device)
    indices = torch.randperm(len(inducing_candidates), device=device)[:num_inducing_actual]
    inducing_indices = inducing_candidates[indices]
    
    model = GAKVariationalGPModel(K_full, inducing_indices)
    model = model.to(device)
    likelihood = gpytorch.likelihoods.BernoulliLikelihood()
    likelihood = likelihood.to(device)
    
    model.train()
    likelihood.train()
    
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()}
    ], lr=0.01)
    
    train_dataset = IndexDataset(train_indices_np, train_labels_np)
    val_dataset = IndexDataset(val_indices_np, val_labels_np)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(device == 'cuda')
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=(device == 'cuda')
    )
    
    num_data = len(train_indices_np)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=num_data)
    
    best_val_accuracy = 0.0
    train_losses = []
    val_accuracies = []
    tensorboard_writer = None
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        tensorboard_writer = SummaryWriter(log_dir=str(log_dir))
    
    for i in range(num_iterations):
        model.train()
        likelihood.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_indices, batch_labels in train_loader:
            batch_indices = batch_indices.to(device)
            batch_labels = batch_labels.to(device).float()
            
            optimizer.zero_grad()
            output = model(batch_indices)
            loss = -mll(output, batch_labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        train_losses.append(avg_loss)
        
        if tensorboard_writer is not None:
            tensorboard_writer.add_scalar('Train/Loss', avg_loss, i + 1)
        
        if (i + 1) % 10 == 0 or (i + 1) == num_iterations:
            val_accuracy = evaluate_model(model, likelihood, val_loader, device=device)
            val_accuracies.append(val_accuracy)
            
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
            
            if tensorboard_writer is not None:
                tensorboard_writer.add_scalar('Val/Accuracy', val_accuracy, i + 1)
            
            if verbose:
                print(f"  Iteration {i+1}/{num_iterations} - Train Loss: {avg_loss:.4f}, Val Acc: {100*val_accuracy:.2f}%")
        elif verbose:
            print(f"  Iteration {i+1}/{num_iterations} - Loss: {avg_loss:.4f}")
    
    if log_dir is not None:
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss', linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(log_dir / 'train_loss.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        if len(val_accuracies) > 0:
            val_iterations = [10 * (j + 1) for j in range(len(val_accuracies))]
            if val_iterations[-1] != num_iterations:
                val_iterations[-1] = num_iterations
            
            plt.figure(figsize=(10, 6))
            plt.plot(val_iterations, [100 * acc for acc in val_accuracies], 
                    marker='o', label='Val Accuracy', linewidth=2, markersize=6)
            plt.xlabel('Iteration')
            plt.ylabel('Accuracy (%)')
            plt.title('Validation Accuracy')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(log_dir / 'val_accuracy.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        if tensorboard_writer is not None:
            tensorboard_writer.close()
    
    return model, likelihood, best_val_accuracy


def evaluate_model(model, likelihood, test_loader, device='cuda'):
    """
    Evaluate a trained GP model on test data.
    
    Parameters
    ----------
    model : GAKVariationalGPModel
        Trained GP model
    likelihood : gpytorch.likelihoods.BernoulliLikelihood
        Trained likelihood
    test_loader : DataLoader
        DataLoader for test data
    device : str, optional
        Device to use (default: 'cuda')
    
    Returns
    -------
    float
        Accuracy on test set
    """
    model.eval()
    likelihood.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_indices, batch_labels in test_loader:
            batch_indices = batch_indices.to(device)
            batch_labels = batch_labels.to(device)
            
            output = model(batch_indices)
            predictions = likelihood(output)
            probs = predictions.mean
            pred_labels = (probs > 0.5).long()
            
            all_preds.append(pred_labels.cpu())
            all_labels.append(batch_labels.cpu())
    
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    accuracy = (all_preds == all_labels).float().mean().item()
    
    return accuracy


def train_gak_gp(num_iterations=100, num_inducing=50, n_splits=5, random_state=42,
                device=None, batch_size=32, k_path=None, y_path=None):
    """
    Train GAK GP classifier using k-fold cross-validation.
    
    Parameters
    ----------
    num_iterations : int, optional
        Number of training iterations per fold (default: 100)
    num_inducing : int, optional
        Number of inducing points (default: 50)
    n_splits : int, optional
        Number of folds for cross-validation (default: 5)
    random_state : int, optional
        Random seed for reproducibility (default: 42)
    device : str, optional
        Device to use ('cuda' or 'cpu', default: auto-detect)
    batch_size : int, optional
        Batch size for DataLoader (default: 32)
    k_path : str or Path, optional
        Path to GAK matrix file (default: "dataset/X_gak_matrix.npy")
    y_path : str or Path, optional
        Path to labels file (default: "dataset/y_labels.npy")
    
    Returns
    -------
    tuple
        (mean_accuracy, std_accuracy, all_accuracies)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    
    print("\nLoading GAK matrix and labels...")
    K_full, y_full = load_gak_data(k_path=k_path, y_path=y_path, device=device)
    
    y_np = y_full.cpu().numpy()
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    accuracies = []
    models = []
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = Path("saved_models/gak_gp") / timestamp
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nStarting {n_splits}-fold cross-validation...")
    print(f"  Training iterations per fold: {num_iterations}")
    print(f"  Inducing points: {num_inducing}")
    print(f"  Batch size: {batch_size}")
    print(f"  Models will be saved to: {model_dir}\n")
    
    for fold_idx, (train_indices, test_indices) in enumerate(skf.split(np.arange(len(y_np)), y_np), 1):
        print(f"Fold {fold_idx}/{n_splits}")
        
        n_train = len(train_indices)
        n_val = max(1, int(0.2 * n_train))
        n_train_actual = n_train - n_val
        
        train_indices_shuffled = train_indices.copy()
        np.random.RandomState(random_state + fold_idx).shuffle(train_indices_shuffled)
        
        train_indices_final = train_indices_shuffled[:n_train_actual]
        val_indices = train_indices_shuffled[n_train_actual:]
        
        print(f"  Train: {len(train_indices_final)} samples, Val: {len(val_indices)} samples, Test: {len(test_indices)} samples")
        
        fold_log_dir = model_dir / f"fold_{fold_idx}_logs"
        
        model, likelihood, best_val_acc = train_single_fold(
            train_indices_final,
            val_indices,
            K_full,
            y_full,
            num_iterations=num_iterations,
            num_inducing=num_inducing,
            device=device,
            verbose=(fold_idx == 1),
            log_dir=fold_log_dir,
            batch_size=batch_size
        )
        
        test_dataset = IndexDataset(test_indices, y_np[test_indices])
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=(device == 'cuda')
        )
        test_accuracy = evaluate_model(model, likelihood, test_loader, device=device)
        accuracies.append(test_accuracy)
        models.append((model, likelihood))
        
        model_path = model_dir / f"gak_gp_fold_{fold_idx}.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'likelihood_state_dict': likelihood.state_dict(),
            'test_accuracy': test_accuracy,
            'best_val_accuracy': best_val_acc,
            'fold': fold_idx,
            'num_inducing': num_inducing
        }, model_path)
        print(f"  Best Val Accuracy: {100 * best_val_acc:.2f}%")
        print(f"  Test Accuracy: {100 * test_accuracy:.2f}%")
        print(f"  Model saved to: {model_path}")
        print()
    
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    
    best_fold_idx = np.argmax(accuracies) + 1
    best_model, best_likelihood = models[best_fold_idx - 1]
    best_model_path = model_dir / "gak_gp_best.pth"
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'likelihood_state_dict': best_likelihood.state_dict(),
        'test_accuracy': accuracies[best_fold_idx - 1],
        'fold': best_fold_idx,
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy
    }, best_model_path)
    
    print("=" * 60)
    print("Cross-Validation Results:")
    print(f"  Mean Accuracy: {100 * mean_accuracy:.2f}%")
    print(f"  Std Accuracy:  {100 * std_accuracy:.2f}%")
    print(f"  Accuracy: {100 * mean_accuracy:.2f}% ± {100 * std_accuracy:.2f}%")
    print(f"\nPer-fold accuracies:")
    for i, acc in enumerate(accuracies, 1):
        marker = " (best)" if i == best_fold_idx else ""
        print(f"  Fold {i}: {100 * acc:.2f}%{marker}")
    print(f"\nBest model (Fold {best_fold_idx}) saved to: {best_model_path}")
    print("=" * 60)
    
    return mean_accuracy, std_accuracy, accuracies


def main():
    """Main entry point with argparse support."""
    parser = argparse.ArgumentParser(
        description='Train GP classifier using precomputed GAK matrix',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--k_path',
        type=str,
        default=None,
        help='Path to GAK matrix file (default: dataset/X_gak_matrix.npy)'
    )
    parser.add_argument(
        '--y_path',
        type=str,
        default=None,
        help='Path to labels file (default: dataset/y_labels.npy)'
    )
    parser.add_argument(
        '--num_iterations',
        type=int,
        default=100,
        help='Number of training iterations per fold (default: 100)'
    )
    parser.add_argument(
        '--num_inducing',
        type=int,
        default=50,
        help='Number of inducing points (default: 50)'
    )
    parser.add_argument(
        '--n_splits',
        type=int,
        default=5,
        help='Number of folds for cross-validation (default: 5)'
    )
    parser.add_argument(
        '--random_state',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'cpu'],
        help='Device to use (default: auto-detect)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for DataLoader (default: 32)'
    )
    
    args = parser.parse_args()
    
    train_gak_gp(
        num_iterations=args.num_iterations,
        num_inducing=args.num_inducing,
        n_splits=args.n_splits,
        random_state=args.random_state,
        device=args.device,
        batch_size=args.batch_size,
        k_path=args.k_path,
        y_path=args.y_path
    )


if __name__ == "__main__":
    main()

