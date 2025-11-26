"""
Train high-resolution GP classifier using SoftDTW kernel.

This script trains a GP model on high-resolution time series (L=1024+)
using a custom SoftDTW kernel that avoids the numerical overflow issues
of tslearn's GAK implementation.

The kernel uses StableSoftDTWKernelV3 with:
- normalize=True for batch-independent, pairwise normalization
- Tiled computation for memory efficiency
- Optional bandwidth pruning for speed
- Pairwise kernel values (k(x,y) depends only on x and y)

The kernel uses the local SoftDTW implementation from models.kernels.backend.soft_dtw_cuda,
which provides both CPU and CUDA support with proper numerical stability.
"""

import torch
import gpytorch
import numpy as np
import argparse
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from models.gp_data import load_all_from_index
from models.kernels.stable_softdtw_kernel_v3 import StableSoftDTWKernelV3


def resample_1d(x, target_len):
    """Resample 1D array to fixed length using linear interpolation."""
    if len(x) == target_len:
        return x.copy()
    old = np.linspace(0, 1, len(x))
    new = np.linspace(0, 1, target_len)
    return np.interp(new, old, x)


def preprocess_lightcurve(curve_dict, target_len):
    """
    Preprocess a single light curve: normalize and resample.
    
    Parameters
    ----------
    curve_dict : dict
        Dictionary with 'flux' array
    target_len : int
        Target sequence length after resampling
    
    Returns
    -------
    torch.Tensor
        Preprocessed time series of shape (target_len, 1)
    
    Raises
    ------
    ValueError
        If median is invalid, or too few valid points after cleaning
    """
    flux = curve_dict['flux']
    
    valid_mask = ~np.isnan(flux)
    flux_clean = flux[valid_mask]
    
    if len(flux_clean) < 10:
        raise ValueError("Too few valid points after removing NaNs")
    
    flux_median = np.median(flux_clean)
    if flux_median == 0 or np.isnan(flux_median):
        raise ValueError("Invalid median flux")
    
    flux_normalized = flux_clean / flux_median
    
    if np.any(np.isnan(flux_normalized)):
        raise ValueError("NaN values present after normalization")
    
    flux_resampled = resample_1d(flux_normalized, target_len)
    
    if np.any(np.isnan(flux_resampled)):
        raise ValueError("NaN values in resampled flux")
    
    flux_tensor = torch.from_numpy(flux_resampled).float()
    flux_tensor = flux_tensor.unsqueeze(-1)
    
    return flux_tensor


class HighResTimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for high-resolution time series.
    
    Loads and preprocesses light curves on-the-fly.
    
    Note: This implementation loads all data into memory for simplicity.
    For very large datasets (N > 1000), consider implementing lazy loading
    to avoid memory issues.
    """
    
    def __init__(self, index_path, target_len, max_samples=None):
        """
        Initialize dataset.
        
        Parameters
        ----------
        index_path : str
            Path to index.csv file
        target_len : int
            Target sequence length after resampling
        max_samples : int, optional
            Maximum number of samples to load (for testing, default: None)
            Useful for quick testing with small subsets
        """
        self.index_path = index_path
        self.target_len = target_len
        
        self.data = []
        self.labels = []
        
        print(f"Loading data from {index_path}...")
        n_skipped = 0
        for tic_id, label, curve_dict, metadata in load_all_from_index(index_path, include_metadata=False):
            try:
                processed = preprocess_lightcurve(curve_dict, target_len)
                self.data.append(processed)
                self.labels.append(label)
                
                if max_samples is not None and len(self.data) >= max_samples:
                    break
            except (ValueError, KeyError) as e:
                n_skipped += 1
                continue
        
        if n_skipped > 0:
            print(f"Skipped {n_skipped} invalid light curves")
        print(f"Loaded {len(self.data)} valid light curves")
        print(f"  CP: {sum(self.labels)}, FP/FA: {len(self.labels) - sum(self.labels)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], torch.tensor(self.labels[idx], dtype=torch.long)


class HighResVariationalGPModel(gpytorch.models.ApproximateGP):
    """
    Variational GP model for high-resolution time series classification.
    
    Uses SoftDTW kernel to handle variable-length time series with
    high temporal resolution (L=1024+).
    """
    
    def __init__(self, inducing_points):
        """
        Initialize the variational GP model.
        
        Parameters
        ----------
        inducing_points : torch.Tensor
            Initial inducing point locations (shape: [num_inducing, L, D])
            where L is sequence length and D is features (1 for flux)
        """
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )
        variational_strategy._jitter_val = 1e-4
        super(HighResVariationalGPModel, self).__init__(variational_strategy)
        
        self.mean_module = gpytorch.means.ConstantMean()
        base_kernel = StableSoftDTWKernelV3(
            gamma=1.0,
            use_cuda=None,
            enforce_psd=False,
            tile_size=12,
            bandwidth=128
        )
        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)
    
    def forward(self, x):
        """
        Forward pass through the GP.
        
        Parameters
        ----------
        x : torch.Tensor
            Input time series (shape: [batch_size, L, D])
            where L is sequence length and D is features
        
        Returns
        -------
        gpytorch.distributions.MultivariateNormal
            GP distribution
        """
        batch_size = x.shape[0]
        mean_input = x.reshape(batch_size, -1)
        mean_x = self.mean_module(mean_input)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def collate_fn(batch):
    """Custom collate function for DataLoader."""
    time_series, labels = zip(*batch)
    time_series_batch = torch.stack(time_series, dim=0)
    labels_batch = torch.stack(labels, dim=0)
    return time_series_batch, labels_batch


def plot_validation_curves_with_neighbors(model, val_loader, log_dir, iteration, device='cuda', num_samples=5):
    """
    Plot validation curves with their closest inducing points.
    
    For each plotted validation curve, shows the exact inducing shapes SoftDTW
    thinks are closest. This helps verify kernel behavior is intuitive.
    
    Parameters
    ----------
    model : HighResVariationalGPModel
        Trained GP model
    val_loader : DataLoader
        Validation data loader
    log_dir : Path
        Directory to save plots
    iteration : int
        Current training iteration
    device : str
        Device to use
    num_samples : int
        Number of validation samples to plot (default: 5)
    """
    model.eval()
    
    val_batch_x, val_batch_y = next(iter(val_loader))
    val_batch_x = val_batch_x.to(device)
    val_batch_y = val_batch_y.to(device)
    
    val_batch_x = val_batch_x[:num_samples]
    val_batch_y = val_batch_y[:num_samples]
    
    inducing_points = model.variational_strategy.inducing_points
    base_k = model.covar_module.base_kernel
    
    with torch.no_grad():
        K_bz = base_k.forward(val_batch_x, inducing_points, diag=False)
    
    closest_indices = torch.argmax(K_bz, dim=1)
    
    gamma = float(base_k.gamma)
    K_clamped = torch.clamp(K_bz, min=1e-8, max=1.0)
    distances = -gamma * torch.log(K_clamped)
    
    # Create plots
    fig, axes = plt.subplots(num_samples, 2, figsize=(14, 3 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        val_curve = val_batch_x[i, :, 0].detach().cpu().numpy()
        closest_idx = closest_indices[i].item()
        closest_inducing = inducing_points[closest_idx, :, 0].detach().cpu().numpy()
        distance = distances[i, closest_idx].item()
        label = val_batch_y[i].item()
        
        axes[i, 0].plot(val_curve, label=f'Val (label={label})', linewidth=2)
        axes[i, 0].set_title(f'Validation Curve {i+1} (label={label})')
        axes[i, 0].set_xlabel('Time Step')
        axes[i, 0].set_ylabel('Normalized Flux')
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 0].legend()
        
        axes[i, 1].plot(val_curve, label=f'Val (label={label})', linewidth=2, alpha=0.7)
        axes[i, 1].plot(closest_inducing, label=f'Closest Inducing (d={distance:.3f})', 
                       linewidth=2, linestyle='--', alpha=0.7)
        axes[i, 1].set_title(f'Validation + Closest Inducing Point\n(distance={distance:.3f})')
        axes[i, 1].set_xlabel('Time Step')
        axes[i, 1].set_ylabel('Normalized Flux')
        axes[i, 1].grid(True, alpha=0.3)
        axes[i, 1].legend()
    
    plt.tight_layout()
    plot_path = log_dir / f"validation_neighbors_iter_{iteration:04d}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    Saved validation neighbor plots to: {plot_path}")


def train_single_fold(train_loader, val_loader, num_iterations=100, num_inducing=100,
                     device='cuda', verbose=False, log_dir=None):
    """
    Train a GP model on a single fold of data with validation monitoring.
    
    Parameters
    ----------
    train_loader : DataLoader
        DataLoader for training data
    val_loader : DataLoader
        DataLoader for validation data
    num_iterations : int, optional
        Number of training iterations (default: 100)
    num_inducing : int, optional
        Number of inducing points (default: 100)
    device : str, optional
        Device to use ('cuda' or 'cpu', default: 'cuda')
    verbose : bool, optional
        Whether to print training progress (default: False)
    log_dir : Path, optional
        Directory to save logs, plots, and tensorboard files (default: None)
    
    Returns
    -------
    tuple
        (model, likelihood, best_val_accuracy) trained on the fold
    """
    sample_batch, _ = next(iter(train_loader))
    sample_batch = sample_batch.to(device)
    
    inducing_samples = []
    for batch_x, _ in train_loader:
        inducing_samples.append(batch_x.to(device))
        total_collected = sum(s.shape[0] for s in inducing_samples)
        if total_collected >= num_inducing:
            break
    
    inducing_candidates = torch.cat(inducing_samples, dim=0)
    num_inducing_actual = min(num_inducing, len(inducing_candidates))
    
    indices = torch.randperm(len(inducing_candidates), device=device)[:num_inducing_actual]
    inducing_points = inducing_candidates[indices].clone()
    
    model = HighResVariationalGPModel(inducing_points)
    model = model.to(device)
    likelihood = gpytorch.likelihoods.BernoulliLikelihood()
    likelihood = likelihood.to(device)
    
    model.train()
    likelihood.train()
    
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()}
    ], lr=0.01)
    
    num_data = len(train_loader.dataset)
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
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).float()
            
            optimizer.zero_grad()
            output = model(batch_x)
            loss = -mll(output, batch_y)
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
            
            if log_dir is not None:
                try:
                    plot_validation_curves_with_neighbors(
                        model, val_loader, log_dir, i + 1, device=device, num_samples=5
                    )
                except Exception as e:
                    if verbose:
                        print(f"    Warning: Failed to plot validation curves: {e}")
            
            if verbose:
                print(f"  Iteration {i+1}/{num_iterations} - Train Loss: {avg_loss:.4f}, Val Acc: {100*val_accuracy:.2f}%")
        elif verbose and (i + 1) % 10 == 0:
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
    model : HighResVariationalGPModel
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
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            output = model(batch_x)
            predictions = likelihood(output)
            probs = predictions.mean
            pred_labels = (probs > 0.5).long()
            
            all_preds.append(pred_labels.cpu())
            all_labels.append(batch_y.cpu())
    
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    accuracy = (all_preds == all_labels).float().mean().item()
    
    return accuracy


def train_highres_gp(index_path="dataset/index.csv", target_len=1024, 
                     batch_size=8, num_inducing=50, num_iterations=50,
                     n_splits=5, random_state=42, device=None, max_samples=150):
    """
    Train high-resolution GP classifier using k-fold cross-validation.
    
    Parameters
    ----------
    index_path : str, optional
        Path to index.csv file (default: "dataset/index.csv")
    target_len : int, optional
        Target sequence length after resampling (default: 1024)
    batch_size : int, optional
        Batch size for DataLoader (default: 32)
    num_inducing : int, optional
        Number of inducing points (default: 100)
    num_iterations : int, optional
        Number of training iterations per fold (default: 100)
    n_splits : int, optional
        Number of folds for cross-validation (default: 5)
    random_state : int, optional
        Random seed for reproducibility (default: 42)
    device : str, optional
        Device to use ('cuda' or 'cpu', default: auto-detect)
    max_samples : int, optional
        Maximum number of samples to load (for testing, default: None)
    
    Returns
    -------
    tuple
        (mean_accuracy, std_accuracy, all_accuracies)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    print(f"Target sequence length: {target_len}")
    
    print("\nLoading dataset...")
    dataset = HighResTimeSeriesDataset(index_path, target_len, max_samples=max_samples)
    
    if len(dataset) == 0:
        raise ValueError("No valid data loaded")
    
    labels = [dataset.labels[i] for i in range(len(dataset))]
    labels_array = np.array(labels)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    accuracies = []
    models = []
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = Path("saved_models/highres_gp") / timestamp
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nStarting {n_splits}-fold cross-validation...")
    print(f"  Training iterations per fold: {num_iterations}")
    print(f"  Inducing points: {num_inducing}")
    print(f"  Batch size: {batch_size}")
    print(f"  Models will be saved to: {model_dir}\n")
    
    for fold_idx, (train_indices, test_indices) in enumerate(skf.split(np.arange(len(dataset)), labels_array), 1):
        print(f"Fold {fold_idx}/{n_splits}")
        
        n_train = len(train_indices)
        n_val = max(1, int(0.2 * n_train))
        n_train_actual = n_train - n_val
        
        train_indices_shuffled = train_indices.copy()
        np.random.RandomState(random_state + fold_idx).shuffle(train_indices_shuffled)
        
        train_indices_final = train_indices_shuffled[:n_train_actual]
        val_indices = train_indices_shuffled[n_train_actual:]
        
        print(f"  Train: {len(train_indices_final)} samples, Val: {len(val_indices)} samples, Test: {len(test_indices)} samples")
        
        train_dataset = torch.utils.data.Subset(dataset, train_indices_final)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            collate_fn=collate_fn,
            pin_memory=(device == 'cuda')
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            pin_memory=(device == 'cuda')
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            pin_memory=(device == 'cuda')
        )
        
        fold_log_dir = model_dir / f"fold_{fold_idx}_logs"
        
        model, likelihood, best_val_acc = train_single_fold(
            train_loader,
            val_loader,
            num_iterations=num_iterations,
            num_inducing=num_inducing,
            device=device,
            verbose=(fold_idx == 1),
            log_dir=fold_log_dir
        )
        
        test_accuracy = evaluate_model(model, likelihood, test_loader, device=device)
        accuracies.append(test_accuracy)
        models.append((model, likelihood))
        
        model_path = model_dir / f"highres_gp_fold_{fold_idx}.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'likelihood_state_dict': likelihood.state_dict(),
            'test_accuracy': test_accuracy,
            'best_val_accuracy': best_val_acc,
            'fold': fold_idx,
            'target_len': target_len,
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
    best_model_path = model_dir / "highres_gp_best.pth"
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'likelihood_state_dict': best_likelihood.state_dict(),
        'test_accuracy': accuracies[best_fold_idx - 1],
        'fold': best_fold_idx,
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'target_len': target_len,
        'num_inducing': num_inducing
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
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Train high-resolution GP classifier with SoftDTW kernel"
    )
    parser.add_argument(
        '--index_path',
        type=str,
        default="dataset/index.csv",
        help='Path to index.csv file (default: dataset/index.csv)'
    )
    parser.add_argument(
        '--target_len',
        type=int,
        default=1024,
        help='Target sequence length after resampling (default: 1024)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Batch size for DataLoader (default: 8)'
    )
    parser.add_argument(
        '--num_inducing',
        type=int,
        default=50,
        help='Number of inducing points (default: 50)'
    )
    parser.add_argument(
        '--num_iterations',
        type=int,
        default=50,
        help='Number of training iterations per fold (default: 50)'
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
        '--max_samples',
        type=int,
        default=150,
        help='Maximum number of samples to load (for testing, default: 150)'
    )
    
    args = parser.parse_args()
    
    train_highres_gp(
        index_path=args.index_path,
        target_len=args.target_len,
        batch_size=args.batch_size,
        num_inducing=args.num_inducing,
        num_iterations=args.num_iterations,
        n_splits=args.n_splits,
        random_state=args.random_state,
        device=args.device,
        max_samples=args.max_samples
    )


if __name__ == "__main__":
    main()

