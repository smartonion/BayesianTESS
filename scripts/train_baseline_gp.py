"""
Train baseline GP classifier on extracted features.
"""

import torch
import gpytorch
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from datetime import datetime


def load_and_clean_data():
    """
    Load feature and label arrays, remove rows with NaN features,
    return clean PyTorch tensors.
    
    Returns
    -------
    tuple
        (X, y) where:
        - X: PyTorch tensor of shape (N, 6) with features
        - y: PyTorch tensor of shape (N,) with binary labels
    """
    dataset_path = Path("dataset")
    x_path = dataset_path / "X_features_baseline.npy"
    y_path = dataset_path / "y_labels_baseline.npy"
    
    if not x_path.exists():
        raise FileNotFoundError(f"Feature matrix not found: {x_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"Labels not found: {y_path}")
    
    X = np.load(x_path)
    y = np.load(y_path)
    
    valid_mask = ~np.any(np.isnan(X), axis=1)
    X_clean = X[valid_mask]
    y_clean = y[valid_mask]
    
    n_removed = len(X) - len(X_clean)
    if n_removed > 0:
        print(f"Removed {n_removed} stars with NaN features ({100*n_removed/len(X):.1f}%)")
    
    print(f"Using {len(X_clean)} stars for training")
    print(f"  CP: {np.sum(y_clean == 1)}, FP/FA: {np.sum(y_clean == 0)}")
    
    X_tensor = torch.from_numpy(X_clean).float()
    y_tensor = torch.from_numpy(y_clean).long()
    
    # Normalize features (zero mean, unit variance)
    X_mean = X_tensor.mean(dim=0, keepdim=True)
    X_std = X_tensor.std(dim=0, keepdim=True) + 1e-6  # Add small epsilon to avoid division by zero
    X_normalized = (X_tensor - X_mean) / X_std
    
    return X_normalized, y_tensor


class VariationalGPModel(gpytorch.models.ApproximateGP):
    """
    Variational GP model for binary classification.
    
    Uses variational inference to approximate the posterior for GP classification.
    """
    
    def __init__(self, inducing_points):
        """
        Initialize the variational GP model.
        
        Parameters
        ----------
        inducing_points : torch.Tensor
            Initial inducing point locations (shape: [num_inducing, num_features])
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
        variational_strategy._jitter_val = 1e-4  # Add jitter for numerical stability
        super(VariationalGPModel, self).__init__(variational_strategy)
        
        # Mean function (constant)
        self.mean_module = gpytorch.means.ConstantMean()
        
        # Kernel (RBF wrapped in ScaleKernel)
        base_kernel = gpytorch.kernels.RBFKernel()
        base_kernel.lengthscale = 1.0  # Initialize with reasonable lengthscale
        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)
    
    def forward(self, x):
        """
        Forward pass through the GP.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features (shape: [batch_size, num_features])
        
        Returns
        -------
        gpytorch.distributions.MultivariateNormal
            GP distribution
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_single_fold(train_x, train_y, num_iterations=100, num_inducing=50, verbose=False):
    """
    Train a GP model on a single fold of data.
    
    Parameters
    ----------
    train_x : torch.Tensor
        Training features
    train_y : torch.Tensor
        Training labels
    num_iterations : int, optional
        Number of training iterations (default: 100)
    num_inducing : int, optional
        Number of inducing points (default: 50)
    verbose : bool, optional
        Whether to print training progress (default: False)
    
    Returns
    -------
    tuple
        (model, likelihood) trained on the fold
    """
    num_inducing_actual = min(num_inducing, len(train_x))
    inducing_points = train_x[:num_inducing_actual].clone()
    model = VariationalGPModel(inducing_points)
    likelihood = gpytorch.likelihoods.BernoulliLikelihood()
    
    model.train()
    likelihood.train()
    
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()}
    ], lr=0.01)
    
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))
    
    for i in range(num_iterations):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y.float())
        loss.backward()
        optimizer.step()
        
        if verbose and (i + 1) % 10 == 0:
            print(f"  Iteration {i+1}/{num_iterations} - Loss: {loss.item():.4f}")
    
    return model, likelihood


def evaluate_model(model, likelihood, test_x, test_y):
    """
    Evaluate a trained GP model on test data.
    
    Parameters
    ----------
    model : VariationalGPModel
        Trained GP model
    likelihood : gpytorch.likelihoods.BernoulliLikelihood
        Trained likelihood
    test_x : torch.Tensor
        Test features
    test_y : torch.Tensor
        Test labels
    
    Returns
    -------
    float
        Accuracy on test set
    """
    model.eval()
    likelihood.eval()
    
    with torch.no_grad():
        output = model(test_x)
        predictions = likelihood(output)
        probs = predictions.mean
        pred_labels = (probs > 0.5).long()
        accuracy = (pred_labels == test_y).float().mean().item()
    
    return accuracy


def train_baseline_gp(num_iterations=100, num_inducing=50, n_splits=5, random_state=42):
    """
    Train baseline GP classifier using k-fold cross-validation.
    
    Parameters
    ----------
    num_iterations : int, optional
        Number of training iterations per fold (default: 100)
    num_inducing : int, optional
        Number of inducing points for variational inference (default: 50)
    n_splits : int, optional
        Number of folds for cross-validation (default: 5)
    random_state : int, optional
        Random seed for reproducibility (default: 42)
    
    Returns
    -------
    tuple
        (mean_accuracy, std_accuracy, all_accuracies)
    """
    print("Loading and cleaning data...")
    X_total, y_total = load_and_clean_data()
    
    print(f"\nTotal dataset: {X_total.shape[0]} stars, {X_total.shape[1]} features")
    print(f"  CP: {torch.sum(y_total == 1).item()}, FP/FA: {torch.sum(y_total == 0).item()}")
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    accuracies = []
    models = []
    
    # Create date-based folder for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = Path("saved_models/baseline_gp") / timestamp
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nStarting {n_splits}-fold cross-validation...")
    print(f"  Training iterations per fold: {num_iterations}")
    print(f"  Inducing points: {num_inducing}")
    print(f"  Models will be saved to: {model_dir}\n")
    
    for fold_idx, (train_indices, test_indices) in enumerate(skf.split(X_total.numpy(), y_total.numpy()), 1):
        print(f"Fold {fold_idx}/{n_splits}")
        print(f"  Train: {len(train_indices)} samples, Test: {len(test_indices)} samples")
        
        train_x = X_total[train_indices]
        train_y = y_total[train_indices]
        test_x = X_total[test_indices]
        test_y = y_total[test_indices]
        
        model, likelihood = train_single_fold(
            train_x, train_y, 
            num_iterations=num_iterations, 
            num_inducing=num_inducing,
            verbose=(fold_idx == 1)
        )
        
        accuracy = evaluate_model(model, likelihood, test_x, test_y)
        accuracies.append(accuracy)
        models.append((model, likelihood))
        
        model_path = model_dir / f"baseline_gp_fold_{fold_idx}.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'likelihood_state_dict': likelihood.state_dict(),
            'accuracy': accuracy,
            'fold': fold_idx
        }, model_path)
        print(f"  Test Accuracy: {100 * accuracy:.2f}%")
        print(f"  Model saved to: {model_path}")
        print()
    
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    
    best_fold_idx = np.argmax(accuracies) + 1
    best_model, best_likelihood = models[best_fold_idx - 1]
    best_model_path = model_dir / "baseline_gp_best.pth"
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'likelihood_state_dict': best_likelihood.state_dict(),
        'accuracy': accuracies[best_fold_idx - 1],
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


if __name__ == "__main__":
    train_baseline_gp()



