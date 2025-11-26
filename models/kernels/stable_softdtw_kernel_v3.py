"""
GPyTorch kernel using SoftDTW for time series data (V3).

This version uses the original soft_dtw_cuda backend with normalize=True
for batch-independent, pairwise normalization. This ensures k(x,y) depends
only on x and y, not on other data in the batch.

Key changes from V2:
- Uses soft_dtw_cuda directly (no wrapper)
- Sets normalize=True for batch-independent normalization
- Removes assume_self parameter (not needed with normalize=True)
- Simplifies normalization to just divide by gamma
- Removes fill_diagonal_ block (self-distances already ~0 with normalize=True)
- Lets ScaleKernel learn overall amplitude
"""

import torch
import gpytorch
from gpytorch.kernels import Kernel
from gpytorch.constraints import Positive
from models.kernels.backend.soft_dtw_cuda import SoftDTW


class StableSoftDTWKernelV3(Kernel):
    """
    GPyTorch kernel based on Soft-DTW for time series similarity (V3).
    
    This version uses normalize=True in the SoftDTW backend to ensure
    batch-independent, pairwise normalization. The kernel value k(x,y)
    depends only on x and y, not on other data in the batch.
    
    Uses tiled computation for memory efficiency, making it scalable to
    large datasets (e.g., 1901 curves at L=1024).
    
    Input tensors should be 3D: (batch, length, features)
    where length is the sequence length (e.g., 1024) and features=1 for flux.
    
    Parameters
    ----------
    gamma : float, optional
        Bandwidth parameter controlling softness of the DTW alignment
        (default: 1.0). Must be positive. This is a fixed hyperparameter.
    use_cuda : bool, optional
        Whether to use CUDA. If None, auto-detects based on device.
        (default: None)
    enforce_psd : bool, optional
        If True, project kernel matrix to nearest PSD matrix (Tier 2).
        This guarantees positive semi-definiteness but makes the kernel
        data-dependent per batch. Default: False.
    tile_size : int, optional
        Size of each tile for tiled kernel computation (default: 32).
        Controls memory usage: memory per tile is O(t^2 L^2).
        For local testing (150 curves): use 64-128.
        For full runs (1901 curves): use 32-64 depending on GPU VRAM.
    bandwidth : int, optional
        Sakoe-Chiba band for pruning, None = full DP (default: None).
        Can significantly speed up computation at L=1024.
        Suggested values: 64-128 for L=1024.
    """
    
    has_lengthscale = False
    
    def __init__(self, gamma=1.0, use_cuda=None, enforce_psd=False,
                 tile_size=32, bandwidth=None, **kwargs):
        """
        Initialize the SoftDTW kernel (V3).
        
        Parameters
        ----------
        gamma : float, optional
            Bandwidth parameter controlling softness of the DTW alignment
            (default: 1.0). Must be positive. This is a fixed hyperparameter.
        use_cuda : bool, optional
            Whether to use CUDA. If None, auto-detects based on device.
            (default: None)
        enforce_psd : bool, optional
            If True, project kernel matrix to nearest PSD matrix (Tier 2).
            This guarantees positive semi-definiteness but makes the kernel
            data-dependent per batch. Default: False.
        tile_size : int, optional
            Size of each tile for tiled kernel computation (default: 32).
            Controls memory usage: memory per tile is O(t^2 L^2).
            For local testing (150 curves): use 64-128.
            For full runs (1901 curves): use 32-64 depending on GPU VRAM.
        bandwidth : int, optional
            Sakoe-Chiba band for pruning, None = full DP (default: None).
            Can significantly speed up computation at L=1024.
            Suggested values: 64-128 for L=1024.
        """
        super(StableSoftDTWKernelV3, self).__init__(**kwargs)
        
        # Gamma is a fixed hyperparameter (not learnable)
        if gamma <= 0:
            raise ValueError(f"gamma must be positive, got {gamma}")
        self.register_buffer("gamma", torch.tensor(float(gamma)))
        
        self._use_cuda_preference = use_cuda
        self.enforce_psd = enforce_psd
        self.tile_size = int(tile_size)
        
        # Sakoe-Chiba band for pruning, None = full DP
        self.bandwidth = bandwidth
        
        self._softdtw_cache = {}
    
    def _normalize_distances(self, distances):
        """
        Normalize distances by dividing by gamma.
        
        With normalize=True in SoftDTW, distances are already normalized
        in a batch-independent, pairwise way. We just need to scale by gamma
        to control the kernel bandwidth.
        
        Parameters
        ----------
        distances : torch.Tensor
            Raw SoftDTW distances (already normalized by SoftDTW backend)
        
        Returns
        -------
        torch.Tensor
            Normalized distances
        """
        return distances / self.gamma
    
    def _ensure_psd(self, K, jitter=1e-6):
        """
        Project kernel matrix to nearest positive semi-definite matrix.
        
        This is a Tier 2 optional feature that guarantees PSD by:
        1. Symmetrizing the matrix
        2. Computing eigendecomposition
        3. Clipping negative eigenvalues to jitter
        4. Reconstructing the matrix
        
        Supports batched kernels: K can be shape (..., N, N)
        
        Parameters
        ----------
        K : torch.Tensor
            Kernel matrix, shape (..., N, N) or (N, N)
        jitter : float, optional
            Minimum eigenvalue threshold (default: 1e-6)
        
        Returns
        -------
        torch.Tensor
            PSD kernel matrix, same shape as input
        """
        # Assume K is (..., N, N) - handle both batched and non-batched
        # Symmetrize just in case of small asymmetry
        K = 0.5 * (K + K.transpose(-1, -2))
        
        # Eigendecomposition (handles batched case automatically)
        eigvals, eigvecs = torch.linalg.eigh(K)
        
        # Clip negative eigenvalues to jitter
        eigvals_clipped = torch.clamp(eigvals, min=jitter)
        
        # Reconstruct PSD matrix: eigvecs @ diag(eigvals_clipped) @ eigvecs.T
        # Use einsum or matmul for proper broadcasting
        K_psd = (eigvecs * eigvals_clipped.unsqueeze(-2)) @ eigvecs.transpose(-1, -2)
        
        return K_psd
    
    def forward(self, x1, x2, diag=False, **params):
        """
        Compute kernel matrix between two sets of time series.
        
        Parameters
        ----------
        x1 : torch.Tensor
            First set of time series, shape (B1, L, D)
        x2 : torch.Tensor
            Second set of time series, shape (B2, L, D)
        diag : bool, optional
            If True, return only diagonal elements (default: False)
        
        Returns
        -------
        torch.Tensor
            Kernel matrix of shape (B1, B2) if diag=False
            or diagonal vector of shape (B1,) if diag=True
        """
        # Input validation
        assert x1.dim() == 3, f"x1 must be 3D (batch, length, features), got {x1.dim()}D"
        assert x2.dim() == 3, f"x2 must be 3D (batch, length, features), got {x2.dim()}D"
        
        device = x1.device
        x1 = x1.to(device)
        x2 = x2.to(device)
        
        B1, L1, D1 = x1.shape
        B2, L2, D2 = x2.shape
        
        if L1 != L2:
            raise ValueError(f"Sequence lengths must match: x1 has L={L1}, x2 has L={L2}")
        if D1 != D2:
            raise ValueError(f"Feature dimensions must match: x1 has D={D1}, x2 has D={D2}")
        
        use_cuda = self._use_cuda_preference
        if use_cuda is None:
            use_cuda = x1.is_cuda
        
        # Gamma is fixed per kernel instance, so cache key only needs device type
        gamma_float = float(self.gamma)
        cache_key = "cuda" if use_cuda else "cpu"
        
        if cache_key not in self._softdtw_cache:
            softdtw = SoftDTW(
                use_cuda=use_cuda,
                gamma=gamma_float,
                normalize=True,  # Batch-independent, pairwise normalization
                bandwidth=self.bandwidth,  # Sakoe-Chiba band for pruning
                dist_func=None,
            )
            self._softdtw_cache[cache_key] = softdtw
        else:
            softdtw = self._softdtw_cache[cache_key]
        
        if diag:
            # Handle diagonal case: k(x_i, x_i) or k(x1[i], x2[i])
            if x1.shape == x2.shape and torch.allclose(x1, x2, atol=1e-6):
                # Self case: k(x_i, x_i) = 1 exactly
                # With normalize=True, self-distances are already ~0 up to numerical noise
                return torch.ones(x1.shape[0], device=x1.device, dtype=x1.dtype)
            else:
                # General diag: compute SoftDTW(x1[i], x2[i]) for each i
                # Note: This requires x1 and x2 to have same batch size
                if B1 != B2:
                    raise ValueError(
                        f"For diag=True with different inputs, batch sizes must match: "
                        f"got B1={B1}, B2={B2}"
                    )
                distances = softdtw(x1, x2)  # (B1,)
                distances_normalized = self._normalize_distances(distances)
                return torch.exp(-distances_normalized)
        
        # Full kernel matrix computation, tiled for memory efficiency
        t = self.tile_size
        kernel_matrix = x1.new_empty(B1, B2)
        
        for i0 in range(0, B1, t):
            i1 = min(i0 + t, B1)
            x1_blk = x1[i0:i1]                      # (b1, L, D)
            b1 = i1 - i0
            
            for j0 in range(0, B2, t):
                j1 = min(j0 + t, B2)
                x2_blk = x2[j0:j1]                  # (b2, L, D)
                b2 = j1 - j0
                
                # Pairwise within block: (b1, b2, L, D) -> (b1*b2, L, D)
                x1_exp = x1_blk.unsqueeze(1).expand(b1, b2, L1, D1)
                x2_exp = x2_blk.unsqueeze(0).expand(b1, b2, L2, D2)
                x1_flat = x1_exp.reshape(b1 * b2, L1, D1)
                x2_flat = x2_exp.reshape(b1 * b2, L2, D2)
                
                d_flat = softdtw(x1_flat, x2_flat)          # (b1*b2,)
                d_blk = d_flat.reshape(b1, b2)
                
                d_blk = self._normalize_distances(d_blk)   # divide by gamma
                kernel_matrix[i0:i1, j0:j1] = torch.exp(-d_blk)
        
        # If this was a self-Gram, enforce exact symmetry to kill tiny numerical asymmetry
        if x1.data_ptr() == x2.data_ptr():
            kernel_matrix = 0.5 * (kernel_matrix + kernel_matrix.transpose(-1, -2))
        
        # Tier 2: Optional PSD enforcement
        if self.enforce_psd:
            kernel_matrix = self._ensure_psd(kernel_matrix)
        
        return kernel_matrix

