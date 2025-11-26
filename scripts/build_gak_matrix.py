"""
Build GAK (Global Alignment Kernel) similarity matrix for main GP model.

This script processes all stars in the training index, normalizes and downsamples
their light curves, computes pairwise GAK similarities, and saves the resulting
matrix to disk for GP classification model training.
"""

import numpy as np
import time
from pathlib import Path
from models.gp_data import load_all_from_index
from tslearn.metrics import cdist_gak, sigma_gak


def build_gak_matrix(
    index_path: str = "dataset/index.csv",
    output_dir: str = "dataset",
    x_filename: str = "X_gak_matrix.npy",
    y_filename: str = "y_labels.npy",
    downsample_factor: int = 10,
    normalize_flux: bool = True,
    sigma: str = "auto",
    max_stars: int = None
):
    """
    Build GAK similarity matrix from all stars in the training index.
    
    Parameters
    ----------
    index_path : str, optional
        Path to index.csv file (default: "dataset/index.csv")
    output_dir : str, optional
        Directory to save output files (default: "dataset")
    x_filename : str, optional
        Filename for GAK matrix (default: "X_gak_matrix.npy")
    y_filename : str, optional
        Filename for labels (default: "y_labels.npy")
    downsample_factor : int, optional
        Downsampling factor - take every Nth point (default: 10)
        Laptop testing (N=150): May need 30-50 for overnight completion
        Cloud production (N=1900): May use 5-10 for higher resolution
    normalize_flux : bool, optional
        Whether to normalize flux by median (default: True)
    sigma : str or float, optional
        GAK kernel bandwidth parameter (default: "auto")
        Controls how quickly similarity drops off with distance
        "auto" uses tslearn's internal heuristics
        Can be set to float value (e.g., 1.0) for manual tuning
    max_stars : int, optional
        Maximum number of stars to process (default: None, process all)
        Useful for testing with a small subset
    
    Returns
    -------
    tuple
        (K, y) where:
        - K: 2D numpy array (N stars × N stars) GAK similarity matrix
        - y: 1D numpy array (N labels)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading light curves from {index_path}...")
    
    flux_list = []
    y_list = []
    star_count = 0
    skipped_count = 0
    
    original_lengths = []
    downsampled_lengths = []
    
    for tic_id, label, curve_dict, metadata in load_all_from_index(index_path, include_metadata=False):
        star_count += 1
        
        t = curve_dict['t']
        flux = curve_dict['flux']
        
        if len(flux) == 0:
            print(f"  Warning: TIC {tic_id} has empty flux array, skipping")
            skipped_count += 1
            continue
        
        original_lengths.append(len(flux))
        
        if normalize_flux:
            flux_median = np.nanmedian(flux)
            if flux_median == 0 or np.isnan(flux_median):
                print(f"  Warning: TIC {tic_id} has invalid median flux, skipping")
                skipped_count += 1
                continue
            flux_normalized = flux / flux_median
        else:
            flux_normalized = flux.copy()
        
        valid_mask = ~np.isnan(flux_normalized)
        flux_clean = flux_normalized[valid_mask]
        
        if len(flux_clean) == 0:
            print(f"  Warning: TIC {tic_id} has no valid flux values after normalization, skipping")
            skipped_count += 1
            continue
        
        flux_downsampled = flux_clean[::downsample_factor]
        
        if len(flux_downsampled) < 10:
            print(f"  Warning: TIC {tic_id} has too few points after downsampling ({len(flux_downsampled)}), skipping")
            skipped_count += 1
            continue
        
        downsampled_lengths.append(len(flux_downsampled))
        flux_list.append(flux_downsampled)
        y_list.append(label)
        
        # Check if we've reached max_stars limit
        if max_stars is not None and len(flux_list) >= max_stars:
            print(f"  Reached max_stars limit ({max_stars}), stopping...")
            break
        
        if star_count % 10 == 0:
            print(f"  Processed {star_count} stars...")
    
    print(f"\nProcessed {star_count} stars total")
    if skipped_count > 0:
        print(f"  Skipped {skipped_count} stars due to invalid data")
    
    if len(flux_list) == 0:
        raise ValueError("No valid stars processed. Check that index.csv exists and contains valid entries.")
    
    if original_lengths and downsampled_lengths:
        avg_original = np.mean(original_lengths)
        avg_downsampled = np.mean(downsampled_lengths)
        print(f"\nDownsampling statistics:")
        print(f"  Average original length: {avg_original:.0f} points")
        print(f"  Average downsampled length: {avg_downsampled:.0f} points")
        print(f"  Downsampling factor: {downsample_factor}")
    
    def resample_1d(x, target_len=512):
        """Resample 1D array to fixed length using linear interpolation."""
        if len(x) == target_len:
            return x.copy()
        old = np.linspace(0, 1, len(x))
        new = np.linspace(0, 1, target_len)
        return np.interp(new, old, x)
    
    target_len = 400
    print(f"\nResampling {len(flux_list)} light curves to fixed length {target_len}...")
    flux_reshaped = [
        resample_1d(f, target_len).reshape(-1, 1).astype(np.float64)
        for f in flux_list
    ]
    print(f"  All series now have length {target_len}")
    
    if sigma == "auto":
        try:
            sigma_est = sigma_gak(flux_reshaped)
            sigma = float(np.clip(sigma_est, 0.1, 5.0))
            print(f"\nsigma_est: {sigma_est:.6f}, using sigma: {sigma:.6f} (clipped to [0.1, 5.0])")
        except Exception as e:
            print(f"\nWarning: sigma_gak failed ({e}), using fixed sigma=1.0")
            sigma = 1.0
    
    n_stars = len(flux_reshaped)
    print(f"\nStarting GAK computation on N={n_stars} stars.")
    print(f"  This may take 30-60+ minutes on a laptop. Please wait...")
    print(f"  Using sigma={sigma}")
    
    start_time = time.time()
    print("\nComputing GAK matrix...")
    try:
        K = cdist_gak(flux_reshaped, flux_reshaped, sigma=sigma, be="numpy")
    except Exception as e:
        print(f"  Error during GAK computation: {e}")
        print(f"  Retrying with sigma=1.0...")
        sigma = 1.0
        K = cdist_gak(flux_reshaped, flux_reshaped, sigma=sigma, be="numpy")
    elapsed_time = time.time() - start_time
    
    print(f"\nGAK computation completed in {elapsed_time/60:.1f} minutes ({elapsed_time:.1f} seconds)")
    
    K = np.array(K)
    y = np.array(y_list, dtype=np.int32)
    
    print(f"\nMatrix properties:")
    print(f"  Shape: {K.shape}")
    print(f"  Min value: {K.min():.6f}")
    print(f"  Max value: {K.max():.6f}")
    print(f"  Mean value: {K.mean():.6f}")
    
    is_symmetric = np.allclose(K, K.T, rtol=1e-10)
    print(f"  Is symmetric: {is_symmetric}")
    
    diagonal_values = np.diag(K)
    diagonal_all_one = np.allclose(diagonal_values, 1.0, rtol=1e-10)
    print(f"  Diagonal all 1.0: {diagonal_all_one}")
    if not diagonal_all_one:
        print(f"    Diagonal range: [{diagonal_values.min():.6f}, {diagonal_values.max():.6f}]")
    
    x_path = output_path / x_filename
    y_path = output_path / y_filename
    
    np.save(x_path, K)
    np.save(y_path, y)
    
    print(f"\nGAK matrix saved:")
    print(f"  K: {x_path} (shape: {K.shape})")
    print(f"  y: {y_path} (shape: {y.shape})")
    
    return K, y


if __name__ == "__main__":
    build_gak_matrix()

