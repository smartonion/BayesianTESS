"""
Build feature matrix for baseline GP model.

This script processes all stars in the training index, extracts features
using extract_features_from_curve, and saves the feature matrix (X) and
labels (y) to disk for baseline GP classification model training.
"""

import numpy as np
from pathlib import Path
from models.gp_data import load_all_from_index
from models.features import extract_features_from_curve


def build_feature_matrix(
    index_path: str = "dataset/index.csv",
    output_dir: str = "dataset",
    x_filename: str = "X_features_baseline.npy",
    y_filename: str = "y_labels_baseline.npy"
):
    """
    Build feature matrix from all stars in the training index.
    
    Parameters
    ----------
    index_path : str, optional
        Path to index.csv file (default: "dataset/index.csv")
    output_dir : str, optional
        Directory to save output files (default: "dataset")
    x_filename : str, optional
        Filename for feature matrix (default: "X_features_baseline.npy")
    y_filename : str, optional
        Filename for labels (default: "y_labels_baseline.npy")
    
    Returns
    -------
    tuple
        (X, y) where:
        - X: 2D numpy array (N stars × 6 features)
        - y: 1D numpy array (N labels)
    """
    X_list = []
    y_list = []
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing stars from {index_path}...")
    
    nan_count = 0
    star_count = 0
    
    for tic_id, label, curve_dict, metadata in load_all_from_index(index_path, include_metadata=False):
        star_count += 1
        
        t = curve_dict['t']
        flux = curve_dict['flux']
        
        features = extract_features_from_curve(t, flux)
        
        if np.any(np.isnan(features)):
            nan_count += 1
        
        X_list.append(features)
        y_list.append(label)
        
        if star_count % 10 == 0:
            print(f"  Processed {star_count} stars...")
    
    print(f"Processed {star_count} stars total")
    
    if len(X_list) == 0:
        raise ValueError("No stars processed. Check that index.csv exists and contains valid entries.")
    
    X = np.vstack(X_list)
    y = np.array(y_list, dtype=np.int32)
    
    x_path = output_path / x_filename
    y_path = output_path / y_filename
    
    np.save(x_path, X)
    np.save(y_path, y)
    
    print(f"\nFeature matrix saved:")
    print(f"  X: {x_path} (shape: {X.shape})")
    print(f"  y: {y_path} (shape: {y.shape})")
    
    if nan_count > 0:
        print(f"\nWarning: {nan_count} stars ({100*nan_count/star_count:.1f}%) have NaN features")
        print("  These rows should be filtered before training.")
    
    return X, y


if __name__ == "__main__":
    build_feature_matrix()

