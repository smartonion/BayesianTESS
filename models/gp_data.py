"""
Data loader for TESS light curves in NPZ format.

This module provides functions to load light curve data from NPZ files,
abstracting away file system details so GP scripts can focus on modeling.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Iterator, Tuple, Optional


def load_lightcurve(tic_id: int, npz_dir: str = "dataset/lightcurves_npz/") -> Dict[str, np.ndarray]:
    """
    Load a single light curve from NPZ file by TIC ID.
    
    Parameters
    ----------
    tic_id : int
        TESS Input Catalog ID
    npz_dir : str, optional
        Directory containing NPZ files (default: "dataset/lightcurves_npz/")
    
    Returns
    -------
    dict
        Dictionary containing light curve arrays:
        - 't': Time array (BJD, float64)
        - 'flux': PDCSAP flux (float32)
        - 'flux_err': Flux error (float32)
        - 'quality': Quality flags (int32)
        - 'sector': Sector numbers (int16)
        - 'camera': Camera IDs (int8)
        - 'ccd': CCD IDs (int8)
        - 'bkg': Background flux (optional, float32)
        - 'crowdsap': Crowding metric (optional, float32)
    
    Raises
    ------
    FileNotFoundError
        If NPZ file for the given TIC ID does not exist
    """
    npz_path = Path(npz_dir) / f"tic_{tic_id}.npz"
    
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found for TIC {tic_id}: {npz_path}")
    
    data = np.load(npz_path)
    
    # Extract all arrays into a dictionary
    curve_dict = {key: data[key] for key in data.files}
    
    return curve_dict


def load_lightcurve_from_path(npz_path: str) -> Dict[str, np.ndarray]:
    """
    Load a single light curve from NPZ file by path.
    
    Parameters
    ----------
    npz_path : str
        Path to NPZ file (can be relative or absolute)
    
    Returns
    -------
    dict
        Dictionary containing light curve arrays (same format as load_lightcurve)
    
    Raises
    ------
    FileNotFoundError
        If NPZ file does not exist
    """
    npz_path = Path(npz_path)
    
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")
    
    data = np.load(npz_path)
    curve_dict = {key: data[key] for key in data.files}
    
    return curve_dict


def load_all_from_index(
    index_path: str = "dataset/index.csv",
    include_metadata: bool = True
) -> Iterator[Tuple[int, int, Dict[str, np.ndarray], Optional[Dict]]]:
    """
    Load all light curves from the training index.
    
    This is a generator function that yields light curves one at a time,
    making it memory-efficient for large datasets.
    
    Parameters
    ----------
    index_path : str, optional
        Path to index.csv file (default: "dataset/index.csv")
    include_metadata : bool, optional
        If True, also yield metadata dict with period_days and t0_bjd (default: True)
    
    Yields
    ------
    tuple
        If include_metadata=True:
            (tic_id, label, curve_dict, metadata_dict)
        If include_metadata=False:
            (tic_id, label, curve_dict, None)
        
        Where:
        - tic_id: TESS Input Catalog ID (int)
        - label: Binary label (1 = CP, 0 = FP/FA) (int)
        - curve_dict: Dictionary with light curve arrays (same as load_lightcurve)
        - metadata_dict: Dictionary with 'period_days' and 't0_bjd' (optional, can be None)
    """
    index_df = pd.read_csv(index_path)
    
    for _, row in index_df.iterrows():
        tic_id = int(row['tic_id'])
        label = int(row['label'])
        npz_path = row['npz_path']
        
        # Load light curve
        # Handle relative paths (from dataset root)
        if not Path(npz_path).is_absolute():
            # If path is relative, assume it's relative to dataset directory
            dataset_root = Path(index_path).parent
            npz_path = dataset_root / npz_path
        
        curve_dict = load_lightcurve_from_path(str(npz_path))
        
        # Prepare metadata if requested
        metadata = None
        if include_metadata:
            metadata = {}
            if pd.notna(row.get('period_days')):
                metadata['period_days'] = float(row['period_days'])
            if pd.notna(row.get('t0_bjd')):
                metadata['t0_bjd'] = float(row['t0_bjd'])
        
        yield tic_id, label, curve_dict, metadata


def load_all_from_index_simple(
    index_path: str = "dataset/index.csv"
) -> Iterator[Tuple[int, int, Dict[str, np.ndarray]]]:
    """
    Simplified version that yields only (tic_id, label, curve_dict).
    
    Convenience function for cases where metadata is not needed.
    
    Parameters
    ----------
    index_path : str, optional
        Path to index.csv file (default: "dataset/index.csv")
    
    Yields
    ------
    tuple
        (tic_id, label, curve_dict)
    """
    for tic_id, label, curve_dict, _ in load_all_from_index(index_path, include_metadata=False):
        yield tic_id, label, curve_dict


def get_index_info(index_path: str = "dataset/index.csv") -> Dict:
    """
    Get summary information about the dataset index.
    
    Parameters
    ----------
    index_path : str, optional
        Path to index.csv file (default: "dataset/index.csv")
    
    Returns
    -------
    dict
        Dictionary with summary statistics:
        - 'total_stars': Total number of stars
        - 'cp_count': Number of confirmed planets (label=1)
        - 'fp_count': Number of false positives/alarms (label=0)
        - 'with_period': Number of stars with period_days
        - 'with_t0': Number of stars with t0_bjd
    """
    index_df = pd.read_csv(index_path)
    
    info = {
        'total_stars': len(index_df),
        'cp_count': len(index_df[index_df['label'] == 1]),
        'fp_count': len(index_df[index_df['label'] == 0]),
        'with_period': index_df['period_days'].notna().sum(),
        'with_t0': index_df['t0_bjd'].notna().sum()
    }
    
    return info

