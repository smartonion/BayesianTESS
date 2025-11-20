"""
DataLoader class for aggregating light curve data into lists for ML training.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple

from .gp_data import load_lightcurve_from_path


class DataLoader:
    """Class-based data loader that aggregates light curve data into lists."""
    
    def __init__(self, index_path: str = "dataset/index.csv"):
        """
        Initialize the DataLoader with an index file.
        
        Parameters
        ----------
        index_path : str, optional
            Path to the index.csv file (default: "dataset/index.csv")
        
        Raises
        ------
        FileNotFoundError
            If the index file does not exist
        ValueError
            If required columns are missing
        """
        self.index_path = Path(index_path)
        
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index file not found: {self.index_path}")
        
        self.index_df = pd.read_csv(self.index_path)
        
        required_columns = ['tic_id', 'npz_path', 'label']
        missing_columns = [col for col in required_columns if col not in self.index_df.columns]
        if missing_columns:
            raise ValueError(f"Index file missing required columns: {missing_columns}")
    
    def load_data(self) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Load all light curve data and aggregate into lists.
        
        Returns
        -------
        tuple
            (X, y) where:
            - X: List of NumPy arrays, each containing flux values for one star
            - y: 1D NumPy array of binary labels (0 = FP/FA, 1 = CP)
        
        Raises
        ------
        FileNotFoundError
            If any NPZ file referenced in the index does not exist
        """
        X = []
        y = []
        dataset_root = self.index_path.parent
        
        for _, row in self.index_df.iterrows():
            npz_path = row['npz_path']
            label = int(row['label'])
            
            if not Path(npz_path).is_absolute():
                full_path = dataset_root / npz_path
            else:
                full_path = Path(npz_path)
            
            flux = self._load_single_star(str(full_path))
            X.append(flux)
            y.append(label)
        
        return X, np.array(y, dtype=np.int32)
    
    def _load_single_star(self, npz_path: str) -> np.ndarray:
        """
        Load flux array for a single star from NPZ file.
        
        Parameters
        ----------
        npz_path : str
            Path to the NPZ file (can be relative or absolute)
        
        Returns
        -------
        np.ndarray
            Flux array for the star
        
        Raises
        ------
        FileNotFoundError
            If the NPZ file does not exist
        KeyError
            If the NPZ file does not contain a 'flux' key
        """
        curve_dict = load_lightcurve_from_path(npz_path)
        
        if 'flux' not in curve_dict:
            raise KeyError(f"NPZ file does not contain 'flux' key: {npz_path}")
        
        return curve_dict['flux']

