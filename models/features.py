"""
Feature extraction for TESS light curves.

This module provides functions to extract fixed-length feature vectors
from variable-length light curves for baseline GP classification models.
"""

import numpy as np
from scipy import stats
from typing import Union


def extract_features_from_curve(t: np.ndarray, flux: np.ndarray) -> np.ndarray:
    """
    Extract statistical and frequency domain features from a light curve.
    
    This function normalizes the flux by median, then computes:
    - Statistical features: mean, std, skewness, kurtosis
    - Frequency domain features: dominant period, peak power
    
    Parameters
    ----------
    t : np.ndarray
        Time array (typically in days, BJD)
    flux : np.ndarray
        Flux array (PDCSAP flux values)
    
    Returns
    -------
    np.ndarray
        1D array with 6 features: [mean, std, skewness, kurtosis, dominant_period, peak_power]
        Returns array of NaNs if any error occurs or invalid input detected.
    
    Examples
    --------
    >>> t = np.linspace(0, 10, 100)
    >>> flux = 1.0 + 0.01 * np.sin(2 * np.pi * t / 2.5)
    >>> features = extract_features_from_curve(t, flux)
    >>> len(features)
    6
    """
    try:
        # Validate inputs
        if len(t) == 0 or len(flux) == 0:
            return np.full(6, np.nan)
        
        if len(t) != len(flux):
            return np.full(6, np.nan)
        
        # Normalize flux by median (handles NaNs in raw data)
        flux_median = np.nanmedian(flux)
        if flux_median == 0 or np.isnan(flux_median):
            return np.full(6, np.nan)
        
        flux_normalized = flux / flux_median
        
        # Remove NaNs from normalized flux for statistics (if any remain)
        flux_clean = flux_normalized[~np.isnan(flux_normalized)]
        if len(flux_clean) == 0:
            return np.full(6, np.nan)
        
        # Statistical features
        mean = np.mean(flux_clean)
        std = np.std(flux_clean)
        skewness = stats.skew(flux_clean)
        kurtosis = stats.kurtosis(flux_clean)
        
        # Frequency domain features
        # Check if signal is constant (no periodic signal)
        if std < 1e-10:
            period = np.nan
            peak_power = np.nan
        else:
            dt = np.median(np.diff(t))
            if dt <= 0 or np.isnan(dt):
                # Invalid time step, return NaN for period features
                period = np.nan
                peak_power = np.nan
            else:
                # For FFT, fill NaNs with 1.0 (normalized median) to preserve array length
                flux_for_fft = flux_normalized.copy()
                flux_for_fft[np.isnan(flux_for_fft)] = 1.0
                
                # Perform FFT
                fft_result = np.fft.rfft(flux_for_fft)
                
                # Get frequency array
                frequencies = np.fft.rfftfreq(len(flux_for_fft), d=dt)
                
                # Calculate power spectrum
                power = np.abs(fft_result) ** 2
                
                # Find peak frequency (excluding DC component at index 0)
                if len(power) > 1:
                    peak_idx = np.argmax(power[1:]) + 1
                    freq_dominant = frequencies[peak_idx]
                    
                    if freq_dominant > 0:
                        period = 1.0 / freq_dominant
                    else:
                        period = np.nan
                    
                    peak_power = power[peak_idx]
                else:
                    # DC-only signal
                    period = np.nan
                    peak_power = np.nan
        
        return np.array([mean, std, skewness, kurtosis, period, peak_power], dtype=np.float64)
    
    except Exception:
        # Return NaN array on any error
        return np.full(6, np.nan)

