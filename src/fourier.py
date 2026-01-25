import numpy as np
from typing import List, Dict

def compute_dft(signal: np.ndarray) -> List[Dict]:
    """
    Computes the Discrete Fourier Transform of a 1D signal.
    
    Args:
        signal: A 1D numpy array (e.g., X or Y coordinates over time).
        
    Returns:
        A list of dictionaries containing 'freq', 'amp', and 'phase' for each coefficient,
        sorted by amplitude in descending order.
    """
    n = len(signal)
    fourier = np.fft.fft(signal)
    
    coefficients = []
    for k, val in enumerate(fourier):
        amp = np.abs(val) / n
        phase = np.angle(val)
        coefficients.append({
            'freq': k,
            'amp': amp,
            'phase': phase
        })
        
    # Sort by amplitude for better visualization of epicycles
    coefficients.sort(key=lambda x: x['amp'], reverse=True)
    
    return coefficients
