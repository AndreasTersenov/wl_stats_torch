"""
Weak Lensing Summary Statistics with PyTorch

A GPU-accelerated package for computing weak lensing summary statistics
including wavelet peak counts and L1-norms.
"""

__version__ = "0.1.0"
__author__ = "Andreas Tersenov"

from .peaks import find_peaks_2d, find_peaks_batch
from .starlet import Starlet2D
from .statistics import WLStatistics

__all__ = [
    "Starlet2D",
    "find_peaks_2d",
    "find_peaks_batch",
    "WLStatistics",
]
