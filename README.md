# Weak Lensing Summary Statistics (PyTorch)

[![Tests](https://github.com/AndreasTersenov/wl_stats_torch/actions/workflows/tests.yml/badge.svg)](https://github.com/AndreasTersenov/wl_stats_torch/actions/workflows/tests.yml)
[![Lint](https://github.com/AndreasTersenov/wl_stats_torch/actions/workflows/lint.yml/badge.svg)](https://github.com/AndreasTersenov/wl_stats_torch/actions/workflows/lint.yml)
[![Documentation](https://github.com/AndreasTersenov/wl_stats_torch/actions/workflows/docs.yml/badge.svg)](https://github.com/AndreasTersenov/wl_stats_torch/actions/workflows/docs.yml)
[![codecov](https://codecov.io/gh/AndreasTersenov/wl_stats_torch/branch/main/graph/badge.svg)](https://codecov.io/gh/AndreasTersenov/wl_stats_torch)
[![PyPI version](https://badge.fury.io/py/wl-stats-torch.svg)](https://badge.fury.io/py/wl-stats-torch)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A GPU-accelerated PyTorch implementation for computing weak lensing summary statistics including:
- Mono-scale peak counts
- Wavelet (Starlet) peak counts  
- Wavelet L1-norm 

This package provides a fast, pure-Python alternative to the C++-dependent CosmoStat implementation, with full GPU support via PyTorch.

## Features

- **Optimized Batch Processing**: 12-19x faster than sequential processing on GPU
- **GPU Acceleration**: All operations are PyTorch-based and can run on CUDA devices
- **No C++ Dependencies**: Pure Python implementation, no compilation required
- **ML-Ready**: Vectorized operations ideal for gradient-based learning workflows
- **Memory Efficient**: Optimized for large-scale cosmological simulations
- **Backward Compatible**: Single-image API unchanged, batch support added seamlessly

## Installation

```bash
pip install -e .
```

Or with development dependencies:
```bash
pip install -e ".[dev]"
```

## Quick Start

### Single Image

```python
import torch
from wl_stats_torch import WLStatistics

# Initialize with device (cpu or cuda)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
stats = WLStatistics(n_scales=5, device=device)

# Your convergence map and noise map
kappa_map = torch.randn(512, 512, device=device)
sigma_map = torch.ones(512, 512, device=device) * 0.01

# Compute statistics
results = stats.compute_all_statistics(
    kappa_map, 
    sigma_map,
    min_snr=-2, 
    max_snr=6, 
    n_bins=31
)

# Access results
peak_counts = results['wavelet_peak_counts']  # Peak counts per scale
l1_norms = results['wavelet_l1_norms']  # L1-norms per scale
mono_peaks = results['mono_peak_counts']  # Mono-scale peak counts
```

### Batch Processing (NEW!)

```python
import torch
from wl_stats_torch import WLStatistics

# Process multiple convergence maps at once
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
stats = WLStatistics(n_scales=6, device=device)

# Batch of 128 convergence maps
kappa_batch = torch.randn(128, 512, 512, device=device)
noise_sigma = 0.01

# Compute statistics for entire batch
results = stats.compute_all_statistics(
    kappa_batch,
    noise_sigma,
    min_snr=-4.0,
    max_snr=15.0,
    n_bins=51,
    l1_nbins=100,
    compute_mono=False
)

# Extract batched features for ML training
wavelet_peaks = torch.stack(results['wavelet_peak_counts'])  # (6, 128, 51)
wavelet_l1 = torch.stack(results['wavelet_l1_norms'])        # (6, 128, 100)

# Reshape to (128, features) for neural network input
features = torch.cat([
    wavelet_peaks.permute(1, 0, 2).flatten(1),  # (128, 306)
    wavelet_l1.permute(1, 0, 2).flatten(1)      # (128, 600)
], dim=1)  # Final shape: (128, 906)

# 12-19x faster than processing sequentially! 🚀
```

**Performance:** Batch processing delivers **12-19x speedup** on NVIDIA A100 compared to sequential processing (validated with 256×256 images, batch size 4).

**See `docs-md/BATCH_PROCESSING.md` for complete batch processing guide with optimization tips!**

## Components

### Starlet Transform
2D à trous wavelet transform with B3-spline kernel:
```python
from wl_stats_torch.starlet import Starlet2D

starlet = Starlet2D(n_scales=5)
wavelet_coeffs = starlet(image)  # Returns (n_scales, H, W)
```

### Peak Detection
Fast vectorized peak detection:
```python
from wl_stats_torch.peaks import find_peaks_2d

peak_positions, peak_heights = find_peaks_2d(
    image, 
    threshold=3.0,
    mask=mask
)
```

### Full Statistics Pipeline
```python
from wl_stats_torch import WLStatistics

stats = WLStatistics(n_scales=5)
results = stats.compute_all_statistics(kappa, sigma)
```

## Examples

See the `examples/` directory for Python scripts and `notebooks/` for Jupyter notebooks:

**Python Scripts** (`examples/`):
- `basic_usage.py` - Simple example with synthetic data
- `cfis_example.py` - Realistic CFIS-like simulation
- `batch_processing.py` - Processing multiple maps efficiently
- `batch_processing_demo.py` - NEW! Comprehensive batch processing demonstrations

**Jupyter Notebooks** (`notebooks/`):
- `cuda_batch_demo.ipynb` - GPU batch processing demonstration
- `des_mock_demo.ipynb` - DES mock catalog analysis
- `pycs_demo.ipynb` - PyCS integration example

## Documentation

- **User Guide**: See `docs-md/` directory for detailed documentation
- **Quick Start**: `docs-md/QUICKSTART.md`
- **Installation**: `docs-md/INSTALL.md`
- **Contributing**: `docs-md/CONTRIBUTING.md`
- **API Reference**: `docs-md/API.md`
- **Batch Processing**: `docs-md/BATCH_PROCESSING.md`
- **Test Fixes**: `docs-md/TEST_FIXES.md`

Build the full documentation with Sphinx:
```bash
make docs
# Open docs/_build/html/index.html
```

## Citation

If you use this code, please cite:
- Original CosmoStat package and relevant papers
- This PyTorch implementation

## License

MIT License - See LICENSE file

## Authors

<!-- Based on the CosmoStat package by Jean-Luc Starck et al. -->
<!-- PyTorch implementation by  -->
Andreas Tersenov
