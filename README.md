# Weak Lensing Summary Statistics (PyTorch)

A GPU-accelerated PyTorch implementation for computing weak lensing summary statistics including:
- Mono-scale peak counts
- Wavelet (Starlet) peak counts  
- Wavelet L1-norm 

This package provides a fast, pure-Python alternative to the C++-dependent CosmoStat implementation, with full GPU support via PyTorch.

## Features

- **GPU Acceleration**: All operations are PyTorch-based and can run on CUDA devices
- **No C++ Dependencies**: Pure Python implementation, no compilation required
- **Batch Processing**: Efficiently process multiple maps simultaneously
- **Memory Efficient**: Optimized for large-scale cosmological simulations

## Installation

```bash
pip install -e .
```

Or with development dependencies:
```bash
pip install -e ".[dev]"
```

## Quick Start

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
    nbins=31
)

# Access results
peak_counts = results['wavelet_peak_counts']  # Peak counts per scale
l1_norms = results['wavelet_l1_norms']  # L1-norms per scale
mono_peaks = results['mono_peak_counts']  # Mono-scale peak counts
```

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

**Jupyter Notebooks** (`notebooks/`):
- `cuda_batch_demo.ipynb` - GPU batch processing demonstration
- `des_mock_demo.ipynb` - DES mock catalog analysis
- `pycs_demo.ipynb` - PyCS integration example

## Documentation

- 📖 **User Guide**: See `docs-md/` directory for detailed documentation
- 🚀 **Quick Start**: `docs-md/QUICKSTART.md`
- 📦 **Installation**: `docs-md/INSTALL.md`
- 🔧 **Contributing**: `docs-md/CONTRIBUTING.md`
- 🔍 **API Reference**: `docs-md/API.md`
- ✅ **Test Fixes**: `docs-md/TEST_FIXES.md`

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
