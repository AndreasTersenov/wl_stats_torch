# wl-stats-torch

[![Tests](https://github.com/AndreasTersenov/wl_stats_torch/actions/workflows/tests.yml/badge.svg)](https://github.com/AndreasTersenov/wl_stats_torch/actions/workflows/tests.yml)
[![Lint](https://github.com/AndreasTersenov/wl_stats_torch/actions/workflows/lint.yml/badge.svg)](https://github.com/AndreasTersenov/wl_stats_torch/actions/workflows/lint.yml)
[![Documentation](https://wl-stats-torch.readthedocs.io/en/latest/?badge=latest)](https://wl-stats-torch.readthedocs.io/en/latest/)
[![codecov](https://codecov.io/gh/AndreasTersenov/wl_stats_torch/branch/main/graph/badge.svg)](https://codecov.io/gh/AndreasTersenov/wl_stats_torch)
[![PyPI version](https://badge.fury.io/py/wl-stats-torch.svg)](https://badge.fury.io/py/wl-stats-torch)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

GPU-accelerated weak lensing summary statistics using PyTorch.

## Overview

**wl-stats-torch** computes weak lensing summary statistics commonly used in cosmological analyses:

- **Mono-scale peak counts** - Peak statistics on smoothed convergence maps
- **Wavelet (Starlet) peak counts** - Multi-scale peak detection using the starlet transform
- **Wavelet L1-norm** - Sparsity measure across wavelet scales

This package provides a fast, pure-Python alternative to the C++-dependent [CosmoStat](https://github.com/CosmoStat) implementation, with full GPU support via PyTorch.

## Key Features

| Feature | Description |
|---------|-------------|
| **Batch Processing** | 12-19x faster than sequential processing on GPU |
| **GPU Acceleration** | All operations run on CUDA devices via PyTorch |
| **No C++ Dependencies** | Pure Python implementation, no compilation required |
| **ML-Ready** | Vectorized operations for gradient-based learning workflows |
| **Memory Efficient** | Optimized for large-scale cosmological simulations |

## Installation

### From PyPI

```bash
pip install wl-stats-torch
```

### From Source

```bash
git clone https://github.com/AndreasTersenov/wl_stats_torch.git
cd wl_stats_torch
pip install -e .
```

With development dependencies:

```bash
pip install -e ".[dev]"
```

## Quick Start

### Single Image Processing

```python
import torch
from wl_stats_torch import WLStatistics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
stats = WLStatistics(n_scales=5, device=device)

# Convergence and noise maps
kappa_map = torch.randn(512, 512, device=device)
sigma_map = torch.ones(512, 512, device=device) * 0.01

# Compute all statistics
results = stats.compute_all_statistics(
    kappa_map,
    sigma_map,
    min_snr=-2,
    max_snr=6,
    n_bins=31
)

peak_counts = results['wavelet_peak_counts']  # Per-scale peak counts
l1_norms = results['wavelet_l1_norms']        # Per-scale L1-norms
mono_peaks = results['mono_peak_counts']      # Mono-scale peaks
```

### Batch Processing

Process multiple convergence maps simultaneously for significant speedups:

```python
import torch
from wl_stats_torch import WLStatistics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
stats = WLStatistics(n_scales=6, device=device)

# Batch of 128 convergence maps
kappa_batch = torch.randn(128, 512, 512, device=device)
noise_sigma = 0.01

results = stats.compute_all_statistics(
    kappa_batch,
    noise_sigma,
    min_snr=-4.0,
    max_snr=15.0,
    n_bins=51
)

# Extract features for ML pipelines
wavelet_peaks = torch.stack(results['wavelet_peak_counts'])  # (6, 128, 51)
wavelet_l1 = torch.stack(results['wavelet_l1_norms'])        # (6, 128, 100)
```

Batch processing delivers **12-19x speedup** on NVIDIA A100 compared to sequential processing.

## Components

### Starlet Transform

2D a trous wavelet transform with B3-spline kernel:

```python
from wl_stats_torch.starlet import Starlet2D

starlet = Starlet2D(n_scales=5)
wavelet_coeffs = starlet(image)  # Returns (n_scales, H, W)
```

### Peak Detection

Fast vectorized peak detection:

```python
from wl_stats_torch.peaks import find_peaks_2d

peak_positions, peak_heights = find_peaks_2d(image, threshold=3.0, mask=mask)
```

## Docker

Run without local installation using Docker:

```bash
# CPU
docker build -t wl-stats-torch:cpu .
docker run -it --rm -v $(pwd)/data:/data wl-stats-torch:cpu

# GPU (requires nvidia-docker)
docker build -t wl-stats-torch:cuda -f Dockerfile.cuda .
docker run -it --rm --gpus all -v $(pwd)/data:/data wl-stats-torch:cuda
```

See [Docker documentation](docs-md/DOCKER.md) for detailed usage.

## Documentation

- [Quick Start Guide](docs-md/QUICKSTART.md)
- [Installation](docs-md/INSTALL.md)
- [API Reference](docs-md/API.md)
- [Batch Processing Guide](docs-md/BATCH_PROCESSING.md)
- [Docker Usage](docs-md/DOCKER.md)
- [Contributing](docs-md/CONTRIBUTING.md)

Full documentation: [wl-stats-torch.readthedocs.io](https://wl-stats-torch.readthedocs.io)

## Examples

**Python Scripts** (`examples/`):
- `basic_usage.py` - Simple example with synthetic data
- `cfis_example.py` - Realistic CFIS-like simulation
- `batch_processing_demo.py` - Comprehensive batch processing examples

**Jupyter Notebooks** (`notebooks/`):
- `cuda_batch_demo.ipynb` - GPU batch processing demonstration
- `des_mock_demo.ipynb` - DES mock catalog analysis

## Citation

If you use this software in your research, please cite:

```bibtex
@software{wl_stats_torch,
  author = {Tersenov, Andreas},
  title = {wl-stats-torch: GPU-accelerated weak lensing summary statistics},
  url = {https://github.com/AndreasTersenov/wl_stats_torch},
  version = {0.1.0},
  year = {2025}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Author

Andreas Tersenov
