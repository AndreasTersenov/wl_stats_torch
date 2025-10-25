# Quick Start Guide

## Installation

```bash
cd wl_summary_stats_torch
pip install -e .
```

## Minimal Example

```python
import torch
from wl_stats_torch import WLStatistics

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
stats = WLStatistics(n_scales=5, device=device)

# Your data
kappa_map = torch.randn(512, 512, device=device)  # Convergence map
sigma_map = torch.ones(512, 512, device=device) * 0.01  # Noise std

# Compute all statistics
results = stats.compute_all_statistics(kappa_map, sigma_map)

# Access results
peak_counts = results['wavelet_peak_counts']  # List of peak counts per scale
l1_norms = results['wavelet_l1_norms']  # List of L1-norms per scale
```

## What Does It Compute?

### 1. Wavelet Peak Counts
Number of local maxima in wavelet coefficients as a function of SNR threshold, computed at multiple scales.

```python
# For each scale j = 1, ..., n_scales:
#   1. Compute wavelet coefficients W_j
#   2. Compute SNR = W_j / noise_level_j
#   3. Find local maxima in SNR map
#   4. Create histogram of peak heights
```

**Use case:** Constraining cosmological parameters from peak statistics

### 2. Wavelet L1-Norm
Sum of absolute wavelet coefficient values as a function of SNR threshold, at each scale.

```python
# For each scale j:
#   1. Bin coefficients by their SNR value
#   2. Sum |W_j| in each bin
```

**Use case:** Measuring non-Gaussianity and higher-order statistics

### 3. Mono-Scale Peak Counts
Peak counts in Gaussian-smoothed map (single-scale analysis).

```python
# 1. Apply Gaussian smoothing with σ = smoothing_sigma
# 2. Compute SNR of smoothed map
# 3. Find peaks and create histogram
```

**Use case:** Traditional peak count analysis, comparison with multi-scale

## Key Features

### ✅ GPU Acceleration
All operations are PyTorch-based and run on GPU automatically:

```python
device = torch.device('cuda')  # Use GPU
# OR
device = torch.device('cpu')   # Use CPU
```

### ✅ No C++ Dependencies
Pure Python/PyTorch - no compilation needed!

### ✅ Batch Processing
Efficiently process multiple maps:

```python
for kappa in kappa_maps:
    results = stats.compute_all_statistics(kappa, sigma)
    # Process results...
```

### ✅ Masking Support
Handle survey footprints and missing data:

```python
mask = torch.ones(512, 512)
mask[bad_regions] = 0

results = stats.compute_all_statistics(kappa, sigma, mask=mask)
```

## Common Use Cases

### CFIS/Euclid-like Survey

```python
# CFIS parameters
SHAPE_NOISE = 0.44
PIX_ARCMIN = 0.4
N_GAL = 7  # gal/arcmin²

sigma_noise = SHAPE_NOISE / np.sqrt(2 * N_GAL * PIX_ARCMIN**2)

stats = WLStatistics(n_scales=5, pixel_arcmin=PIX_ARCMIN)
results = stats.compute_all_statistics(kappa_noisy, sigma_noise)
```

### With Survey Mask

```python
# Create mask (1 = observed, 0 = not observed)
mask = create_survey_footprint()

results = stats.compute_all_statistics(
    kappa_map,
    sigma_map,
    mask=mask
)
```

### Multiple Realizations

```python
all_results = []
for kappa_sim in simulations:
    results = stats.compute_all_statistics(kappa_sim, sigma)
    all_results.append(results)

# Compute mean and covariance across realizations
```

## Visualization

```python
from wl_stats_torch.visualization import (
    plot_peak_histograms,
    plot_l1_norms,
    plot_wavelet_scales
)

# Plot peak counts
plot_peak_histograms(
    results['peak_bins'],
    results['wavelet_peak_counts'],
    log_scale=True,
    save_path='peaks.png'
)

# Plot L1-norms
plot_l1_norms(
    results['l1_bins'],
    results['wavelet_l1_norms'],
    save_path='l1norms.png'
)

# Visualize wavelet scales
plot_wavelet_scales(
    results['snr'],
    peak_positions=results['wavelet_peak_positions'],
    mark_peaks=True,
    save_path='scales.png'
)
```

## Examples

Three complete examples are provided in `examples/`:

1. **`basic_usage.py`** - Simple synthetic data example
2. **`cfis_example.py`** - Realistic CFIS-like simulation
3. **`batch_processing.py`** - Efficient processing of multiple maps

Run them:
```bash
cd examples
python basic_usage.py
python cfis_example.py
python batch_processing.py
```

## Performance Tips

### GPU vs CPU
GPU is typically 10-50x faster:

```python
# GPU (recommended)
device = torch.device('cuda')
# ~0.1 seconds per 512x512 map

# CPU (fallback)
device = torch.device('cpu')
# ~2-5 seconds per 512x512 map
```

### Memory Management
For large batches:

```python
torch.cuda.empty_cache()  # Clear GPU cache between batches
```

### Optimize Parameters
```python
# Faster (fewer bins/scales)
results = stats.compute_all_statistics(
    kappa, sigma,
    n_bins=21,          # Fewer bins
    l1_nbins=30,        # Fewer L1 bins
    compute_mono=False  # Skip mono-scale
)
```

## Next Steps

- **API Documentation**: See `API.md` for complete API reference
- **Installation Guide**: See `INSTALL.md` for detailed setup
- **Tests**: Run `pytest tests/` to verify installation
- **Examples**: Explore `examples/` directory

## Citation

If you use this package, please cite:

1. The original CosmoStat package and relevant papers
2. This PyTorch implementation

## Support

For issues, questions, or contributions:
- GitHub Issues: [your-repo-url]
- Email: [your-email]

## Comparison with Original CosmoStat

| Feature | CosmoStat (C++) | wl-stats-torch (PyTorch) |
|---------|----------------|--------------------------|
| Language | Python + C++ | Pure Python + PyTorch |
| Compilation | Required | Not required |
| GPU Support | No | Yes (CUDA) |
| Dependencies | pysparse, C++ libs | PyTorch, NumPy, SciPy |
| Speed (CPU) | Fast | Similar |
| Speed (GPU) | N/A | 10-50x faster |
| Batch Processing | Manual | Built-in |
| Spherical (HEALPix) | Yes | Not yet implemented |

The PyTorch version provides the same core functionality for 2D (flat-sky) weak lensing analysis, with the added benefits of GPU acceleration and no compilation requirements.
