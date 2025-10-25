# API Documentation

## Core Classes

### WLStatistics

Main class for computing weak lensing summary statistics.

```python
from wl_stats_torch import WLStatistics

stats = WLStatistics(
    n_scales=5,              # Number of wavelet scales
    device=None,             # torch.device or None for auto-detect
    pixel_arcmin=1.0         # Pixel resolution in arcminutes
)
```

#### Methods

##### `compute_all_statistics`

Compute all statistics in one call.

```python
results = stats.compute_all_statistics(
    image,                    # Convergence map (H, W)
    noise_sigma,             # Noise std map (H, W) or scalar
    mask=None,               # Optional observation mask (H, W)
    min_snr=-2.0,           # Minimum SNR for histograms
    max_snr=6.0,            # Maximum SNR for histograms
    n_bins=31,              # Number of bins for peak histograms
    l1_nbins=40,            # Number of bins for L1-norm
    compute_mono=True,       # Whether to compute mono-scale peaks
    mono_smoothing_sigma=2.0, # Smoothing for mono-scale
    verbose=False            # Print progress
)
```

**Returns:** Dictionary with keys:
- `wavelet_coeffs`: Wavelet coefficients (n_scales, H, W)
- `noise_levels`: Noise std for each coefficient (n_scales, H, W)
- `snr`: Signal-to-noise ratio (n_scales, H, W)
- `peak_bins`: Bin centers for peak histograms
- `wavelet_peak_counts`: List of peak count histograms per scale
- `wavelet_peak_positions`: List of peak positions per scale
- `wavelet_peak_heights`: List of peak heights per scale
- `l1_bins`: List of bin centers for L1-norms per scale
- `wavelet_l1_norms`: List of L1-norms per scale
- `mono_peak_bins`: Bin centers for mono-scale peaks
- `mono_peak_counts`: Mono-scale peak counts

##### `compute_wavelet_transform`

Compute wavelet transform and SNR.

```python
results = stats.compute_wavelet_transform(
    image,           # Input map (H, W)
    noise_sigma,    # Noise std map (H, W)
    mask=None       # Optional mask (H, W)
)
```

##### `compute_wavelet_peak_counts`

Compute peak count histograms at all scales.

```python
bin_centers, peak_counts = stats.compute_wavelet_peak_counts(
    min_snr=-2.0,
    max_snr=6.0,
    n_bins=31,
    mask=None,
    verbose=False
)
```

##### `compute_wavelet_l1_norms`

Compute L1-norms as function of SNR.

```python
bins_list, l1_norms_list = stats.compute_wavelet_l1_norms(
    n_bins=40,
    mask=None,
    min_snr=None,
    max_snr=None
)
```

##### `compute_mono_scale_peaks`

Compute mono-scale peak counts with Gaussian smoothing.

```python
bin_centers, counts = stats.compute_mono_scale_peaks(
    image,
    noise_sigma,
    smoothing_sigma=2.0,
    min_snr=-2.0,
    max_snr=6.0,
    n_bins=31,
    mask=None
)
```

### Starlet2D

2D Starlet (à trous wavelet) transform.

```python
from wl_stats_torch.starlet import Starlet2D

starlet = Starlet2D(
    n_scales=5,     # Total number of scales (including coarse)
    device=None     # torch.device or None
)
```

#### Methods

##### `forward`

Apply starlet transform.

```python
coeffs = starlet(
    x,                   # Input tensor (H, W) or (B, 1, H, W)
    return_coarse=True,  # Include coarse scale
    return_dict=False    # Return dict with additional info
)
```

**Returns:** Wavelet coefficients (B, n_scales, H, W)

##### `reconstruct`

Reconstruct image from coefficients.

```python
reconstructed = starlet.reconstruct(
    wavelet_coeffs,  # Coefficients (B, n_scales, H, W)
    gen2=True        # Use second generation reconstruction
)
```

##### `get_noise_levels`

Propagate noise through transform.

```python
noise_levels = starlet.get_noise_levels(
    noise_sigma,  # Noise std map (H, W)
    mask=None     # Optional mask
)
```

##### `get_snr`

Compute SNR for wavelet coefficients.

```python
snr = starlet.get_snr(
    image,           # Input image (H, W)
    noise_sigma,    # Noise std map
    mask=None,       # Optional mask
    keep_sign=False  # Preserve coefficient sign
)
```

## Peak Detection Functions

### find_peaks_2d

Find local maxima in 2D image.

```python
from wl_stats_torch.peaks import find_peaks_2d

positions, heights = find_peaks_2d(
    image,                # 2D tensor (H, W)
    threshold=None,       # Minimum peak value
    mask=None,           # Optional mask
    include_border=False, # Include border peaks
    ordered=True         # Sort by height
)
```

**Returns:**
- positions: Tensor (N, 2) with (row, col) coordinates
- heights: Tensor (N,) with peak values

### find_peaks_batch

Find peaks in batch of images.

```python
from wl_stats_torch.peaks import find_peaks_batch

results = find_peaks_batch(
    images,          # Tensor (B, 1, H, W)
    threshold=None,
    masks=None,
    include_border=False,
    ordered=True
)
```

**Returns:** List of (positions, heights) tuples

### mono_scale_peaks_smoothed

Compute mono-scale peaks with Gaussian smoothing.

```python
from wl_stats_torch.peaks import mono_scale_peaks_smoothed

bin_centers, counts, (positions, heights) = mono_scale_peaks_smoothed(
    image,
    sigma_noise,
    smoothing_sigma=2.0,
    mask=None,
    bins=None,
    min_snr=-2.0,
    max_snr=6.0,
    n_bins=31
)
```

## Visualization Functions

### plot_peak_histograms

Plot peak count histograms for multiple scales.

```python
from wl_stats_torch.visualization import plot_peak_histograms

plot_peak_histograms(
    bin_centers,         # Bin centers
    peak_counts,        # List of counts per scale
    scale_labels=None,   # Optional scale labels
    title="Wavelet Peak Counts",
    xlabel="SNR",
    ylabel="Peak Counts",
    log_scale=True,
    figsize=(10, 6),
    save_path=None      # Save to file
)
```

### plot_l1_norms

Plot L1-norms for multiple scales.

```python
from wl_stats_torch.visualization import plot_l1_norms

plot_l1_norms(
    l1_bins,            # List of bin centers per scale
    l1_norms,          # List of L1-norms per scale
    scale_labels=None,
    title="Wavelet L1-Norms",
    xlabel="SNR",
    ylabel="L1-Norm",
    log_scale=False,
    xlim=None,
    figsize=(10, 6),
    save_path=None
)
```

### plot_wavelet_scales

Visualize wavelet scales with optional peak markers.

```python
from wl_stats_torch.visualization import plot_wavelet_scales

plot_wavelet_scales(
    wavelet_coeffs,      # Coefficients (n_scales, H, W)
    peak_positions=None,  # Optional peak positions
    titles=None,
    cmap='viridis',
    vmin=None,
    vmax=None,
    figsize=(15, 10),
    mark_peaks=True,
    save_path=None
)
```

### plot_snr_map

Plot SNR map for specific scale.

```python
from wl_stats_torch.visualization import plot_snr_map

plot_snr_map(
    snr_coeffs,         # SNR coefficients (n_scales, H, W)
    scale_idx=0,        # Which scale to plot
    peak_positions=None,
    title=None,
    cmap='RdBu_r',
    vmin=-5,
    vmax=5,
    figsize=(10, 8),
    save_path=None
)
```

## Data Types

All functions accept and return PyTorch tensors. Common shapes:

- **Image**: `(H, W)` or `(B, 1, H, W)`
- **Wavelet coefficients**: `(n_scales, H, W)` or `(B, n_scales, H, W)`
- **Peak positions**: `(N, 2)` where N is number of peaks
- **Peak heights**: `(N,)`
- **Histograms**: `(n_bins,)`

## Device Handling

All operations respect the device of input tensors. For GPU acceleration:

```python
device = torch.device('cuda')
image = torch.randn(512, 512, device=device)
stats = WLStatistics(n_scales=5, device=device)
results = stats.compute_all_statistics(image, 0.01)
```
