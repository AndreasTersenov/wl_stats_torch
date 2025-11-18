# API Documentation

## Core Classes

### WLStatistics

Main class for computing weak lensing summary statistics.

```python
from wl_stats_torch import WLStatistics

stats = WLStatistics(
    n_scales=5,              # Number of wavelet scales
    device=None,             # torch.device or None for auto-detect
    pixel_arcmin=1.0,        # Pixel resolution in arcminutes
    dtype=torch.float64      # Data type for computations (default: float64)
)
```

#### Methods

##### `compute_all_statistics`

Compute all statistics in one call. **Supports optimized batch processing!**

🚀 **Performance**: Batch processing delivers **12-19x speedup** over sequential processing on GPU (validated on NVIDIA A100 with 256×256 images, batch size 4).

```python
results = stats.compute_all_statistics(
    image,                    # Convergence map (H, W) or batch (B, H, W)
    noise_sigma,             # Noise: scalar, (H, W), or (B, H, W)
    mask=None,               # Optional mask: None, (H, W), or (B, H, W)
    min_snr=-2.0,           # Minimum SNR for histograms
    max_snr=6.0,            # Maximum SNR for histograms
    n_bins=31,              # Number of bins for peak histograms
    l1_nbins=40,            # Number of bins for L1-norm
    l1_min_snr=None,        # Minimum SNR for L1-norm (uses min_snr if None)
    l1_max_snr=None,        # Maximum SNR for L1-norm (uses max_snr if None)
    compute_mono=True,       # Whether to compute mono-scale peaks
    mono_smoothing_sigma=2.0, # Smoothing for mono-scale
    verbose=False,           # Print progress
    clamp_overflow=False     # Include out-of-range values in edge bins
)
```

**💡 See [BATCH_PROCESSING.md](BATCH_PROCESSING.md) for complete batch processing guide and [BATCH_OPTIMIZATION.md](BATCH_OPTIMIZATION.md) for technical optimization details.**

**Parameters:**
- `image`: Input convergence map(s)
  - Single: `(H, W)` 
  - Batch: `(B, H, W)`
- `noise_sigma`: Noise standard deviation
  - Scalar: same for all pixels/images
  - Map `(H, W)`: same pattern for all images in batch
  - Batch map `(B, H, W)`: different per image
- `mask`: Observation mask
  - `None`: no masking
  - Map `(H, W)`: same mask for all images in batch
  - Batch map `(B, H, W)`: different per image

**Returns:** Dictionary with keys (shapes depend on input):

*Single image input (H, W):*
- `wavelet_coeffs`: `(n_scales, H, W)`
- `noise_levels`: `(n_scales, H, W)`
- `snr`: `(n_scales, H, W)`
- `peak_bins`: `(n_bins,)`
- `wavelet_peak_counts`: List of `(n_bins,)` per scale
- `wavelet_peak_positions`: List of peak positions per scale
- `wavelet_peak_heights`: List of peak heights per scale
- `l1_bins`: List of `(l1_nbins,)` per scale
- `wavelet_l1_norms`: List of `(l1_nbins,)` per scale
- `mono_peak_bins`: `(n_bins,)` (if `compute_mono=True`)
- `mono_peak_counts`: `(n_bins,)` (if `compute_mono=True`)

*Batch input (B, H, W):*
- `wavelet_coeffs`: `(B, n_scales, H, W)`
- `noise_levels`: `(B, n_scales, H, W)`
- `snr`: `(B, n_scales, H, W)`
- `peak_bins`: `(n_bins,)`
- `wavelet_peak_counts`: List of `(B, n_bins)` per scale
- `wavelet_peak_positions`: List of lists (per scale, per batch item)
- `wavelet_peak_heights`: List of lists (per scale, per batch item)
- `l1_bins`: List of `(l1_nbins,)` per scale
- `wavelet_l1_norms`: List of `(B, l1_nbins)` per scale
- `mono_peak_bins`: `(n_bins,)` (if `compute_mono=True`)
- `mono_peak_counts`: `(B, n_bins)` (if `compute_mono=True`)

**Example (Batch Processing):**
```python
# Process 128 convergence maps at once
images = torch.randn(128, 256, 128, device='cuda')
stats = WLStatistics(n_scales=6, device='cuda')
results = stats.compute_all_statistics(images, noise_sigma=0.001)

# Extract batched features for ML training
wavelet_peaks = torch.stack(results['wavelet_peak_counts'])  # (6, 128, 31)
features = wavelet_peaks.permute(1, 0, 2).flatten(1)  # (128, 186)
```

##### `compute_wavelet_transform`

Compute wavelet transform and SNR. **Supports batch processing!**

```python
results = stats.compute_wavelet_transform(
    image,           # Input map (H, W) or batch (B, H, W)
    noise_sigma,    # Noise: scalar, (H, W), or (B, H, W)
    mask=None       # Optional mask: None, (H, W), or (B, H, W)
)
```

**Returns:** Dictionary with:
- Single input: `wavelet_coeffs`, `noise_levels`, `snr` as `(n_scales, H, W)`
- Batch input: `wavelet_coeffs`, `noise_levels`, `snr` as `(B, n_scales, H, W)`

##### `compute_wavelet_peak_counts`

Compute peak count histograms at all scales. **Supports batch processing!**

```python
bin_centers, peak_counts = stats.compute_wavelet_peak_counts(
    min_snr=-2.0,
    max_snr=6.0,
    n_bins=31,
    mask=None,            # None, (H, W), or (B, H, W)
    verbose=False,
    clamp_overflow=False  # Include out-of-range peaks in edge bins
)
```

**Returns:**
- `bin_centers`: `(n_bins,)`
- `peak_counts`: List of tensors per scale
  - Single image: `(n_bins,)` per scale
  - Batch: `(B, n_bins)` per scale

##### `compute_wavelet_l1_norms`

Compute L1-norms as function of SNR. **Supports batch processing!**

```python
bins_list, l1_norms_list = stats.compute_wavelet_l1_norms(
    n_bins=40,
    mask=None,            # None, (H, W), or (B, H, W)
    min_snr=None,
    max_snr=None,
    clamp_overflow=False  # Include out-of-range values in edge bins
)
```

**Returns:**
- `bins_list`: List of `(n_bins,)` per scale
- `l1_norms_list`: List of tensors per scale
  - Single image: `(n_bins,)` per scale
  - Batch: `(B, n_bins)` per scale

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
    mask=None,
    clamp_overflow=False  # Include out-of-range peaks in edge bins
)
```

### Starlet2D

2D Starlet (à trous wavelet) transform.

```python
from wl_stats_torch.starlet import Starlet2D

starlet = Starlet2D(
    n_scales=5,     # Total number of scales (including coarse)
    device=None,    # torch.device or None
    dtype=torch.float32  # Data type for computations (default: float32)
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

##### `get_scale_resolution`

Get the effective resolution (FWHM) of each wavelet scale in arcminutes.

```python
resolutions = starlet.get_scale_resolution(
    pixel_size_arcmin=1.0  # Size of one pixel in arcminutes
)
```

**Returns:** List of resolutions for each scale (including coarse)

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

### peaks_to_histogram

Compute histogram of peak heights.

```python
from wl_stats_torch.peaks import peaks_to_histogram

counts = peaks_to_histogram(
    peak_heights,        # Tensor of peak values (N,)
    bins,               # Bin edges (n_bins+1,)
    digitize_mode=True, # Use np.digitize-like behavior (default)
    clamp_overflow=False # Include out-of-range values in edge bins
)
```

**Returns:** Histogram counts, shape (n_bins,)

**Note:** When `clamp_overflow=False` (default), values outside the bin range are excluded, matching CosmoStat/pycs behavior. When `clamp_overflow=True`, out-of-range values are included in the edge bins.

## Utility Functions

### fft_convolve2d

Perform 2D convolution using FFT (equivalent to scipy.signal.fftconvolve with mode='same').

```python
from wl_stats_torch.starlet import fft_convolve2d

result = fft_convolve2d(
    signal,  # Input signal (H, W)
    kernel   # Convolution kernel (H, W)
)
```

**Returns:** Convolved result, shape (H, W)

**Note:** This function stays entirely in PyTorch, avoiding CPU transfers and numpy conversions. It matches scipy's behavior for 'same' mode convolution and is used internally by the Starlet transform.

### mono_scale_peaks_smoothed

Compute mono-scale peaks with Gaussian smoothing.

```python
from wl_stats_torch.peaks import mono_scale_peaks_smoothed

```python
bin_centers, counts, (positions, heights) = mono_scale_peaks_smoothed(
    image,
    sigma_noise,
    smoothing_sigma=2.0,
    mask=None,
    bins=None,
    min_snr=-2.0,
    max_snr=6.0,
    n_bins=31,
    clamp_overflow=False  # Include out-of-range peaks in edge bins
)
```
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

### plot_comparison

Compare the same statistic across multiple result sets.

```python
from wl_stats_torch.visualization import plot_comparison

plot_comparison(
    results_list,        # List of result dictionaries
    labels,             # Labels for each result set
    statistic='wavelet_peak_counts',  # Statistic to compare
    scale_idx=0,        # Which scale to plot
    title=None,
    log_scale=True,
    figsize=(10, 6),
    save_path=None
)
```

**Supported statistics:**
- `'wavelet_peak_counts'`: Compare peak count histograms
- `'wavelet_l1_norms'`: Compare L1-norm curves
- `'mono_peak_counts'`: Compare mono-scale peak counts

## Data Types

All functions accept and return PyTorch tensors. Common shapes:

- **Image**: `(H, W)` or `(B, 1, H, W)`
- **Wavelet coefficients**: `(n_scales, H, W)` or `(B, n_scales, H, W)`
- **Peak positions**: `(N, 2)` where N is number of peaks
- **Peak heights**: `(N,)`
- **Histograms**: `(n_bins,)`

### Important Notes

**`clamp_overflow` Parameter:**
Many histogram and binning functions include a `clamp_overflow` parameter (default: `False`):
- When `False`: Values outside the specified SNR range are excluded from histograms. This matches CosmoStat/pycs reference implementation behavior.
- When `True`: Values below `min_snr` are included in the first bin, and values above `max_snr` are included in the last bin.

**Data Type Defaults:**
- `Starlet2D` uses `torch.float32` by default (optimized for GPU performance)
- `WLStatistics` uses `torch.float64` by default (matches NumPy/CosmoStat precision)
- Both can be overridden via the `dtype` parameter

## Device Handling

All operations respect the device of input tensors. For GPU acceleration:

```python
device = torch.device('cuda')
image = torch.randn(512, 512, device=device)
stats = WLStatistics(n_scales=5, device=device)
results = stats.compute_all_statistics(image, 0.01)
```
