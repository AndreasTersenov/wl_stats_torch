# Signal-to-Noise Calculation Fix

## Problem
The `wl_stats_torch` package was producing very different signal-to-noise ratio (SNR) results compared to the reference `cosmostat` package for the same weak lensing statistics (monoscale peak counts, multiscale peak counts, wavelet l1-norm). While the starlet transform itself produced identical results, the noise level calculation was completely different, leading to incorrect SNR maps with the wrong scale.

## Root Cause
The original implementation in `wl_stats_torch/starlet.py` attempted to compute noise levels by:
1. Squaring the convolution kernel
2. Convolving the variance map with the squared kernel using dilated convolution

This approach was **fundamentally incorrect**. The `cosmostat` implementation uses a different approach:
1. Create a delta function (impulse) at the center of an image
2. Apply the wavelet transform to get the **impulse response** at each scale
3. **Square the wavelet coefficients** (not the kernel)
4. Convolve the variance map with these squared impulse responses using FFT convolution

The key difference: cosmostat computes the impulse response of the **entire wavelet transform**, not just the convolution kernel. This correctly accounts for how noise propagates through the multi-scale decomposition.

## Solution
Rewrote the `get_noise_levels()` method in `wl_stats_torch/starlet.py` to match the cosmostat approach:

```python
def get_noise_levels(self, noise_sigma, mask=None):
    # 1. Create impulse (delta function) at center
    impulse = torch.zeros(1, 1, height, width, ...)
    impulse[0, 0, height // 2, width // 2] = 1.0
    
    # 2. Get wavelet impulse response
    impulse_coeffs = self.forward(impulse, return_coarse=True)
    
    # 3. Square the impulse response coefficients
    impulse_coeffs_squared = impulse_coeffs ** 2
    
    # 4. Convolve variance map with squared impulse response for each scale
    variance_map = noise_sigma ** 2
    for scale_idx in range(self.n_scales):
        kernel = impulse_coeffs_squared[0, scale_idx, :, :]
        var_scale = scipy_signal.fftconvolve(variance_map, kernel, mode='same')
        # ... convert to torch and accumulate
    
    # 5. Take square root to get noise standard deviations
    noise_levels = torch.sqrt(variance_all)
```

## Verification
Test results show **perfect agreement** (0.00% difference) between noise levels computed by `wl_stats_torch` and `cosmostat`:

```
Noise Levels Comparison:
  Scale 0: Max rel diff = 0.00%
  Scale 1: Max rel diff = 0.00%
  Scale 2: Max rel diff = 0.00%
  Scale 3: Max rel diff = 0.00%
  Scale 4: Max rel diff = 0.00%
```

The computed noise standard deviations at each scale now match exactly:
```
CosmoStat:        wl_stats_torch:
Scale 0: 0.004681  Scale 0: 0.004681
Scale 1: 0.000664  Scale 1: 0.000664
Scale 2: 0.000242  Scale 2: 0.000242
Scale 3: 0.000130  Scale 3: 0.000130
Scale 4: 0.000123  Scale 4: 0.000123
```

## Impact
This fix ensures that:
1. **Signal-to-noise maps** now have the correct scale and values
2. **Peak detection** will identify the same peaks at correct significance levels
3. **Multiscale peak counts** will match those from cosmostat
4. **L1-norm statistics** will be computed correctly
5. Results from `wl_stats_torch` are now **directly comparable** with existing cosmostat-based analyses

## Files Modified
- `wl_stats_torch/starlet.py`: Rewrote `get_noise_levels()` method
- Added `import numpy as np` and `from scipy import signal as scipy_signal`

## Testing
Run the verification test:
```bash
python test_cosmostat_comparison.py
```

This confirms that noise levels match perfectly between implementations.
