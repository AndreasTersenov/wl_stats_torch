# dtype/device Consistency Fixes in peaks.py

## Summary

Fixed several minor dtype and device consistency issues in `peaks.py` to ensure proper type propagation throughout the pipeline.

## Issues Fixed

### 1. Empty histogram return dtype
**Problem:**
```python
return torch.zeros(len(bins) - 1, device=bins.device, dtype=torch.float32)
```
Hardcoded `float32` wouldn't match bins dtype (could be float64 elsewhere).

**Solution:**
```python
return torch.zeros(len(bins) - 1, device=bins.device, dtype=bins.dtype)
```
Now matches the bins dtype for consistency.

### 2. dtype alignment in searchsorted
**Problem:**
```python
bins = bins.to(peak_heights.device)
```
Only aligned device, not dtype. `torch.searchsorted` requires matching dtypes.

**Solution:**
```python
bins = bins.to(peak_heights.device, dtype=peak_heights.dtype)
```
Now aligns both device and dtype before searchsorted operation.

### 3. Final return dtype in peaks_to_histogram
**Problem:**
```python
return counts[:n_bins].float()
```
`.float()` always returns float32, ignoring bins dtype.

**Solution:**
```python
return counts[:n_bins].to(dtype=bins.dtype)
```
Maintains dtype consistency with input bins.

### 4. Redundant conditional in mono_scale_peaks_smoothed
**Problem:**
```python
image_dtype = image.dtype if image.ndim == 4 else image.dtype
```
Both branches return the same thing - redundant.

**Solution:**
```python
image_dtype = image.dtype
```
Simplified to single assignment after image is already 4D.

### 5. Bins creation dtype
**Problem:**
```python
bins = torch.linspace(min_snr, max_snr, n_bins + 1, device=device)
```
Defaults to float32, doesn't match image dtype.

**Solution:**
```python
bins = torch.linspace(min_snr, max_snr, n_bins + 1, device=device, dtype=image_dtype)
```
Uses image dtype for consistency.

### 6. Variance map dtype
**Problem:**
```python
variance_map = (sigma_noise_map ** 2).unsqueeze(0).unsqueeze(0)
```
Might have different dtype than Gaussian kernel.

**Solution:**
```python
variance_map = (sigma_noise_map ** 2).unsqueeze(0).unsqueeze(0).to(dtype=image_dtype)
```
Ensures consistent dtype with image and kernels.

## Benefits

1. **Type safety**: All operations maintain consistent dtypes throughout
2. **Flexibility**: Works correctly with both float32 and float64
3. **GPU efficiency**: Proper dtype matching avoids implicit conversions
4. **Debugging**: Easier to track data types through the pipeline
5. **Correctness**: Avoids subtle precision issues from mixed dtypes

## Testing

Created `test_dtype_consistency.py` with comprehensive tests:
- ✓ Empty histogram returns correct dtype
- ✓ Non-empty histograms maintain dtype
- ✓ Mixed dtypes align correctly
- ✓ mono_scale_peaks_smoothed maintains float64
- ✓ mono_scale_peaks_smoothed maintains float32
- ✓ Spatially-varying noise maps work
- ✓ find_peaks_2d maintains input dtype

## Backward Compatibility

All changes are backward compatible:
- Default behavior unchanged (still uses float64 from WLStatistics)
- Existing code continues to work
- No API changes
- Tests pass

## Files Modified

- `wl_stats_torch/peaks.py`:
  - Line ~211: Empty histogram return dtype
  - Line ~213: Device and dtype alignment
  - Line ~268: Final return dtype
  - Line ~320: Simplified image_dtype assignment
  - Line ~395: Bins creation with dtype
  - Line ~369: Variance map dtype alignment

## Date

January 2025
