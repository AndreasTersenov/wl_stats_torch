# Batch Processing Quick Reference

## TL;DR

```python
# Single image (old way - still works)
results = stats.compute_all_statistics(image, noise_sigma=0.001)

# Batch (new way - automatic!)
results = stats.compute_all_statistics(images, noise_sigma=0.001)
```

That's it! The API detects batch dimension automatically.

## Input Shapes

| Parameter | Single | Batch |
|-----------|--------|-------|
| `image` | `(H, W)` | `(B, H, W)` |
| `noise_sigma` | scalar or `(H, W)` | scalar, `(H, W)`, or `(B, H, W)` |
| `mask` | `None` or `(H, W)` | `None`, `(H, W)`, or `(B, H, W)` |

## Output Shapes

| Result | Single | Batch |
|--------|--------|-------|
| `wavelet_coeffs` | `(n_scales, H, W)` | `(B, n_scales, H, W)` |
| `wavelet_peak_counts[i]` | `(n_bins,)` | `(B, n_bins)` |
| `wavelet_l1_norms[i]` | `(l1_nbins,)` | `(B, l1_nbins)` |

## Common Patterns

### ML Training
```python
# In training loop
for batch_images, labels in dataloader:
    results = stats.compute_all_statistics(batch_images, 0.001)
    features = extract_features(results)  # (B, 906)
    outputs = model(features)
```

### Extract Features
```python
# Get batched features
peaks = torch.stack(results['wavelet_peak_counts'])  # (n_scales, B, n_bins)
l1 = torch.stack(results['wavelet_l1_norms'])       # (n_scales, B, l1_nbins)

# Reshape to (B, features)
features = torch.cat([
    peaks.permute(1, 0, 2).flatten(1),
    l1.permute(1, 0, 2).flatten(1)
], dim=1)
```

### Noise Formats
```python
# Scalar (same for all)
results = stats.compute_all_statistics(images, noise_sigma=0.001)

# Shared map (H, W)
noise_map = torch.ones(256, 128) * 0.001
results = stats.compute_all_statistics(images, noise_sigma=noise_map)

# Per-sample (B, H, W)
noise_batch = torch.rand(B, 256, 128) * 0.002
results = stats.compute_all_statistics(images, noise_sigma=noise_batch)
```

### Mask Formats
```python
# No mask
results = stats.compute_all_statistics(images, 0.001, mask=None)

# Shared mask (H, W)
mask = torch.ones(256, 128)
mask[:10, :] = 0
results = stats.compute_all_statistics(images, 0.001, mask=mask)

# Per-sample mask (B, H, W)
masks = torch.ones(B, 256, 128)
for i in range(B):
    masks[i, :10+i, :] = 0
results = stats.compute_all_statistics(images, 0.001, mask=masks)
```

## Typical Batch Sizes

| Use Case | Batch Size | Memory (~256x128) |
|----------|-----------|-------------------|
| Development | 4-8 | 100-200 MB |
| Training | 32-64 | 1-2 GB |
| Production | 128+ | 3-5 GB |

## Documentation

- 📘 Full Guide: `docs-md/BATCH_PROCESSING.md`
- 📖 API Reference: `docs-md/API.md`
- 🎯 Examples: `examples/batch_processing_demo.py`
- ✅ Tests: `test_batch_processing.py`

## Migration

**No changes needed!** Your existing code works as-is:

```python
# This still works exactly the same
image = torch.randn(256, 128)
results = stats.compute_all_statistics(image, 0.001)
```

**To use batching, just pass `(B, H, W)` instead:**

```python
# Now you can do this too
images = torch.randn(128, 256, 128)
results = stats.compute_all_statistics(images, 0.001)
```

## Common Issues

### OOM Error
```python
# Problem: Batch too large
images = torch.randn(1000, 256, 128)  # Too big!

# Solution: Smaller batches
for i in range(0, len(images), 64):
    batch = images[i:i+64]
    results = stats.compute_all_statistics(batch, 0.001)
```

### Shape Mismatch
```python
# Problem: Wrong batch size
images = torch.randn(64, 256, 128)
noise = torch.randn(32, 256, 128)  # Mismatch!

# Solution: Match dimensions
noise = 0.001  # Scalar works for any batch size
# OR
noise = torch.randn(256, 128)  # Shared works for any batch size
# OR  
noise = torch.randn(64, 256, 128)  # Must match batch size
```

## Performance Tips

1. **Batch size**: Sweet spot is 32-128 for typical maps
2. **Disable mono**: Set `compute_mono=False` if not needed
3. **Pre-allocate**: Create tensors on GPU directly
4. **Use appropriate dtype**: `float32` vs `float64`

```python
# Optimized setup
stats = WLStatistics(n_scales=6, device='cuda', dtype=torch.float32)
images = torch.randn(64, 256, 128, device='cuda', dtype=torch.float32)

results = stats.compute_all_statistics(
    images,
    noise_sigma=0.001,
    compute_mono=False,  # Skip if not needed
    verbose=False
)
```

## Questions?

See full documentation in `docs-md/BATCH_PROCESSING.md`
