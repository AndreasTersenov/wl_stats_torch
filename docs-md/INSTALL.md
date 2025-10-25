# Installation Guide

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- NumPy >= 1.20.0
- SciPy >= 1.7.0
- Matplotlib >= 3.3.0

## Installation Methods

### From Source (Recommended)

1. Clone or download the repository:
```bash
cd wl_summary_stats_torch
```

2. Install in development mode:
```bash
pip install -e .
```

3. Or install with development dependencies:
```bash
pip install -e ".[dev]"
```

### GPU Support

The package will automatically use CUDA if available. To verify:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

## Verifying Installation

Run the basic tests:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_starlet.py -v

# Run with coverage
pytest tests/ --cov=wl_stats_torch --cov-report=html
```

Or run a quick example:

```bash
cd examples
python basic_usage.py
```

## Troubleshooting

### CUDA Out of Memory

If you encounter CUDA out of memory errors:

1. Reduce image size or batch size
2. Use CPU instead: `device = torch.device('cpu')`
3. Clear CUDA cache: `torch.cuda.empty_cache()`

### Import Errors

If you get import errors, make sure the package is installed:

```bash
pip show wl-stats-torch
```

If not found, reinstall:

```bash
pip install -e .
```

### Performance Issues

For optimal performance:

1. Use GPU if available
2. Process multiple maps in batches
3. Use appropriate precision (float32 is usually sufficient)

```python
# Example: Process on GPU
device = torch.device('cuda')
stats = WLStatistics(n_scales=5, device=device)
```

## Optional Dependencies

For development and testing:

```bash
pip install pytest pytest-cov black isort flake8
```

For running examples and creating visualizations:

```bash
pip install jupyter ipython
```

## Uninstallation

```bash
pip uninstall wl-stats-torch
```
