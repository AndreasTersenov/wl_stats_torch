# Quick Reference

## Installation
```bash
# User installation
pip install -e .

# Developer installation (with all tools)
pip install -e ".[dev]"
```

## Development Commands
```bash
make help          # Show all commands
make format        # Format code (black + isort)
make lint          # Check code quality (flake8)
make test          # Run tests (pytest)
make docs          # Build documentation
make all           # Run format, lint, test, and docs
make clean         # Remove build artifacts
make build         # Build distribution packages
```

## Code Quality
```bash
# Format code automatically
make format

# Check linting
make lint

# Run tests with coverage
make test-cov
```

## Documentation
```bash
# Build HTML documentation
make docs

# View documentation
firefox docs/_build/html/index.html

# Clean documentation
make docs-clean
```

## Testing
```bash
# All tests
pytest

# Specific file
pytest tests/test_peaks.py

# Specific test
pytest tests/test_peaks.py::TestPeakDetection::test_find_peaks_simple

# With coverage
pytest --cov=wl_stats_torch
```

## Project Structure
```
wl_stats_torch/
├── wl_stats_torch/      # Source code
├── tests/               # Test suite
├── docs/                # Documentation
├── examples/            # Example scripts
├── pyproject.toml       # Package configuration
├── Makefile             # Development commands
└── CONTRIBUTING.md      # Developer guide
```

## Key Files
- **pyproject.toml** - Package metadata and tool configuration
- **.flake8** - Linting configuration
- **Makefile** - Common development commands
- **MANIFEST.in** - Files to include in distribution
- **tests/conftest.py** - Shared pytest fixtures

## Configuration
- **Line length**: 100 characters (black, flake8)
- **Import sorting**: black profile (isort)
- **Test coverage**: HTML + terminal reports
- **Documentation theme**: Read the Docs

## Batch Processing Quick Start

### Basic Usage
```python
from wl_stats_torch import WLStatistics
import torch

device = torch.device('cuda')
stats = WLStatistics(n_scales=6, device=device)

# Batch of convergence maps
images = torch.randn(128, 512, 512, device=device)  # (B, H, W)
results = stats.compute_all_statistics(images, noise_sigma=0.01)

# 12-19x faster than sequential! 🚀
```

### Performance Tips
```python
# ✅ Good batch sizes for 256×256 maps
batch_size = 4-32

# ✅ Disable mono-scale if not needed
results = stats.compute_all_statistics(images, noise, compute_mono=False)

# ✅ Use GPU
device = torch.device('cuda')  # Not 'cpu'!
```

### Validation
```bash
# Run correctness and performance validation
python validate_optimizations.py

# Expected: ✅ 10-20x speedup on GPU
```

### Documentation
- **Complete Guide**: `docs-md/BATCH_PROCESSING.md`
- **Optimization Details**: `docs-md/BATCH_OPTIMIZATION.md`
- **Examples**: `examples/batch_processing.py`

## Before Committing
```bash
make all  # Format, lint, test, and build docs
```

## Building a Release
```bash
make build  # Creates dist/ with source and wheel
```

## Help
- See `CONTRIBUTING.md` for detailed developer guide
- See `BATCH_PROCESSING.md` for batch processing guide
- See `BATCH_OPTIMIZATION.md` for optimization details
- Run `make help` for command list
