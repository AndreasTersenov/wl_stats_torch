# Developer Guide

This guide covers development workflows, testing, and contributing to wl-stats-torch.

## Development Setup

### Install Development Dependencies

```bash
pip install -e ".[dev]"
```

This installs the package in editable mode along with development tools:
- pytest (testing)
- pytest-cov (coverage reporting)
- black (code formatting)
- isort (import sorting)
- flake8 (linting)
- sphinx (documentation)
- sphinx-rtd-theme (documentation theme)
- sphinx-autodoc-typehints (type hints in docs)

## Development Workflow

### Quick Commands via Makefile

```bash
make help          # Show all available commands
make install-dev   # Install with dev dependencies
make test          # Run tests
make test-cov      # Run tests with coverage report
make lint          # Check code style with flake8
make format        # Format code with black and isort
make format-check  # Check formatting without changing files
make docs          # Build documentation
make clean         # Remove build artifacts
make build         # Build distribution packages
make all           # Run format, lint, test, and docs
```

### Code Formatting

This project uses `black` and `isort` for consistent code formatting.

**Format all code:**
```bash
make format
```

**Check formatting without modifying:**
```bash
make format-check
```

Configuration is in `pyproject.toml`:
- Line length: 100
- isort profile: black (compatible with black)

### Linting

Linting with `flake8` checks for code quality issues.

**Run linting:**
```bash
make lint
```

Configuration is in `.flake8`:
- Compatible with black formatting
- Ignores E203, E501, W503
- Allows F401 in `__init__.py` files

### Testing

Tests use `pytest` with coverage reporting.

**Run all tests:**
```bash
make test
```

**Run tests with coverage:**
```bash
make test-cov
```

**Run specific test file:**
```bash
pytest tests/test_peaks.py
```

**Run specific test:**
```bash
pytest tests/test_peaks.py::TestPeakDetection::test_find_peaks_simple
```

Test configuration is in `pyproject.toml`:
- Coverage reports: HTML and terminal
- Test discovery: `tests/test_*.py`

### Documentation

Documentation is built with Sphinx and uses the Read the Docs theme.

**Build documentation:**
```bash
make docs
```

**View documentation:**
```bash
# Open docs/_build/html/index.html in your browser
firefox docs/_build/html/index.html
```

**Clean documentation build:**
```bash
make docs-clean
```

Documentation source files are in `docs/`:
- `conf.py` - Sphinx configuration
- `index.rst` - Main documentation page
- `installation.rst` - Installation guide
- `quickstart.rst` - Quick start guide
- `api.rst` - API reference
- `examples.rst` - Examples

### Building Distribution Packages

**Build source and wheel distributions:**
```bash
make build
```

This creates:
- `dist/wl-stats-torch-*.tar.gz` (source distribution)
- `dist/wl_stats_torch-*.whl` (wheel distribution)

### Pre-commit Workflow

Before committing code, run:

```bash
make all
```

This runs formatting, linting, tests, and builds documentation.

## Project Structure

```
wl_stats_torch/
├── wl_stats_torch/          # Main package
│   ├── __init__.py          # Package initialization
│   ├── starlet.py           # Wavelet transform
│   ├── peaks.py             # Peak detection
│   ├── statistics.py        # Main statistics class
│   └── visualization.py     # Plotting utilities
├── tests/                   # Test suite
│   ├── conftest.py          # Shared pytest fixtures
│   ├── test_peaks.py        # Peak detection tests
│   ├── test_starlet.py      # Wavelet tests
│   └── test_statistics.py   # Statistics tests
├── docs/                    # Documentation
│   ├── conf.py              # Sphinx configuration
│   ├── *.rst                # Documentation pages
│   └── Makefile             # Documentation build
├── examples/                # Example scripts
├── pyproject.toml           # Package configuration
├── Makefile                 # Development commands
├── MANIFEST.in              # Package data files
└── README.md                # User documentation
```

## Configuration Files

### pyproject.toml
Main configuration file containing:
- Package metadata (name, version, authors, etc.)
- Dependencies
- Development dependencies
- Tool configurations (black, isort, pytest, coverage)

### .flake8
Flake8 linting configuration:
- Line length matching black (100)
- Ignores for black compatibility
- Exclusions for generated/vendor code

### Makefile
Common development commands for easy access.

## Contributing Guidelines

### Code Style
- Use black for formatting (100 char line length)
- Use isort for import sorting
- Follow PEP 8 guidelines
- Add type hints where appropriate
- Write docstrings for all public functions and classes

### Testing
- Write tests for all new features
- Maintain or improve code coverage
- Test on both CPU and GPU when applicable
- Use pytest fixtures for common test setup

### Documentation
- Update relevant documentation for new features
- Include code examples in docstrings
- Add examples to `examples/` directory for complex use cases
- Build documentation to check for errors

### Git Workflow
1. Format and lint code: `make format && make lint`
2. Run tests: `make test`
3. Update documentation if needed: `make docs`
4. Commit with descriptive message
5. Push changes

## Troubleshooting

### Import Errors
If you get import errors after installation:
```bash
pip install -e .  # Reinstall in editable mode
```

### Test Failures
- Check if dependencies are up to date: `pip install -e ".[dev]" --upgrade`
- Clear pytest cache: `rm -rf .pytest_cache`
- Check CUDA availability for GPU tests: `python -c "import torch; print(torch.cuda.is_available())"`

### Documentation Build Errors
- Check Sphinx version: `sphinx-build --version`
- Clean build directory: `make docs-clean && make docs`
- Verify all RST files have proper syntax

## Release Process

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md` (if exists)
3. Run full test suite: `make all`
4. Build distributions: `make build`
5. Tag release: `git tag v0.1.0`
6. Push with tags: `git push --tags`
7. Upload to PyPI: `python -m twine upload dist/*`
