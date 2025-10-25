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
- See `PACKAGE_SETUP_COMPLETE.md` for setup summary
- Run `make help` for command list
