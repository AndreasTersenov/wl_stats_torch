.PHONY: help install install-dev test lint format clean docs build

help:
	@echo "Available commands:"
	@echo "  make install      - Install the package"
	@echo "  make install-dev  - Install with development dependencies"
	@echo "  make test         - Run tests with pytest"
	@echo "  make lint         - Run linting with flake8"
	@echo "  make format       - Format code with black and isort"
	@echo "  make format-check - Check code formatting without modifying"
	@echo "  make docs         - Build Sphinx documentation"
	@echo "  make clean        - Remove build artifacts"
	@echo "  make build        - Build distribution packages"
	@echo "  make all          - Run format, lint, test, and docs"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	pytest

test-cov:
	pytest --cov=wl_stats_torch --cov-report=html --cov-report=term-missing

lint:
	flake8 wl_stats_torch tests

format:
	isort wl_stats_torch tests examples
	black wl_stats_torch tests examples

format-check:
	isort --check-only wl_stats_torch tests examples
	black --check wl_stats_torch tests examples

docs:
	cd docs && $(MAKE) html

docs-clean:
	cd docs && $(MAKE) clean

clean: docs-clean
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

build: clean
	python -m build

all: format lint test docs
	@echo "All checks passed!"
