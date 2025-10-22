#!/bin/bash

# Quick setup script for wl-stats-torch package
# Usage: bash setup.sh

echo "================================"
echo "WL-Stats-Torch Setup Script"
echo "================================"
echo

# Check Python version
echo "Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check if Python >= 3.8
if python -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo "✓ Python version OK"
else
    echo "✗ Python >= 3.8 required"
    exit 1
fi
echo

# Install package
echo "Installing package..."
pip install -e .
if [ $? -eq 0 ]; then
    echo "✓ Package installed"
else
    echo "✗ Installation failed"
    exit 1
fi
echo

# Install development dependencies (optional)
read -p "Install development dependencies? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing development dependencies..."
    pip install -e ".[dev]"
    echo "✓ Development dependencies installed"
fi
echo

# Verify installation
echo "Verifying installation..."
python verify_installation.py
if [ $? -eq 0 ]; then
    echo
    echo "================================"
    echo "✓ Setup completed successfully!"
    echo "================================"
    echo
    echo "Next steps:"
    echo "  1. Run examples:"
    echo "     cd examples && python basic_usage.py"
    echo
    echo "  2. Run tests:"
    echo "     pytest tests/ -v"
    echo
    echo "  3. Read documentation:"
    echo "     cat QUICKSTART.md"
    echo
else
    echo
    echo "================================"
    echo "⚠️  Setup completed with warnings"
    echo "================================"
    echo
    echo "Please review the errors above."
    echo "The package may still work, but some features might be limited."
    echo
fi
