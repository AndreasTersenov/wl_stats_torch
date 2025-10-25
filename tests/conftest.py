"""
Pytest configuration and shared fixtures.
"""

import pytest
import torch


@pytest.fixture(scope="session")
def device():
    """Get the available device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def use_cuda():
    """Check if CUDA is available."""
    return torch.cuda.is_available()
