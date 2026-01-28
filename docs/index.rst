wl-stats-torch
==============

GPU-accelerated weak lensing summary statistics using PyTorch.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api
   examples

Overview
--------

**wl-stats-torch** computes weak lensing summary statistics commonly used in cosmological analyses:

* **Mono-scale peak counts** - Peak statistics on smoothed convergence maps
* **Wavelet (Starlet) peak counts** - Multi-scale peak detection using the starlet transform
* **Wavelet L1-norm** - Sparsity measure across wavelet scales

This package provides a fast, pure-Python alternative to the C++-dependent CosmoStat
implementation, with full GPU support via PyTorch.

Key Features
------------

* **Batch Processing** - 12-19x faster than sequential processing on GPU
* **GPU Acceleration** - All operations run on CUDA devices via PyTorch
* **No C++ Dependencies** - Pure Python implementation, no compilation required
* **ML-Ready** - Vectorized operations for gradient-based learning workflows
* **Memory Efficient** - Optimized for large-scale cosmological simulations
* **Docker Support** - Pre-configured containers for CPU and GPU environments

Quick Example
-------------

.. code-block:: python

   import torch
   from wl_stats_torch import WLStatistics

   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   stats = WLStatistics(n_scales=5, device=device)

   # Single image
   kappa_map = torch.randn(512, 512, device=device)
   sigma_map = torch.ones(512, 512, device=device) * 0.01

   results = stats.compute_all_statistics(kappa_map, sigma_map)

   # Batch processing (12-19x faster on GPU)
   kappa_batch = torch.randn(128, 512, 512, device=device)
   results = stats.compute_all_statistics(kappa_batch, 0.01)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
