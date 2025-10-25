Weak Lensing Statistics with PyTorch
=====================================

A GPU-accelerated PyTorch implementation for computing weak lensing summary statistics.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api
   examples

Features
--------

* **GPU Acceleration**: All operations are PyTorch-based and can run on CUDA devices
* **No C++ Dependencies**: Pure Python implementation, no compilation required
* **Batch Processing**: Efficiently process multiple maps simultaneously
* **Memory Efficient**: Optimized for large-scale cosmological simulations

Summary Statistics
------------------

This package computes three key weak lensing summary statistics:

* **Mono-scale peak counts**: Peak detection on smoothed convergence maps
* **Wavelet peak counts**: Multi-scale peak detection using Starlet wavelet decomposition
* **Wavelet L1-norm**: L1-norm of wavelet coefficients at different scales

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
