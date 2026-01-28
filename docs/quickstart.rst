Quick Start
===========

Basic Usage
-----------

Here's a simple example to compute all summary statistics:

.. code-block:: python

   import torch
   from wl_stats_torch import WLStatistics

   # Initialize with device
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   stats = WLStatistics(n_scales=5, device=device)

   # Create example convergence and noise maps
   kappa_map = torch.randn(512, 512, device=device)
   sigma_map = torch.ones(512, 512, device=device) * 0.01

   # Compute all statistics
   results = stats.compute_all_statistics(
       kappa_map,
       sigma_map,
       min_snr=-2,
       max_snr=6,
       n_bins=31
   )

   # Access results
   peak_counts = results['wavelet_peak_counts']  # List of per-scale peak counts
   l1_norms = results['wavelet_l1_norms']        # List of per-scale L1-norms
   mono_peaks = results['mono_peak_counts']      # Mono-scale peak counts

Batch Processing
----------------

Process multiple convergence maps simultaneously for significant speedups (12-19x on GPU):

.. code-block:: python

   import torch
   from wl_stats_torch import WLStatistics

   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   stats = WLStatistics(n_scales=6, device=device)

   # Batch of 128 convergence maps (batch_size, height, width)
   kappa_batch = torch.randn(128, 512, 512, device=device)
   noise_sigma = 0.01  # Can be scalar or tensor

   # Compute statistics for entire batch at once
   results = stats.compute_all_statistics(
       kappa_batch,
       noise_sigma,
       min_snr=-4.0,
       max_snr=15.0,
       n_bins=51,
       l1_nbins=100,
       compute_mono=False  # Skip mono-scale peaks for speed
   )

   # Extract batched features for ML pipelines
   wavelet_peaks = torch.stack(results['wavelet_peak_counts'])  # (n_scales, batch, n_bins)
   wavelet_l1 = torch.stack(results['wavelet_l1_norms'])        # (n_scales, batch, l1_nbins)

   # Reshape for neural network input: (batch, features)
   features = torch.cat([
       wavelet_peaks.permute(1, 0, 2).flatten(1),
       wavelet_l1.permute(1, 0, 2).flatten(1)
   ], dim=1)

Wavelet Decomposition
---------------------

Use the Starlet wavelet transform directly:

.. code-block:: python

   import torch
   from wl_stats_torch.starlet import Starlet2D

   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   starlet = Starlet2D(n_scales=5)

   # Input image
   image = torch.randn(512, 512, device=device)

   # Decompose - returns (n_scales, H, W) tensor
   wavelet_coeffs = starlet(image)

   print(f"Output shape: {wavelet_coeffs.shape}")  # (5, 512, 512)

Peak Detection
--------------

Find peaks in convergence maps:

.. code-block:: python

   import torch
   from wl_stats_torch.peaks import find_peaks_2d

   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

   # Create a test map
   kappa_map = torch.randn(512, 512, device=device)

   # Find peaks above threshold
   positions, heights = find_peaks_2d(kappa_map, threshold=3.0)

   print(f"Found {len(heights)} peaks")

Using Masks
-----------

Apply masks to exclude regions from analysis:

.. code-block:: python

   import torch
   from wl_stats_torch import WLStatistics

   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   stats = WLStatistics(n_scales=5, device=device)

   kappa_map = torch.randn(512, 512, device=device)
   sigma_map = torch.ones(512, 512, device=device) * 0.01

   # Create mask (1 = valid, 0 = masked)
   mask = torch.ones(512, 512, device=device)
   mask[:50, :] = 0  # Mask top 50 rows
   mask[-50:, :] = 0  # Mask bottom 50 rows

   results = stats.compute_all_statistics(
       kappa_map,
       sigma_map,
       mask=mask
   )
