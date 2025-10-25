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
       nbins=31
   )

   # Access results
   print("Mono-scale peaks:", results['mono_peaks'])
   print("Wavelet peaks:", results['wavelet_peaks'])
   print("L1-norms:", results['l1_norms'])

Batch Processing
----------------

Process multiple maps efficiently:

.. code-block:: python

   import torch
   from wl_stats_torch import WLStatistics

   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   stats = WLStatistics(n_scales=5, device=device)

   # Batch of maps (N, H, W)
   kappa_batch = torch.randn(10, 512, 512, device=device)
   sigma_batch = torch.ones(10, 512, 512, device=device) * 0.01

   # Process all maps
   all_results = []
   for kappa, sigma in zip(kappa_batch, sigma_batch):
       results = stats.compute_all_statistics(kappa, sigma)
       all_results.append(results)

Wavelet Decomposition
---------------------

Use the Starlet wavelet transform directly:

.. code-block:: python

   import torch
   from wl_stats_torch import Starlet2D

   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   starlet = Starlet2D(n_scales=5, device=device)

   # Input image
   image = torch.randn(512, 512, device=device)

   # Decompose
   coeffs = starlet.decompose(image)

   # coeffs is a list of wavelet coefficients at each scale
   print(f"Number of scales: {len(coeffs)}")
   print(f"Shape of each scale: {coeffs[0].shape}")

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
   positions, heights = find_peaks_2d(kappa_map, threshold=3.0, ordered=True)
   
   print(f"Found {len(positions)} peaks")
   print(f"Peak heights: {heights}")
