Installation
============

Requirements
------------

* Python >= 3.8
* PyTorch >= 2.0.0
* NumPy >= 1.20.0
* SciPy >= 1.7.0
* Matplotlib >= 3.3.0

Install from PyPI
-----------------

The easiest way to install wl-stats-torch:

.. code-block:: bash

   pip install wl-stats-torch

Install from Source
-------------------

Clone the repository and install in development mode:

.. code-block:: bash

   git clone https://github.com/AndreasTersenov/wl_stats_torch.git
   cd wl_stats_torch
   pip install -e .

For development with testing and documentation tools:

.. code-block:: bash

   pip install -e ".[dev]"

Docker
------

Run wl-stats-torch in a container without installing dependencies locally.

**CPU Version:**

.. code-block:: bash

   docker build -t wl-stats-torch:cpu .
   docker run -it --rm -v $(pwd)/data:/data wl-stats-torch:cpu

**GPU Version** (requires nvidia-docker):

.. code-block:: bash

   docker build -t wl-stats-torch:cuda -f Dockerfile.cuda .
   docker run -it --rm --gpus all -v $(pwd)/data:/data wl-stats-torch:cuda

**Docker Compose:**

.. code-block:: bash

   # CPU
   docker compose run --rm wl-stats-cpu

   # GPU
   docker compose run --rm wl-stats-gpu

See the ``docs-md/DOCKER.md`` file for detailed Docker usage instructions.

GPU Support
-----------

The package automatically detects and uses CUDA-enabled GPUs if available.
To use GPU acceleration, ensure you have:

* CUDA-compatible GPU
* CUDA Toolkit (version compatible with your PyTorch installation)
* PyTorch with CUDA support

Check your PyTorch CUDA support:

.. code-block:: python

   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   if torch.cuda.is_available():
       print(f"Device: {torch.cuda.get_device_name(0)}")

Verify Installation
-------------------

Test that the package is installed correctly:

.. code-block:: python

   from wl_stats_torch import WLStatistics
   import torch

   stats = WLStatistics(n_scales=3)
   kappa = torch.randn(64, 64)
   results = stats.compute_all_statistics(kappa, 0.01)
   print("Installation successful!")
