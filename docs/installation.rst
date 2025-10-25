Installation
============

Requirements
------------

* Python >= 3.8
* PyTorch >= 2.0.0
* NumPy >= 1.20.0
* SciPy >= 1.7.0
* Matplotlib >= 3.3.0

Install from source
-------------------

Clone the repository and install in development mode:

.. code-block:: bash

   git clone https://github.com/AndreasTersenov/wl_stats_torch.git
   cd wl_stats_torch
   pip install -e .

For development with testing and documentation tools:

.. code-block:: bash

   pip install -e ".[dev]"

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
   print(torch.cuda.is_available())
   print(torch.cuda.get_device_name(0))
