Examples
========

Example scripts are available in the ``examples/`` directory and Jupyter notebooks
in the ``notebooks/`` directory of the repository.

Python Scripts
--------------

Example scripts in the ``examples/`` directory:

Basic Usage
~~~~~~~~~~~

``examples/basic_usage.py`` - Simple example with synthetic data demonstrating
the core functionality of computing weak lensing statistics.

Batch Processing Demo
~~~~~~~~~~~~~~~~~~~~~

``examples/batch_processing_demo.py`` - Comprehensive batch processing examples
showing how to efficiently process multiple convergence maps simultaneously
with 12-19x speedup on GPU.

CFIS Example
~~~~~~~~~~~~

``examples/cfis_example.py`` - Working with realistic CFIS survey-like simulations.

Jupyter Notebooks
-----------------

Interactive examples in the ``notebooks/`` directory:

* ``cuda_batch_demo.ipynb`` - GPU batch processing demonstration with benchmarks
* ``des_mock_demo.ipynb`` - DES mock catalog analysis
* ``pycs_demo.ipynb`` - PyCS integration example

Running Examples
----------------

To run the basic example:

.. code-block:: bash

   cd examples
   python basic_usage.py

To run the batch processing demo:

.. code-block:: bash

   cd examples
   python batch_processing_demo.py

For Jupyter notebooks:

.. code-block:: bash

   cd notebooks
   jupyter notebook cuda_batch_demo.ipynb
