Benchmarks
=========

This section contains benchmarks comparing different aspects of the PNMF implementation.

Simple vs Expanded ELBO Modes
------------------------------

.. note::
   This notebook requires ``pnmf`` to be installed. Install from PyPI with ``pip install pnmf``.

The following Jupyter notebook compares the two ELBO computation modes available in PNMF:

.. raw:: html

    <div style="margin-bottom: 20px;"></div>

**`mode='simple'`**
   Uses full Monte Carlo estimation for all terms in the Poisson log-likelihood.

**`mode='expanded'`** (default)
   Uses a hybrid approach with Monte Carlo estimation for the first term and
   analytic computation for the second term using the Gaussian moment-generating function.

.. nbsphinx:: ../benchmarks/simple_vs_expanded.ipynb

Key Takeaways
-------------

Based on the benchmark results:

* **Convergence Speed**: The expanded mode typically converges faster due to lower variance
  from the analytic computation of the second term.

* **Final ELBO**: Both modes should converge to similar ELBO values, though the expanded mode
  may reach a higher value due to more stable gradients.

* **Variance**: The simple mode has higher variance because all terms are estimated via
  Monte Carlo, while the expanded mode reduces variance by computing ``E[exp(F)]`` analytically.

Running the Benchmark Locally
------------------------------

You can also run the benchmark locally using the standalone Python script:

.. code-block:: bash

   python benchmarks/simple_vs_expanded.py

Or launch the Jupyter notebook:

.. code-block:: bash

   jupyter notebook benchmarks/simple_vs_expanded.ipynb
