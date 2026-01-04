Benchmarks
=========

This section contains benchmarks comparing different aspects of the PNMF implementation.

Simple vs Expanded ELBO Modes
------------------------------

.. note::
   This notebook requires ``pnmf`` to be installed. Install from PyPI with ``pip install pnmf``.

The following Jupyter notebook compares the two ELBO computation modes available in PNMF:

**`mode='simple'`**
   Uses full Monte Carlo estimation for all terms in the Poisson log-likelihood.

**`mode='expanded'`** (default)
   Uses a hybrid approach with Monte Carlo estimation for the first term and
   analytic computation for the second term using the Gaussian moment-generating function.

Convergence Comparison
~~~~~~~~~~~~~~~~~~~~~~

The following plot shows the convergence behavior of both modes over 8000 iterations.
The left panel shows the **loss** (negative ELBO) on a log scale, and the right panel
shows the distance to convergence.

.. image:: ../benchmarks/convergence_comparison.png
   :align: center
   :width: 100%

**Benchmark Parameters:**

* **Monte Carlo samples (E)**: 10 (reduced variance in gradient estimates)
* **Learning rate**: 0.005 (conservative for stable convergence)
* **Optimizer**: Adam
* **Max iterations**: 8000 (tolerance: 1e-4)
* **Data**: 200 samples × 100 features, 5 true components
* **Data generation**: Poisson sampling for integer counts (appropriate for model)
* **Device**: MPS (Apple Silicon) with automatic detection (CUDA > MPS > CPU)
* **Random seed**: 42 for full reproducibility

**Key Results:**

* **Convergence speed**: Expanded mode converges ~1.08x faster (7022 vs 7592 iterations)
* **Final ELBO values**: -47322.6055 (simple) vs -47198.6055 (expanded)
* **ELBO difference**: 124.0 (small, indicating both estimators converge to nearly the same solution)
* **Reconstruction error**: 0.241737 (simple) vs 0.241310 (expanded) - nearly identical
* **Winner**: Expanded mode (faster convergence + higher final ELBO + lower reconstruction error)

.. raw:: html

    <div style="margin-bottom: 20px;"></div>

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
