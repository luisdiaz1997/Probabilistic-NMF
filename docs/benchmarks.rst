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

The following plot shows the convergence behavior of both modes over 2000 iterations.
The left panel shows the **loss** (negative ELBO) on a log scale, and the right panel
shows the distance to convergence.

.. image:: ../benchmarks/convergence_comparison.png
   :align: center
   :width: 100%

**Key Observations:**

* Both modes achieve similar final ELBO values (-51111 simple vs -49931 expanded)
* The difference (1180) is small, indicating both estimators converge to nearly the same solution
* The loss curves on a log scale clearly show the optimization trajectory
* Using E=3 Monte Carlo samples (default) provides good gradient estimates
* Adam optimizer with lr=0.01 works well for this problem (SGD diverges)
* Data is generated as integer counts via Poisson sampling, appropriate for the model
* Full reproducibility: np.random.seed(42) + torch.manual_seed(42)

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
