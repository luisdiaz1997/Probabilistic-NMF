.. PNMF documentation master file, created by sphinx-quickstart.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PNMF's documentation!
=================================

PNMF (Probabilistic Non-negative Matrix Factorization) is a Python package that implements
variational inference for Poisson factorization with a scikit-learn compatible API.

.. note::
   This project is licensed under the GNU General Public License v2.0 (GPL-2.0).

Installation
------------

Install from PyPI:

.. code-block:: bash

   pip install pnmf

Or install from source:

.. code-block:: bash

   git clone https://github.com/luisfcd/Probabilistic-NMF.git
   cd Probabilistic-NMF
   pip install -e .


Quick Start
-----------

.. code-block:: python

   from PNMF import PNMF
   import numpy as np

   # Create sample data
   X = np.random.rand(100, 50)

   # Initialize and fit
   model = PNMF(n_components=5, random_state=42, verbose=True)
   W = model.fit_transform(X)

   # Access results
   print(f"Components shape: {model.components_.shape}")  # (5, 50)
   print(f"Transformed shape: {W.shape}")                 # (100, 5)
   print(f"ELBO: {model.elbo_}")

Mathematical Formulation
-------------------------

Model
~~~~~

The PNMF model factorizes a non-negative matrix :math:`X \in \mathbb{R}_{+}^{N \times D}` as:

.. math::

   X \approx W @ \exp(F)

where:

- :math:`W \in \mathbb{R}_{+}^{D \times L}` are the loadings (learned parameters)
- :math:`F \in \mathbb{R}^{L \times N}` are the latent factors (random variables)

Variational Inference
~~~~~~~~~~~~~~~~~~~~~

We use **variational inference** to approximate the posterior distribution over the latent factors :math:`F`.
The variational distribution :math:`q(F)` is a Gaussian with learnable mean and scale parameters.

The Evidence Lower BOund (ELBO) is:

.. math::

   \mathcal{L} = \mathbb{E}_{q(F)}[\log p(X|F)] - \text{KL}[q(F) \,||\, p(F)]

where:

- :math:`\mathbb{E}_{q(F)}[\log p(X|F)]` is the expected log-likelihood under the Poisson model
- :math:`\text{KL}[q(F) \,||\, p(F)]` is the KL divergence between the variational and prior distributions

Optimization
~~~~~~~~~~~~

The model is optimized by maximizing the ELBO using gradient ascent with the reparameterization trick.
The gradient is estimated using Monte Carlo sampling with :math:`E` samples.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api
   examples



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
