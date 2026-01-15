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
   transformed = model.fit_transform(X)

   # Access results
   print(f"Components shape: {model.components_.shape}")  # (5, 50)
   print(f"Transformed shape: {transformed.shape}")                 # (100, 5)
   print(f"ELBO: {model.elbo_}")

Mathematical Formulation
-------------------------

Model
~~~~~

Given a non-negative data matrix :math:`Y \in \mathbb{R}_{+}^{N \times D}`, the PNMF model assumes each entry follows a Poisson distribution:

.. math::

   y_{ij} \sim \text{Poisson}(\lambda_{ij})

where the rate parameter :math:`\lambda_{ij}` is given by:

.. math::

   \lambda_{ij} = \sum_{\ell=1}^{L} W_{j\ell} \exp(F_{i\ell})

with the following components:

- :math:`W \in \mathbb{R}_{+}^{D \times L}` are the loadings (learned parameters)
- :math:`F \in \mathbb{R}^{N \times L}` are the latent factors (random variables)

The latent factors follow a Gaussian prior:

.. math::

   F_{i\ell} \sim \mathcal{N}(0, \sigma^2)

Variational Inference
~~~~~~~~~~~~~~~~~~~~~

We use **variational inference** to approximate the posterior distribution over the latent factors :math:`F`.
The variational distribution :math:`q(F)` is a Gaussian with learnable mean and scale parameters:

.. math::

   F_{i\ell} \sim q(F_{i\ell}) = \mathcal{N}(\mu_{i\ell}, \sigma_{i\ell}^2)

The Evidence Lower BOund (ELBO) is:

.. math::

   \mathcal{L} = \mathbb{E}_{q(F)}[\log p(Y|F)] - \text{KL}[q(F) \,||\, p(F)]

Expected Log-Likelihood Expansion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The expected log-likelihood term expands as follows. For a Poisson observation model, the log-likelihood is:

.. math::

   \log p(Y_{ij} \mid \lambda_{ij}) = Y_{ij} \log \lambda_{ij} - \lambda_{ij} - \log (Y_{ij}!)

Substituting the NSF parameterization :math:`\lambda_{ij} = \sum_{\ell} W_{j\ell} e^{F_{i\ell}}`, the expected log-likelihood becomes:

.. math::

   \mathbb{E}_{q(F_{i\cdot})}[\log p(Y_{ij} \mid F_{i\cdot})] =
   Y_{ij} \, \mathbb{E}_{q(F_{i\cdot})}\!\left[ \log \sum_{\ell} W_{j\ell} e^{F_{i\ell}} \right]
   - \sum_{\ell} W_{j\ell}\, \mathbb{E}_{q(F_{i\ell})}\!\left[ e^{F_{i\ell}} \right]
   - \log(Y_{ij}!)

The second expectation can be computed **analytically** using the moment-generating function of the Gaussian:

.. math::

   \mathbb{E}_{q(F_{i\ell})} \left[ e^{F_{i\ell}} \right]
   = \exp\left( \mu_{i\ell} + \frac{1}{2} \sigma_{i\ell}^2 \right)

This gives the final form:

.. math::

   \mathbb{E}_{q(F_{i \cdot})} [ \log p(Y_{ij} \mid F_{i \cdot}) ]
   = Y_{ij} \, \underbrace{\mathbb{E}_{q(F_{i \cdot})} \left[ \log \sum_\ell W_{j\ell} e^{F_{i\ell}} \right]}_{\text{Monte Carlo}}
   - \underbrace{\sum_\ell W_{j\ell} \, \exp\left( \mu_{i\ell} + \frac{1}{2} \sigma_{i\ell}^2 \right)}_{\text{Analytic}}
   - \log (Y_{ij}!)

where:
- The first term (log of sum) is estimated via Monte Carlo sampling with the reparameterization trick
- The second term (sum of exponentials) is computed analytically
- The third term is constant with respect to the parameters

Optimization
~~~~~~~~~~~~

The model is optimized by maximizing the ELBO using gradient ascent. The variational parameters
:math:`\mu_{i\ell}` and :math:`\sigma_{i\ell}` and the loadings :math:`W_{j\ell}` are learned jointly.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api
   examples
   benchmarks
   natural_gradient



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
