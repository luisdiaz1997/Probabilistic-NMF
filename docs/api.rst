API Reference
=============

This page contains the API reference documentation for PNMF.

Main Interface
--------------

.. autoclass:: PNMF.PNMF
   :members:
   :undoc-members:
   :show-inheritance:

Transform and Utility Functions
-------------------------------

Factor Extraction
~~~~~~~~~~~~~~~~~

.. autofunction:: PNMF.transforms.log_factors

.. autofunction:: PNMF.transforms.factors

.. autofunction:: PNMF.transforms.factor_uncertainty

.. autofunction:: PNMF.transforms.factor_samples

Model Accessors
~~~~~~~~~~~~~~~

.. autofunction:: PNMF.transforms.get_loadings

.. autofunction:: PNMF.transforms.get_prior

Conditional Inference
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: PNMF.transforms.transform_F

.. autofunction:: PNMF.transforms.transform_W

Prior Utilities
~~~~~~~~~~~~~~~

.. autofunction:: PNMF.transforms.log_factors_from_prior

.. autofunction:: PNMF.transforms.factors_from_prior

.. autofunction:: PNMF.transforms.uncertainty_from_prior

PyTorch Models
--------------

.. autoclass:: PNMF.models.PoissonFactorization
   :members:
   :undoc-members:
   :show-inheritance:

Priors
------

.. autoclass:: PNMF.priors.GaussianPrior
   :members:
   :undoc-members:
   :show-inheritance:

Custom Modules
--------------

.. autoclass:: PNMF.custom_modules.ConstrainedParameter
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: PNMF.custom_modules.PositiveParameter
   :members:
   :undoc-members:
   :show-inheritance:
