Examples
========

Basic Usage
-----------

.. code-block:: python

   from PNMF import PNMF
   import numpy as np

   # Create sample data (positive floats)
   np.random.seed(42)
   X = np.random.rand(100, 50)

   # Initialize model with 5 components
   model = PNMF(n_components=5, random_state=42)

   # Fit the model
   transformed = model.fit_transform(X)

   # Access the learned components
   components = model.components_

   print(f"Transformed data shape: {transformed.shape}")     # (100, 5)
   print(f"Components shape: {components.shape}")           # (5, 50)
   print(f"Final ELBO: {model.elbo_}")             # Evidence Lower Bound
   print(f"Iterations: {model.n_iter_}")           # Number of iterations

Initialization Methods
----------------------

Different initialization strategies can significantly affect convergence speed and final model quality:

.. code-block:: python

   from PNMF import PNMF
   import numpy as np

   X = np.random.poisson(5, size=(100, 50)).astype(np.float32)

   # Random initialization (simple baseline)
   model_random = PNMF(n_components=5, init='random', random_state=42)

   # NNDSVD initialization (sparse, good for sparse data)
   model_nndsvd = PNMF(n_components=5, init='nndsvd', random_state=42)

   # NNDSVD with average fill (dense, recommended for most cases)
   model_nndsvda = PNMF(n_components=5, init='nndsvda', random_state=42)

   # NNDSVD with random fill (faster dense alternative)
   model_nndsvdar = PNMF(n_components=5, init='nndsvdar', random_state=42)

   # K-means clustering initialization
   model_kmeans = PNMF(n_components=5, init='k-means', random_state=42)

   # Auto initialization (None): uses 'nndsvda' if n_components <= min(n_samples, n_features)
   # otherwise falls back to 'random'
   model_auto = PNMF(n_components=5, init=None, random_state=42)

   # Fit and compare
   model_random.fit(X)
   model_nndsvda.fit(X)

   print(f"Random init ELBO: {model_random.elbo_:.2f}")
   print(f"NNDSVDa init ELBO: {model_nndsvda.elbo_:.2f}")

The available initialization methods are:

- ``None`` (default): Auto-select 'nndsvda' if n_components <= min(n_samples, n_features), otherwise 'random'
- ``'random'``: Non-negative random matrices, scaled with ``sqrt(X.mean() / n_components)``
- ``'nndsvd'``: Nonnegative Double SVD - produces sparse results, best for sparse factorization
- ``'nndsvda'``: NNDSVD with zeros filled with average of X - dense factorization, recommended for most cases
- ``'nndsvdar'``: NNDSVD with zeros filled with small random values - faster dense alternative
- ``'k-means'``: K-means clustering based initialization - good for clustered data

Using Different Loadings Modes
-------------------------------

The :class:`~PNMF.custom_modules.PositiveParameter` class supports three modes for enforcing positivity:

.. code-block:: python

   from PNMF import PNMF

   # Projected gradient (default): clamps values >= 0 after each step
   model1 = PNMF(n_components=5, loadings_mode='projected')

   # Softplus: uses softplus transformation for smooth positivity
   model2 = PNMF(n_components=5, loadings_mode='softplus')

   # Exponential: uses exp transformation
   model3 = PNMF(n_components=5, loadings_mode='exp')

PyTorch API
-----------

For more flexibility, you can use the PyTorch-native API:

.. code-block:: python

   import torch
   from PNMF import PoissonFactorization, GaussianPrior
   import numpy as np

   # Prepare data (note: PyTorch expects D x N format)
   X = np.random.rand(100, 50)
   y = torch.from_numpy(X.T.astype(np.float32))  # (50, 100)

   # Create variational prior
   L = 5  # number of components
   prior = GaussianPrior(y, L=L)

   # Create model
   model = PoissonFactorization(prior, y, L=L)

   # Forward pass with E Monte Carlo samples
   rate, qF, pF = model(E=3)

   print(f"Rate shape: {rate.shape}")  # (50, 100)
   print(f"Variational distribution: {qF}")
   print(f"Prior distribution: {pF}")

Customizing Optimization
------------------------

.. code-block:: python

   from PNMF import PNMF

   # Custom optimization parameters
   model = PNMF(
       n_components=10,
       max_iter=500,           # Maximum iterations
       tol=1e-5,               # Tolerance for early stopping
       learning_rate=0.02,     # Learning rate for optimizer
       E=5,                    # Monte Carlo samples for ELBO
       verbose=True,           # Print progress
       random_state=42
   )

   model.fit(X)

Extracting Latent Factors
-------------------------

After fitting a model, you can extract latent factors in different forms:

.. code-block:: python

   from PNMF import PNMF, log_factors, factors, factor_uncertainty, factor_samples
   import numpy as np

   # Fit model
   X = np.random.poisson(5, size=(100, 50)).astype(np.float32)
   model = PNMF(n_components=5, random_state=42)
   model.fit(X)

   # Get log-space factors (μ from q(F) = Normal(μ, σ²))
   F_log = log_factors(model)  # Shape: (100, 5)

   # Get exp-space factors using moment-generating function
   # E[exp(F)] = exp(μ + σ²/2)
   F_exp = factors(model)  # Shape: (100, 5)

   # Get exp-space factors without MGF correction (biased)
   F_exp_biased = factors(model, use_mgf=False)  # exp(μ)

   # Get uncertainty (standard deviation)
   F_std = factor_uncertainty(model)  # Shape: (100, 5)

   # Get variance
   F_var = factor_uncertainty(model, return_variance=True)

   # Sample from the variational posterior
   samples = factor_samples(model, n_samples=100)  # Shape: (100, 100, 5)
   exp_samples = factor_samples(model, n_samples=100, return_exp=True)

Accessing Model Components
--------------------------

.. code-block:: python

   from PNMF import PNMF, get_loadings, get_prior
   import numpy as np

   X = np.random.poisson(5, size=(100, 50)).astype(np.float32)
   model = PNMF(n_components=5).fit(X)

   # Get loadings matrix W (shape: n_features x n_components)
   W = get_loadings(model)  # Shape: (50, 5)

   # This is equivalent to model.components_.T
   assert np.array_equal(W, model.components_.T)

   # Get the GaussianPrior object for advanced users
   prior = get_prior(model)
   qF, pF = prior()  # Get variational and prior distributions

Conditional Inference
---------------------

Transform New Data with Full Variational Inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :func:`~PNMF.transforms.transform_F` function learns a full variational
distribution for new data, given fixed loadings W:

.. code-block:: python

   from PNMF import PNMF, transform_F, get_loadings
   from PNMF.transforms import log_factors_from_prior, factors_from_prior
   import numpy as np

   # Fit model on training data
   X_train = np.random.poisson(5, size=(100, 50)).astype(np.float32)
   model = PNMF(n_components=5, max_iter=100).fit(X_train)
   W = get_loadings(model)

   # Transform new data with full VI (better than NNLS-based transform)
   X_test = np.random.poisson(5, size=(20, 50)).astype(np.float32)
   F_test = transform_F(X_test, W, max_iter=100)  # Shape: (20, 5)

   # Or get the full prior object for uncertainty quantification
   prior_test = transform_F(X_test, W, max_iter=100, return_prior=True)
   F_test_log = log_factors_from_prior(prior_test)
   F_test_exp = factors_from_prior(prior_test)

Learning New Loadings
~~~~~~~~~~~~~~~~~~~~~

The :func:`~PNMF.transforms.transform_W` function learns new loadings W
conditioned on fixed latent factors F:

.. code-block:: python

   from PNMF import PNMF, transform_W, log_factors
   import numpy as np

   # Fit model
   X = np.random.poisson(5, size=(100, 50)).astype(np.float32)
   model = PNMF(n_components=5, max_iter=100).fit(X)
   F_log = log_factors(model)

   # Learn new W for same data (useful for transfer learning)
   W_new = transform_W(X, F_log, max_iter=100)  # Shape: (50, 5)

Uncertainty Quantification
--------------------------

PNMF provides full uncertainty quantification through the variational posterior:

.. code-block:: python

   from PNMF import PNMF, factors, factor_uncertainty, factor_samples, get_loadings
   import numpy as np

   # Fit model
   X = np.random.poisson(5, size=(100, 50)).astype(np.float32)
   model = PNMF(n_components=5, max_iter=100).fit(X)

   # Point estimates
   F_exp = factors(model)
   F_std = factor_uncertainty(model)

   # Reconstruct with uncertainty
   W = get_loadings(model)
   X_recon = F_exp @ W.T

   # Propagate uncertainty through sampling
   n_mc_samples = 1000
   samples = factor_samples(model, n_samples=n_mc_samples, return_exp=True)

   # Compute reconstruction uncertainty
   reconstructions = np.array([s @ W.T for s in samples])
   recon_mean = reconstructions.mean(axis=0)
   recon_std = reconstructions.std(axis=0)

   print(f"Reconstruction uncertainty: {recon_std.mean():.4f}")
