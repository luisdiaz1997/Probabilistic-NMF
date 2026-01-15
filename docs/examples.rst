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
