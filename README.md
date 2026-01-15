# PNMF: Probabilistic Non-negative Matrix Factorization

A PyTorch-based implementation of Probabilistic NMF with projected gradient optimization, compatible with scikit-learn's API.

## Installation

```bash
pip install pnmf
```

For development installation:

```bash
git clone https://github.com/luisdiaz1997/Probabilistic-NMF.git
cd Probabilistic-NMF
pip install -e .
```

## Quick Start

```python
from PNMF import PNMF
import numpy as np

# Create some sample data
X = np.random.rand(100, 50)

# Initialize and fit the model
model = PNMF(n_components=5, init='random', random_state=42)
transformed = model.fit_transform(X)  # Transformed data: (100, 5)
components = model.components_        # Components: (5, 50)

# Reconstruct the data
X_reconstructed = model.inverse_transform(transformed)
```

## Model

PNMF uses a Poisson likelihood for the observed data:

- y<sub>ij</sub> ~ Poisson(λ<sub>ij</sub>)
- λ<sub>ij</sub> = Σ<sub>l</sub> W<sub>jl</sub> exp(F<sub>il</sub>)
- F<sub>il</sub> ~ N(0, σ²)

where **W** ≥ 0 are the loadings and **F** are the latent factors (variational inference with Gaussian prior).

## Features

- **Probabilistic NMF**: Uses Poisson factorization for count data
- **Projected Gradients**: Three modes for enforcing non-negativity:
  - `softplus`: Uses softplus transformation
  - `exp`: Uses exponential transformation
  - `projected`: Uses projected gradient descent (default)
- **sklearn-compatible API**: Works seamlessly with scikit-learn workflows
- **GPU Support**: Automatic CUDA detection and utilization

## API Reference

### `PNMF(n_components=2, init='random', init_mode='projected', max_iter=200, tol=1e-4, learning_rate=0.01, random_state=None, verbose=False, device='auto')`

Parameters:
- `n_components`: Number of latent components (default: 2)
- `init`: Initialization method - 'random' or 'custom' (default: 'random')
- `init_mode`: Non-negativity enforcement - 'softplus', 'exp', or 'projected' (default: 'projected')
- `max_iter`: Maximum number of iterations (default: 200)
- `tol`: Tolerance for convergence (default: 1e-4)
- `learning_rate`: Learning rate for gradient descent (default: 0.01)
- `random_state`: Random seed for reproducibility (default: None)
- `verbose`: Whether to print progress messages (default: False)
- `device`: Device to use - 'cpu', 'cuda', or 'auto' (default: 'auto')

Methods:
- `fit(X, y=None, W=None, H=None)`: Fit the model to data X
- `transform(X)`: Transform X using the fitted model
- `fit_transform(X, **kwargs)`: Fit the model and transform X
- `inverse_transform(transformed)`: Transform data back to original space

## License

GNU General Public License v2.0

## Author

Luis Chumpitaz Diaz

## Acknowledgments

This implementation borrows code from [GPzoo](https://github.com/luisdiaz1997/GPzoo).
