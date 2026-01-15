# PNMF Development Notes

## Project Overview

This document describes the development of the PNMF (Probabilistic Non-negative Matrix Factorization) library, a pip-installable Python package with a scikit-learn compatible API using **variational inference**.

## What Was Done

### 1. Package Structure

```
Probabilistic-NMF/
├── PNMF/                    # Python package module
│   ├── __init__.py          # Package initialization and exports
│   ├── models.py            # PoissonFactorization (PyTorch) + PNMF (sklearn API)
│   ├── priors.py            # GaussianPrior class for variational inference
│   ├── elbo.py              # Expected log-likelihood and ELBO computation
│   ├── optimizers.py        # Custom optimizers (NaturalGradientDescent)
│   └── custom_modules.py    # Constrained parameter classes (ConstrainedParameter, PositiveParameter)
├── tests/                   # Test suite
│   ├── __init__.py
│   └── test_pnmf.py         # Pytest tests for all components
├── benchmarks/              # Benchmark scripts and notebooks
│   ├── README.md            # Benchmark documentation
│   ├── simple_vs_expanded.py    # Standalone benchmark script
│   └── simple_vs_expanded.ipynb # Jupyter notebook with visualizations
├── docs/                    # Sphinx documentation
│   ├── conf.py              # Sphinx configuration with MathJax3 and nbsphinx
│   ├── index.rst            # Landing page with mathematical formulation
│   ├── api.rst              # Auto-generated API reference
│   ├── examples.rst         # Usage examples
│   ├── benchmarks.rst       # Benchmark page with embedded notebook
│   └── requirements.txt     # Documentation dependencies
├── setup.py                 # Minimal setup for backwards compatibility
├── pyproject.toml           # Package metadata and all dependencies
├── .readthedocs.yaml        # Read the Docs configuration
├── README.md                # Documentation
├── LICENSE                  # GPL v2.0 license
└── .gitignore               # Git ignore patterns
```

### 2. Code Borrowed from GPzoo

The following components were adapted from [GPzoo](https://github.com/luisdiaz1997/GPzoo):

**From `gpzoo/modules.py`:**
- `PositiveParameter` class - Handles constrained positive parameters with three modes:
  - `softplus`: Uses softplus transformation for positivity
  - `exp`: Uses exponential transformation for positivity
  - `projected`: Uses projected gradient descent (clamps values >= 0 after each step) [DEFAULT]

**From `gpzoo/gp.py`:**
- `GaussianPrior` class - Variational distribution with mean/scale parameters
- Returns (qF, pF) for ELBO computation
- **NEW**: Now supports natural gradient parameterization (`use_natural_gradients=True`)

**From `gpzoo/likelihoods.py`:**
- `PoissonFactorization` base class - Variational Poisson factorization
- Removed the `V` parameter (sample-specific scaling) for simplicity

### 3. Architecture: Variational Inference

The model uses **variational inference** with:
- **GaussianPrior**: Variational distribution qF over latent factors F
- **PoissonFactorization**: PyTorch nn.Module with W (PositiveParameter) and prior
- **ELBO optimization**: `ELBO = E[log p(Y|F)] - KL[q(F) || p(F)]`

**Model equation:**
```
X ≈ exp(F) @ W.T
```
where:
- F is the latent factor matrix (sample-specific, sampled from Gaussian variational distribution)
- W is the loading matrix (learned via PositiveParameter, projected gradient by default)
- For sklearn API: X (n_samples, n_features) ≈ exp(F) (n_samples, n_components) @ W.T (n_components, n_features)
- Internal representation: X (D, N) ≈ W (D, L) @ exp(F) (L, N)

### 4. Key Features Implemented

**sklearn-compatible API:**
```python
from PNMF import PNMF
import numpy as np

# Initialize (variational inference by default)
model = PNMF(n_components=5, random_state=42)

# Fit and transform
transformed = model.fit_transform(X)    # Shape: (n_samples, n_components) - exp(F)
components = model.components_         # Shape: (n_components, n_features) - W.T

# Access ELBO
print(f"ELBO: {model.elbo_}")
```

**PyTorch-native API:**
```python
from PNMF import PoissonFactorization, GaussianPrior
import torch

y = torch.from_numpy(X.T.astype(np.float32))  # (D, N)
prior = GaussianPrior(y, L=5)
model = PoissonFactorization(prior, y, L=5)
rate, qF, pF = model(E=3)  # Returns rate tensor and distributions
```

**Default Parameters:**
- `n_components`: 10
- `loadings_mode`: `'projected'` (clamp after each step)
- `mode`: `'expanded'` (hybrid Monte Carlo + analytic ELBO)
- `training_mode`: `'standard'` (standard gradient descent)
- `E`: 3 (Monte Carlo samples for ELBO)
- `max_iter`: 200
- `tol`: 1e-4
- `learning_rate`: 0.01
- `optimizer`: `'Adam'` (Adam, AdamW, NAdam, SGD, RMSprop)

### 5. Installation

**PyPI package name:** `pnmf` (lowercase)
**Python import:** `from PNMF import PNMF` (uppercase)

### 6. License

- **License:** GNU General Public License v2.0 (GPL-2.0)
- **Author:** Luis Chumpitaz Diaz

## Technical Details

### ELBO Computation Modes

The PNMF implementation supports **three modes** for computing the Evidence Lower BOund (ELBO):

#### `mode='expanded'` (Default)

Uses a **hybrid approach** combining Monte Carlo estimation with analytic computation:

**First term (Monte Carlo):**
```
Y_ij * E_q[log Σ_l W_jl * exp(F_il)]
```
Estimated via Monte Carlo with reparameterization trick.

**Second term (Analytic):**
```
Σ_l W_jl * E_q[exp(F_il)]
```
Computed exactly using the Gaussian moment-generating function: `E[exp(F)] = exp(μ + σ²/2)`

**Third term (Constant):**
```
-Σ_ij log(Y_ij!)
```
Poisson normalization constant computed via `torch.lgamma(X + 1).sum()`

**Advantages:**
- Lower variance due to analytic second term
- More stable gradients
- Typically converges faster

#### `mode='simple'`

Uses **full Monte Carlo estimation** via `torch.distributions.Poisson.log_prob()`:

```
log p(X|rate) = X * log(rate) - rate - log(X!)
```

All terms are estimated via Monte Carlo sampling using PyTorch's built-in Poisson distribution.

**Advantages:**
- Clean implementation using PyTorch's distribution API
- Numerically stable (PyTorch handles edge cases)
- Easier to understand

**Disadvantages:**
- Higher variance in gradient estimates
- May converge more slowly
- Less stable optimization

#### `mode='lower-bound'`

Uses **Jensen's inequality** for a fully analytic lower bound with **zero Monte Carlo sampling**:

**Jensen's sandwich bounds for the log-sum-exp term:**
```
log Σ W * exp(μ) ≤ E[log Σ W * exp(F)] ≤ log Σ W * exp(μ + σ²/2)
```

The lower bound uses the left inequality:
```
E[log Σ W * exp(F)] ≥ log Σ W * exp(E[F]) = log Σ W * exp(μ)
```

**Advantages:**
- **Fastest**: No Monte Carlo sampling overhead
- **Zero variance**: Fully deterministic gradients
- **True lower bound**: Mathematically guaranteed
- **Best for large-scale**: When q(F) is parameterized by neural networks or GPs

**Disadvantages:**
- Lower final ELBO (expected, as it's a bound)
- May converge to slightly different local optimum

**Usage:**
```python
# Lower-bound mode (fastest, fully analytic)
model_lb = PNMF(n_components=5, mode='lower-bound')

# Expanded mode (hybrid, default)
model_exp = PNMF(n_components=5, mode='expanded')

# Simple mode (full Monte Carlo)
model_simple = PNMF(n_components=5, mode='simple')
```

### Variational Inference

The model maximizes the Evidence Lower BOund (ELBO):
```
ELBO = E[log p(Y|F)] - KL[q(F) || p(F)]
```

See the **ELBO Computation Modes** section above for details on how the expected log-likelihood is computed.

- **KL divergence**: `torch.distributions.kl_divergence(qF, pF)`
- **Reparameterization**: `F = qF.rsample((E,))` for gradient estimation

### Classes

**NaturalToMuS** (`custom_modules.py`)
- Custom autograd function for natural parameter conversion
- Forward: (θ₁, θ₂) → (μ, s) where θ₁ = μ/s², θ₂ = -1/(2s²)
- Backward: Returns gradients w.r.t. expectation parameters (η₁, η₂)
- Enables natural gradient descent for variational inference

**NaturalGradientDescent** (`models.py`)
- Custom PyTorch optimizer implementing NGD for variational parameters
- Uses learning rate scaled by `1/num_data` as per natural gradient theory
- Default learning rate: `lr=0.1`

**GaussianPrior** (`priors.py`)
- Variational mean parameter (nn.Parameter) [standard mode]
- Variational scale parameter (PositiveParameter with softplus) [standard mode]
- Natural parameters (θ₁, θ₂) for NGD [natural mode]
- `use_natural_gradients` parameter to switch between modes
- Returns (qF, pF) distributions

**PoissonFactorization** (`models.py`)
- PyTorch nn.Module
- Has W (PositiveParameter) for loadings
- `forward(E=3)` returns (rate, qF, pF)

**PNMF** (`models.py`)
- sklearn-compatible wrapper
- Creates GaussianPrior internally
- Uses ELBO loss instead of NLL
- `training_mode` parameter: `'standard'` or `'natural'`

### PositiveParameter Class

A sophisticated parameter class that:
- Stores an unconstrained `_raw` parameter internally
- Applies transformations (`softplus`, `exp`, or `projected`) to enforce positivity
- Provides a tensor-like interface for easy manipulation
- Supports projection via `project()` method for constrained optimization

## Usage Example

```python
from PNMF import PNMF
import numpy as np

# Create sample data (positive floats)
X = np.random.rand(100, 50)

# Initialize and fit
model = PNMF(n_components=5, random_state=42, verbose=True)
transformed = model.fit_transform(X)

# Access results
print(f"Components shape: {model.components_.shape}")  # (5, 50)
print(f"Transformed shape: {transformed.shape}")                 # (100, 5)
print(f"ELBO: {model.elbo_}")
print(f"Iterations: {model.n_iter_}")
```

## Testing

### Running Tests

The package includes a comprehensive test suite using pytest:

```bash
# Run all tests
python -m pytest tests/test_pnmf.py -v

# Run specific test class
python -m pytest tests/test_pnmf.py::TestELBOModes -v

# Run with coverage (if pytest-cov installed)
python -m pytest tests/test_pnmf.py --cov=PNMF
```

### Test Coverage

The test suite covers:
- **TestPNMFBasic**: sklearn API (fit, transform, fit_transform, inverse_transform)
- **TestELBOModes**: All three ELBO modes (simple, expanded, lower-bound)
- **TestTrainingModes**: Standard and natural gradient training
- **TestELBOFunctions**: Direct testing of ELBO computation functions
- **TestPyTorchAPI**: PyTorch-native API (PoissonFactorization, GaussianPrior)
- **TestParameterValidation**: Input validation and error handling

### Quick Verification

Quick test to verify the implementation works:

```python
from PNMF import PNMF
import numpy as np

# Create integer count data (appropriate for Poisson model)
np.random.seed(42)
X = np.random.poisson(lam=5, size=(50, 30)).astype(np.float32)

# Initialize and fit
model = PNMF(n_components=5, random_state=42, verbose=True, max_iter=20)
transformed = model.fit_transform(X)

print(f'Components shape: {model.components_.shape}')  # (5, 30)
print(f'Transformed shape: {transformed.shape}')       # (50, 5)
print(f'ELBO: {model.elbo_:.4f}')
print(f'Iterations: {model.n_iter_}')
```

## Documentation

The project uses **Sphinx** with the **Read the Docs** theme for documentation.

### Building Documentation Locally

```bash
# Install documentation dependencies
pip install -r docs/requirements.txt

# Build HTML documentation
sphinx-build -b html docs/ docs/_build/html

# View in browser
open docs/_build/html/index.html  # macOS
# or
xdg-open docs/_build/html/index.html  # Linux
```

### Documentation Features

- **MathJax3**: Full LaTeX support for mathematical equations
  - Custom macros: `\E` for expectation, `\KL` for KL divergence, `\calL` for loss
- **Autodoc**: Automatic API documentation from docstrings
- **Intersphinx**: Links to Python, PyTorch, NumPy, and scikit-learn docs
- **Napoleon**: Support for Google and NumPy style docstrings
- **nbsphinx**: Support for rendering Jupyter notebooks in the documentation

### Benchmark Page

The documentation includes a **Benchmark** page that compares all three ELBO computation modes:

- **Notebook**: `benchmarks/simple_vs_expanded.ipynb` embedded directly in the docs
- **Comparison**: Convergence speed, final ELBO, and reconstruction error for all three modes
- **Visualizations**: Log-log scale ELBO convergence plots and distance-to-convergence curves
- **Mathematical background**: Jensen's inequality sandwich bounds derivation
- **Benchmark results** (8000 iterations, E=10, lr=0.005, Adam):
  - **Convergence**: Lower Bound (6947) > Expanded (7022) > Simple (7592)
  - **Final ELBO**: Expanded (-47198.61) > Simple (-47322.61) > Lower Bound (-47543.07)
  - **Reconstruction error**: Expanded (0.241310) > Lower Bound (0.241432) > Simple (0.241737)
  - **Winner**: Lower Bound (fastest), Expanded (best ELBO)
- **Recommended usage**:
  - `mode='lower-bound'`: Large datasets, neural network/GP posteriors
  - `mode='expanded'`: Best final ELBO, standard applications (default)
  - `mode='simple'`: Debugging, baseline comparisons
- **Device**: MPS (Apple Silicon) with automatic detection (CUDA > MPS > CPU)

To run the benchmark locally:
```bash
# Standalone script
python benchmarks/simple_vs_expanded.py

# Jupyter notebook
jupyter notebook benchmarks/simple_vs_expanded.ipynb
```

### Read the Docs

The `.readthedocs.yaml` file configures automatic builds on Read the Docs:

1. Connect the GitHub repository to Read the Docs
2. Builds are triggered automatically on commits
3. Documentation is deployed to `https://pnmf.readthedocs.io/`

## Future Work

Potential improvements:
- Fix SGD optimizer divergence (add gradient clipping or per-parameter learning rates)
- Add more initialization methods (e.g., 'nndsvd', 'k-means')
- Implement online/mini-batch learning
- Add support for sparse matrices
- Include more unit tests
- Add benchmarking against sklearn NMF

## Recent Changes

### 2025-01-14: Separate expected log-likelihood from KL divergence in elbo.py

**What was changed:**
- Refactored `PNMF/elbo.py` to separate expected log-likelihood (modes) from KL divergence
- Renamed ELBO functions to expected log-likelihood functions:
  - `compute_elbo_simple()` → `compute_expected_log_lik_simple()`
  - `compute_elbo_expanded()` → `compute_expected_log_lik_expanded()`
  - `compute_elbo_lower_bound()` → `compute_expected_log_lik_lower_bound()`
- Added new dispatcher `compute_expected_log_lik()` for expected log-likelihood only
- Added `compute_kl_divergence()` as a separate function
- Updated `compute_elbo()` to accept optional `kl_fn` parameter for custom KL implementations
- Exported new functions in `__init__.py`

**Why it matters:**
- **Custom KL divergence**: Can now pass a custom KL function to `compute_elbo()`
- **Better modularity**: Expected log-likelihood and KL divergence are now independent
- **Future extensibility**: Easier to add custom KL implementations (e.g., for different priors)
- **Cleaner API**: Users can access individual components if needed

**Module structure:**
```python
# PNMF/elbo.py

# Helper
def poisson_log_likelihood(X, rate) -> Tensor

# Expected log-likelihood functions (modes)
def compute_expected_log_lik_simple(rate, X) -> Tensor
def compute_expected_log_lik_expanded(rate, qF, X, W) -> Tensor
def compute_expected_log_lik_lower_bound(qF, X, W) -> Tensor
def compute_expected_log_lik(mode, rate, qF, X, W) -> Tensor  # dispatcher

# KL divergence
def compute_kl_divergence(qF, pF) -> Tensor

# Full ELBO (combines expected log-lik - KL)
def compute_elbo(mode, rate, qF, pF, X, W, kl_fn=None) -> Tensor
```

**Usage with custom KL:**
```python
from PNMF import compute_elbo

# Custom KL function
def my_custom_kl(qF, pF):
    # Custom implementation
    return ...

# Use custom KL in ELBO computation
loss = compute_elbo(mode, rate, qF, pF, X, W, kl_fn=my_custom_kl)
```

### 2025-01-14: Refactor ELBO computation into separate module

**What was changed:**
- Created new `PNMF/elbo.py` module with all ELBO computation functions
- Extracted from `models.py`:
  - `poisson_log_likelihood()` - helper function
  - `compute_elbo_simple()` - full Monte Carlo ELBO
  - `compute_elbo_expanded()` - hybrid MC + analytic ELBO
  - `compute_elbo_lower_bound()` - fully analytic Jensen's bound ELBO
  - `compute_elbo()` - dispatcher function
- Updated `models.py` to import and use the new `elbo.py` module
- Removed ~180 lines of ELBO methods from `PNMF` class

**Why it matters:**
- **Better separation of concerns**: ELBO computation is now isolated from model architecture
- **Cleaner codebase**: `models.py` focuses on model structure, `elbo.py` on loss computation
- **Easier testing**: ELBO functions can be tested independently
- **Reusability**: ELBO functions can be used outside the PNMF class if needed

### 2025-01-03: Add Natural Gradient Training Mode (Commit `XXXXXXX`)

**What was changed:**
- Added `NaturalToMuS` autograd function to `custom_modules.py` for natural parameter conversion
- Added `NaturalGradientDescent` optimizer class to `models.py` implementing NGD for variational parameters
- Modified `GaussianPrior` class to support natural parameterization (`use_natural_gradients` parameter)
- Added `training_mode` parameter to `PNMF` class (`'standard'` or `'natural'`)
- Updated `fit()` method to use dual optimizers in natural mode:
  - NGD for variational parameters (θ₁, θ₂)
  - Adam/other for W parameters

**Why it matters:**
- **Better convergence**: Natural gradient mode achieves ~20% better ELBO than standard mode
- **Faster optimization**: NGD uses the Fisher information matrix for more efficient parameter updates
- **Theoretically sound**: Natural gradients follow the geometry of the variational distribution

**Usage:**
```python
# Standard training mode (default)
model_std = PNMF(n_components=5, training_mode='standard')

# Natural gradient training mode
model_nat = PNMF(n_components=5, training_mode='natural')
```

**Technical details:**
- **Natural parameterization**: Gaussian variational distribution parameterized by (θ₁, θ₂) instead of (μ, s):
  - θ₁ = μ/s² (natural parameter for mean)
  - θ₂ = -1/(2s²) (natural parameter for precision)
- **Natural gradient computation**: The `NaturalToMuS` autograd function computes gradients w.r.t. expectation parameters (η₁, η₂):
  - η₁ = μ
  - η₂ = s² + μ²
- **NGD optimizer**: Learning rate `lr=0.1` scaled by `1/num_data` as per natural gradient theory

**Benchmark results** (50 iterations, 5 components, 50 samples):
- **Standard + expanded**: ELBO = -5266.88
- **Standard + lower-bound**: ELBO = -6528.74
- **Natural + expanded**: ELBO = -4186.40 (**+21% improvement**)
- **Natural + lower-bound**: ELBO = -4989.78 (**+24% improvement**)

**Recommended usage:**
- **`training_mode='natural'`**: Better ELBO, faster convergence, recommended for most applications
- **`training_mode='standard'`**: Baseline comparison, simpler implementation

### 2025-01-03: Add Lower-Bound ELBO Mode with Jensen's Inequality (Commit `c802385`)

**What was changed:**
- Added `mode='lower-bound'`: Fully analytic ELBO using Jensen's lower bound
- Updated `mode='simple'` to use `torch.distributions.Poisson.log_prob()` directly
- Added `_elbo_lower_bound()` method with zero Monte Carlo sampling
- Updated benchmark to compare all three modes (simple, expanded, lower-bound)
- Added mathematical background with Jensen's inequality sandwich bounds to documentation
- Updated `docs/benchmarks.rst` with new results and recommended usage guide

**Why it matters:**
- **Fastest mode**: Lower-bound mode is ~1.09x faster than simple mode, ~1.01x faster than expanded mode
- **Zero variance**: No Monte Carlo sampling means fully deterministic gradients
- **Best for large-scale**: When q(F) comes from neural networks or GPs, this avoids expensive sampling
- **True lower bound**: Mathematically guaranteed lower bound on ELBO via Jensen's inequality

**Usage:**
```python
# Lower-bound mode (fastest, fully analytic)
model_lb = PNMF(n_components=5, mode='lower-bound')

# Expanded mode (best ELBO, default)
model_exp = PNMF(n_components=5, mode='expanded')

# Simple mode (full Monte Carlo)
model_simple = PNMF(n_components=5, mode='simple')
```

**Technical details:**
- Jensen's inequality provides sandwich bounds for the log-sum-exp term:
  ```
  log Σ W * exp(μ) ≤ E[log Σ W * exp(F)] ≤ log Σ W * exp(μ + σ²/2)
  ```
- Lower bound uses left inequality: `E[log Σ W * exp(F)] ≥ log Σ W * exp(E[F]) = log Σ W * exp(μ)`
- No MC sampling required - all computations are analytic

**Benchmark results** (8000 iterations, E=10, lr=0.005, Adam):
- **Convergence**: Lower Bound (6947 iter) > Expanded (7022) > Simple (7592)
- **Final ELBO**: Expanded (-47198.61, highest) > Simple (-47322.61) > Lower Bound (-47543.07)
- **Reconstruction error**: Expanded (0.241310, lowest) > Lower Bound (0.241432) > Simple (0.241737)

**Recommended usage:**
- **`mode='lower-bound'`**: Large datasets, neural network/GP posteriors, fast prototyping
- **`mode='expanded'`**: Best final ELBO, standard applications (default)
- **`mode='simple'`**: Debugging, baseline comparisons

### 2025-01-03: Add Optimizer Parameter and Improve Benchmark (Commit `7f587cc`)

**What was changed:**
- Added `optimizer` parameter to `PNMF` class (Adam, AdamW, NAdam, SGD, RMSprop)
- Added `return_history` parameter to `fit()` method (Keras-style API)
- Simplified benchmark code to use model's `fit()` method directly
- Updated benchmark to use 2000 iterations with Adam optimizer
- Generate integer count data via Poisson sampling (appropriate for Poisson model)
- Plot loss (-ELBO) on log scale instead of ELBO
- Added gradient clipping stability fix for scale parameter (clamp to >= 1e-8)
- Updated documentation with new benchmark results

**Why it matters:**
- Users can now choose between different optimizers (Adam works best, SGD diverges)
- Cleaner API: `history, model = model.fit(X, return_history=True)` like Keras
- Integer count data is more appropriate for the Poisson likelihood
- Loss on log scale better shows optimization trajectory

**Usage:**
```python
# With history tracking (Keras-style)
history, model = model.fit(X, return_history=True)

# Different optimizers
model_adam = PNMF(n_components=5, optimizer='Adam')      # Works well
model_nadam = PNMF(n_components=5, optimizer='NAdam')    # Works well
model_sgd = PNMF(n_components=5, optimizer='SGD')        # Diverges (needs fix)
```

**Technical details:**
- SGD diverges due to unstable gradients - needs gradient clipping or adaptive learning rate
- Scale parameter clamped to >= 1e-8 to prevent Normal distribution validation errors
- Benchmark now uses `exp(qF.mean) @ W.T` for proper reconstruction

**Known Issues:**
- **SGD optimizer** diverges - needs gradient clipping or per-parameter learning rates

### 2025-01-03: Add ELBO Mode Parameter and Benchmarks (Commit `1950650`)

**What was changed:**
- Added `mode` parameter to `PNMF` and `PoissonFactorization` classes
- Implemented `mode='simple'`: Full Monte Carlo ELBO estimation
- Implemented `mode='expanded'` (default): Hybrid Monte Carlo + analytic expectation
- Added `_elbo_simple()` and `_elbo_expanded()` methods with dispatcher `_elbo()`
- Created `benchmarks/` folder with comparison notebook and standalone script
- Added `nbsphinx` support to render Jupyter notebooks in documentation
- Added `docs/benchmarks.rst` page with embedded benchmark notebook

**Why it matters:**
- Users can now choose between two ELBO computation strategies
- `expanded` mode (default) has lower variance and typically converges faster
- `simple` mode is more straightforward and easier to understand
- Benchmark page allows direct comparison of convergence speed and final ELBO

**Usage:**
```python
# Simple mode (full Monte Carlo)
model = PNMF(n_components=5, mode='simple')

# Expanded mode (hybrid, default)
model = PNMF(n_components=5, mode='expanded')
```

**Technical details:**
- `mode='simple'`: All terms estimated via Monte Carlo
  - `log_lik = (X * log(rate) - rate - lgamma(X + 1)).mean()`
- `mode='expanded'`: Second term computed analytically
  - `E[exp(F)] = exp(μ + σ²/2)` (Gaussian moment-generating function)
  - Reduces variance in gradient estimates

### 2025-01-03: Add Poisson Normalization Constant (Commit `a7c872b`)

**What was changed:**
- Added `-torch.lgamma(X + 1).sum()` to the ELBO calculation in `PNMF/models.py`
- This implements the `-log(Y!)` term from the complete Poisson log-PMF
- Added `tqdm` progress bar for better user experience during training

**Why it matters:**
- The ELBO now includes the full Poisson normalization constant
- ELBO values are directly comparable to `torch.distributions.Poisson.log_prob()`
- Results can be properly compared across different implementations

**Technical details:**
```
log p(k|λ) = k*log(λ) - λ - log(k!)
                        ^^^^^^^^^^
                        Added this term
```

Using `torch.lgamma(X + 1)` since `lgamma(n+1) = log(n!)` by definition.

**Verification:**
```python
# Manual computation now matches torch.distributions exactly
log_prob_manual = X*torch.log(rate) - rate - torch.lgamma(X + 1)
log_prob_torch = torch.distributions.Poisson(rate).log_prob(X)
# log_prob_manual ≈ log_prob_torch (exact match)
```
