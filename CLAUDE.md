# PNMF Development Notes

## Development Workflow

**Important**: Every time we make a new plan or significant feature, we will create a new git branch. The PLAN.md file lives on the feature branch, not on main.

## Project Overview

This document describes the development of the PNMF (Probabilistic Non-negative Matrix Factorization) library, a pip-installable Python package with a scikit-learn compatible API using **variational inference**.

## What Was Done

### 1. Package Structure

```
PNMF/
├── models.py            # PoissonFactorization (PyTorch) + PNMF (sklearn API)
├── priors.py            # GaussianPrior class for variational inference
├── elbo.py              # Expected log-likelihood and ELBO computation
├── optimizers.py        # NaturalGradientDescent optimizer
├── custom_modules.py    # PositiveParameter, NaturalToMuS
├── transforms.py        # Factor extraction utilities
└── initialization.py    # W/F initialization methods

tests/
├── test_pnmf.py         # Core tests
├── test_transforms.py   # Transforms tests
├── test_spatial.py      # SVGP tests
├── test_spatial_training.py  # Spatial training tests
└── test_lcgp.py         # LCGP tests
```

### 2. Code Borrowed from GPzoo

Components adapted from [GPzoo](https://github.com/luisdiaz1997/GPzoo):
- `PositiveParameter`, `CholeskyParameter`, `LowRankPlusDiagonal`, `LowRankFactor` (modules.py)
- `GaussianPrior`, `SVGP`, `LCGP`, `MGGP_SVGP`, `MGGP_LCGP` (gp.py)
- `batched_Matern32`, `batched_MGGP_Matern32` (kernels.py)
- `mggp_kmeans_inducing_points()`, `kmeans_inducing_points()` (model_utilities.py)

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

### 4. Spatial Mode (SVGP and LCGP Priors)

When `spatial=True`, the latent factors F are modeled by a spatial Gaussian Process over spatial coordinates instead of an independent Gaussian prior. PNMF supports two spatial GP approximations:

1. **SVGP (Sparse Variational GP)**: Uses a subset of inducing points (M << N) with full Cholesky covariance. Good for N < 10,000.
2. **LCGP (Locally Conditioned GP)**: Uses ALL points as inducing (M = N) with low-rank + diagonal covariance and locally conditioned KL. Excellent for N > 10,000.

**Key idea**: Replace `GaussianPrior` (independent per-sample) with a spatial GP (`SVGP` or `LCGP`) while keeping the same `PoissonFactorization` likelihood and ELBO framework.

Both modes support:
- **Single-group**: Standard spatial smoothing without group structure (`multigroup=False`)
- **Multi-group (MGGP)**: Group-aware spatial smoothing via `batched_MGGP_Matern32` kernel (`multigroup=True`)

**Spatial API:**
```python
from PNMF import PNMF

model = PNMF(
    n_components=10,
    spatial=True,                # Enables GP prior
    # GP-specific parameters
    num_inducing=3000,          # M: number of inducing points
    lengthscale=1.0,            # Kernel lengthscale
    sigma=1.0,                  # Kernel output scale
    group_diff_param=10.0,      # MGGP group difference parameter
    jitter=1e-5,                # Numerical stability
    train_lengthscale=False,    # Freeze lengthscale (default)
    # Standard PNMF parameters
    mode='expanded',
    max_iter=500,
    learning_rate=0.01,
    batch_size=1000,            # Mini-batching for large N
    y_batch_size=500,           # Feature batching
)

# spatial=True requires coordinates and groups
history, model = model.fit(
    X,                          # (N, D) count matrix
    coordinates=coordinates,    # (N, 2) spatial coordinates - REQUIRED
    groups=groups,              # (N,) integer group codes - REQUIRED when multigroup=True
    return_history=True,
)

# Transform new data with spatial coordinates
transformed = model.transform(X_new, coordinates=coords_new, groups=groups_new)
```

**LCGP API (single-group, no groups):**
```python
from PNMF import PNMF
import numpy as np

# Generate spatial data (no groups needed)
N, D = 1000, 500
X = np.random.poisson(5, size=(N, D)).astype(np.float32)
coordinates = np.random.rand(N, 2).astype(np.float32) * 100

# LCGP without groups — uses LCGP + batched_Matern32
model = PNMF(
    n_components=10,
    spatial=True,
    local=True,              # Use LCGP (locally conditioned GP)
    multigroup=False,             # No groups needed
    # LCGP-specific parameters
    K=50,                        # 50 nearest neighbors for local conditioning
    rank=55,                      # Low-rank component rank (default: K+5)
    low_rank_mode='softplus',      # Constraint mode
    precompute_knn=True,           # Precompute KNN at init
    # Standard spatial parameters
    lengthscale=4.0,
    sigma=1.0,
    jitter=1e-5,
    # Standard PNMF parameters
    mode='expanded',
    max_iter=500,
    learning_rate=0.01,
    y_batch_size=500,
)

# No groups argument needed
history, model = model.fit(
    X,
    coordinates=coordinates,
    return_history=True,
)

# Transform (no groups needed)
transformed = model.transform(X, coordinates=coordinates)
```

**LCGP API (multi-group with groups):**
```python
# Generate spatial data with groups
groups = np.random.randint(0, 4, size=N)

# LCGP with groups — uses MGGP_LCGP + batched_MGGP_Matern32
model = PNMF(
    n_components=10,
    spatial=True,
    local=True,
    multigroup=True,              # Enable multi-group
    # LCGP-specific parameters
    K=50,
    rank=55,
    low_rank_mode='softplus',
    precompute_knn=True,
    # Standard spatial parameters
    lengthscale=4.0,
    sigma=1.0,
    group_diff_param=10.0,         # Controls group similarity (MGGP-specific)
    jitter=1e-5,
    # Standard PNMF parameters
    mode='expanded',
    max_iter=500,
    learning_rate=0.01,
    y_batch_size=500,
)

# groups argument required when multigroup=True
history, model = model.fit(
    X,
    coordinates=coordinates,
    groups=groups,
    return_history=True,
)

# Transform (groups required)
transformed = model.transform(X, coordinates=coordinates, groups=groups)
```

**SVGP vs LCGP Comparison:**
```python
# SVGP: Fewer inducing points, full Cholesky covariance
model_svgp = PNMF(spatial=True, local=False, multigroup=False, num_inducing=3000)

# LCGP: All points as inducing, locally conditioned KL (better for large N)
model_lcgp = PNMF(spatial=True, local=True, multigroup=False, K=50)
```

**Spatial Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `spatial` | `False` | Enable spatial GP prior |
| `local` | `False` | Use LCGP (all points as inducing) instead of SVGP. Only used when `spatial=True` |
| `kernel` | `'Matern32'` | Kernel function |
| `multigroup` | `False` | Use MGGP (multi-group GP) |
| `num_inducing` | `3000` | Number of inducing points M (SVGP only, ignored for LCGP) |
| `lengthscale` | `1.0` | Kernel lengthscale |
| `sigma` | `1.0` | Kernel output scale |
| `group_diff_param` | `10.0` | Group difference scaling (MGGP only) |
| `jitter` | `1e-5` | Numerical stability |
| `train_lengthscale` | `False` | Whether to train kernel lengthscale |
| `cholesky_mode` | `'exp'` | Cholesky diagonal constraint (SVGP only) |
| `diagonal_only` | `False` | Diagonal-only variational covariance (SVGP only) |
| `inducing_allocation` | `'proportional'` | How to distribute inducing points across groups (SVGP only) |
| `K` | `50` | Number of nearest neighbors for LCGP local conditioning (LCGP only) |
| `rank` | `None` | Rank of low-rank component for LCGP. If None, defaults to `min(M, K + 5)` (LCGP only) |
| `low_rank_mode` | `'softplus'` | Constraint mode for LCGP LowRankPlusDiagonal: `'softplus'` or `'exp'` (LCGP only) |
| `precompute_knn` | `True` | Whether to precompute KNN indices at initialization for LCGP (LCGP only) |

**Prior type derivation** (no `prior` parameter needed):
- `spatial=False` → `GaussianPrior`
- `spatial=True, local=False` → `SVGP`
- `spatial=True, local=True` → `LCGP`

**Architecture differences (GaussianPrior vs SVGP vs LCGP):**

| Aspect | GaussianPrior | SVGP (MGGP_SVGP) | LCGP (MGGP_LCGP) |
|--------|--------------|-----------|-----------|
| Parameters | mu (L,N), sigma (L,N) | mu (L,M), Lu (L,M,M), Z (M,2), kernel params | mu (L,M), Lu (L,M,R), Z (M,2), D (L,M), V (L,M,R), kernel params |
| Inducing points | N/A | Subset via k-means (M << N) | **ALL points** (M = N) |
| Covariance | Diagonal only | Full Cholesky (L, M, M) | Low-rank + diagonal: S = D + VV^T |
| Forward input | None (or idx) | coordinates (N,2), groups (N,) | coordinates (N,2), groups (N,) |
| Forward output | (qF, pF) | (qF, qU, pU) | (qF, qU, pU) |
| KL divergence | Gaussian KL, scales with N/batch_size | Whitened KL on inducing points, no N-scaling | **Locally conditioned KL** via K nearest neighbors |
| Mini-batch | Index into mu/sigma columns | Pass coordinate subset to GP forward | Pass coordinate subset + update KNN indices |
| transform() | NNLS multiplicative updates | GP predictive at new coordinates | GP predictive at new coordinates (same as SVGP) |
| Dependency | PyTorch only | PyTorch + GPzoo (lazy import) | PyTorch + GPzoo (lazy import) |
| Scalability | Good for small N | Good for N < 10,000 | **Excellent for N > 10,000** |
| Spatial resolution | Independent per sample | Limited by M inducing points | **Full resolution** (every point) |

**Key LCGP differences from SVGP:**
- **Inducing points = all data**: LCGP uses all N points as inducing (no subset selection)
- **Low-rank covariance**: Instead of full Cholesky (O(M²)), uses S = D + VV^T (O(MR))
- **Locally conditioned KL**: KL computed using only K nearest neighbors per point (O(MK²) vs O(M³))
- **Better scalability**: LCGP excels for large datasets (N > 10,000) with full spatial resolution

**LCGP GPzoo overrides** (in `gpzoo/gp.py`, class `LCGP`):

LCGP inherits SVGP → WSVGP but overrides three methods that WSVGP's defaults can't handle (because `Lu` is `LowRankPlusDiagonal`, not a simple tensor):
- **`apply_constraints()`** — Returns `(mu, None)`. WSVGP's version calls `self.Lu.data` which fails on `LowRankPlusDiagonal`. Returning `None` for Lu defers to `reshape_parameters()`.
- **`reshape_parameters()`** — Slices mu by KNN indices, calls `self.Lu.get_block(knn_idx)` to extract local K×K covariance blocks, then Cholesky-factors them. Follows the same pattern as VNNGP but uses `LowRankPlusDiagonal.get_block()` instead of raw Lu indexing.
- **`forward()`** — Calls `super(SVGP, self).forward()` (WSVGP's forward) then squeezes the extra dimension, matching VNNGP's pattern.

**KNN Convention:**
- **Training** (`knn_idz`): `calculate_knn(Z)[:, 1:]` — excludes FIRST column (self-match, since Z=X for LCGP)
- **Inference** (`knn_idx`): `calculate_knn(X)[:, :-1]` — excludes LAST column (keeps nearest K, since X→Z lookup may include self as nearest)
- Both yield exactly K neighbors. Training uses `knn_idz` for KL; inference uses `knn_idx` for forward.
- KNN is set in two inference paths:
  - `PNMF.transform()` in `models.py` — before calling `self._prior()`
  - `_get_spatial_qF()` in `transforms.py` — before calling `model._prior()` (used by `log_factors`, `get_factors`, `factor_uncertainty`, `factor_samples`)

**ELBO for spatial mode:**
```
ELBO = E[log p(Y|F)] - KL(q(U) || p(U))
```
- Expected log-likelihood uses the **same three modes** (simple/expanded/lower-bound) — the GP predictive qF is still Normal
- KL is the whitened KL on inducing points via `gp.kl_divergence(qU, pU)`, NOT scaled by N/batch_size
- `elbo.py` is unchanged — only the KL source differs

**Initialization:**
- **W**: Data-aware initialization shared by both spatial and non-spatial (`_initialize_W()`)
- **Inducing points Z**: K-means selection via `mggp_kmeans_inducing_points()` (critical for GP quality)
- **mu (inducing means)**: Random `N(0, 1)` scale (spatial uses `_create_spatial_prior()`, non-spatial uses `_initialize_mu_nonspatial()`)
- **Lu (Cholesky)**: Random diagonal initialization via `CholeskyParameter`


### 5. Key Features Implemented

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
- `loadings_mode`: `'projected'` (clamp after each step). Also: `'softplus'`, `'exp'`, `'multiplicative'`
- `mode`: `'expanded'` (hybrid Monte Carlo + analytic ELBO)
- `training_mode`: `'standard'` (standard gradient descent)
- `E`: 3 (Monte Carlo samples for ELBO, auto-set to 1 for lower-bound mode)
- `max_iter`: 200
- `tol`: 1e-4
- `learning_rate`: 0.01
- `optimizer`: `'Adam'` (Adam, AdamW, NAdam, SGD, RMSprop)
- `scheduler`: `'one_cycle'` (OneCycleLR with warmup)
- `init`: `'random'` (also: `'nndsvd'`, `'nndsvda'`, `'nndsvdar'`, `'k-means'`)

### 6. Installation

**PyPI package name:** `pnmf` (lowercase)
**Python import:** `from PNMF import PNMF` (uppercase)

### 7. License

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
- `forward(E=3)` returns (terms, qF, pF) for non-spatial, (terms, qF, qU, pU) for spatial
- Accepts `coordinates`, `groups`, `spatial` args to branch between GaussianPrior and GP forward

**PNMF** (`models.py`)
- sklearn-compatible wrapper
- Creates GaussianPrior (non-spatial), SVGP/MGGP_SVGP (spatial), or LCGP/MGGP_LCGP (spatial+local) internally
- Uses ELBO loss instead of NLL
- `training_mode` parameter: `'standard'` or `'natural'` (natural not supported with spatial)
- `spatial` parameter enables GP prior mode
- Key internal methods:
  - `_create_spatial_prior()` — builds SVGP/MGGP_SVGP or LCGP/MGGP_LCGP with kernel, inducing points, batched mu/Lu
  - `_initialize_W()` — shared W initialization for both spatial and non-spatial
  - `_initialize_mu_nonspatial()` — variational mean init for non-spatial models
  - `_create_optimizer()` — optimizer factory (replaces duplicated code)
- LCGP inference paths (`transform()`, `_get_spatial_qF()`) set KNN indices before calling `forward()` (see KNN Convention below)

### PositiveParameter Class

A sophisticated parameter class that:
- Stores an unconstrained `_raw` parameter internally
- Applies transformations (`softplus`, `exp`, or `projected`) to enforce positivity
- Provides a tensor-like interface for easy manipulation
- Supports projection via `project()` method for constrained optimization

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
- **TestSpatialValidation**: Spatial parameter validation (requires coords, groups, etc.)
- **TestSpatialFit**: Basic spatial fitting (multigroup, single-group, batching)
- **TestSpatialTransform**: Spatial transform and fit_transform
- **TestSpatialFactorExtraction**: Factor extraction functions with spatial models
- **TestSpatialTraining**: Training integration tests for spatial mode (convergence, ELBO modes)
- **TestLCGPValidation**: LCGP parameter validation (K, rank, low_rank_mode, etc.)
- **TestLCGPFit**: LCGP fitting (no-groups, multigroup, ELBO modes, custom rank, etc.)
- **TestLCGPBatching**: LCGP with sample/feature/both mini-batching
- **TestLCGPTransform**: LCGP transform, fit_transform, new coordinates
- **TestLCGPFactorExtraction**: Factor extraction with LCGP (log_factors, get_factors, uncertainty, samples, loadings)
- **TestLCGPNoGroups**: LCGP without groups (fit, transform, factor extraction, all-points-as-inducing verification)

Spatial and LCGP tests require `gpzoo` and are auto-skipped if not installed (`@pytest.mark.skipif`).

## Documentation

The project uses **Sphinx** with **Read the Docs** theme. Build locally with:
```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs/ docs/_build/html
```

**Benchmark notebook**: `benchmarks/simple_vs_expanded.ipynb` compares all three ELBO modes. Run with:
```bash
python benchmarks/simple_vs_expanded.py  # standalone
jupyter notebook benchmarks/simple_vs_expanded.ipynb  # interactive
```

**Recommended ELBO modes:**
- `mode='expanded'`: Best final ELBO, standard applications (default)
- `mode='lower-bound'`: Large datasets, neural network/GP posteriors (fastest)
- `mode='simple'`: Debugging, baseline comparisons

## Future Work

Potential improvements:
- Fix SGD optimizer divergence (add gradient clipping or per-parameter learning rates)
- Add support for sparse matrices
- Add benchmarking against sklearn NMF
- GP regression-based μ initialization for spatial mode
- Support additional GP kernels beyond Matern32
- Train kernel hyperparameters (currently sigma and group_diff_param are frozen)
