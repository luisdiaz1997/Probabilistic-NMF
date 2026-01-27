# Plan: General Transforms and Utility Functions for PNMF

## Overview

Add general transform functions and utility functions to PNMF that allow:
1. Conditioning on either F or W to learn the other
2. Easy extraction of latent factors (log-space, exp-space, with uncertainty)
3. More flexible inference scenarios

## Background

### Current Model Structure

The PNMF model factorizes data as:
```
X ≈ exp(F) @ W.T
```

Where:
- `X`: (N, D) input data matrix
- `F`: (N, L) latent factors (sample-specific, from variational distribution)
- `W`: (D, L) loadings matrix (learned positive parameters)

Internal representation uses transposed shapes:
- `Y`: (D, N) = X.T
- Rate: `W (D, L) @ exp(F) (L, N)`
- Variational distribution: `q(F) = Normal(μ, σ²)`

### Existing Transform

The current `transform(X, W)` method fixes W and finds F for new data using NNLS (non-negative least squares). This is not fully Bayesian.

## Proposed Changes

### New Module: `PNMF/transforms.py`

#### 1. Conditional Inference Functions

**`transform_W(Y_new, F, ...)`** - Learn new W conditioned on F
- Input: Y_new (N_new × D), F (N_new × L or N × L for subset)
- Output: W_new (D × L) as numpy array
- Use case: New data with fixed/similar latent factors

**`transform_F(Y_new, W, ...)`** - Learn new F conditioned on W
- Input: Y_new (N_new × D), W (D × L)
- Output: GaussianPrior object with learned q(F)
- Use case: New data with fixed loadings (full VI, better than NNLS)

#### 2. Factor Extraction Functions

**`log_factors(model)`** - Get log(F) means
- Returns: μ from q(F) = Normal(μ, σ²)
- Shape: (N, L) sklearn-style or (L, N) internal
- Use: Access latent factors in log-space

**`factors(model)`** - Get exp(F) values
- Returns: exp(μ) or E[exp(F)] = exp(μ + σ²/2)
- Shape: (N, L) sklearn-style
- Use: Reconstructed latent factors in original space (this is what `transform` returns)

**`factor_uncertainty(model)`** - Get F uncertainty
- Returns: σ (std dev) or σ² (variance) from q(F)
- Shape: (N, L) sklearn-style
- Use: Quantify uncertainty in latent factors

**`factor_samples(model, n_samples=100)`** - Sample from q(F)
- Returns: Multiple samples of F from variational posterior
- Shape: (n_samples, N, L)
- Use: Propagate uncertainty through downstream analysis

#### 3. Model Accessor Functions

**`get_loadings(model)`** - Get W matrix
- Returns: W (D × L) loadings
- Consistent interface regardless of internal representation

**`get_prior(model)`** - Get the GaussianPrior object
- Returns: The full GaussianPrior for advanced users

### Design Decisions

1. **Flexible input types**: Accept ndarray, torch.Tensor, or PNMF objects
2. **Consistent shapes**: Return sklearn-style (N, L) for user-facing functions
3. **Return types**:
   - Conditional inference: Return learned parameters (numpy or objects)
   - Factor extraction: Return numpy arrays by default, torch tensors optional
4. **Backward compatibility**: Keep existing `transform()` method, add new functions

## Implementation Steps

1. Create `PNMF/transforms.py` with:
   - `transform_W()` function
   - `transform_F()` function
   - `log_factors()` function
   - `factors()` function
   - `factor_uncertainty()` function
   - `factor_samples()` function
   - `get_loadings()` function
   - `get_prior()` function

2. Update `PNMF/__init__.py` to export new functions

3. Create `tests/test_transforms.py` with:
   - Tests for conditional inference functions
   - Tests for factor extraction functions
   - Tests for utility functions

4. Update `docs/api.rst` with new functions

5. Add usage examples to `docs/examples.rst`

## Files to Modify/Create

- `PNMF/transforms.py` (new)
- `PNMF/__init__.py` (update exports)
- `tests/test_transforms.py` (new)
- `docs/api.rst` (update)
- `docs/examples.rst` (update)
- `PLAN.md` (delete after merge)

## Example Usage

```python
from PNMF import PNMF, transform_W, transform_F
from PNMF.transforms import log_factors, exp_factors, factor_uncertainty, factor_samples
import numpy as np

# Fit original model
X = np.random.rand(100, 50)
model = PNMF(n_components=5).fit(X)

# === Factor Extraction ===
# Get latent factors in different forms
F_log = log_factors(model)           # μ from q(F), shape (100, 5)
F_exp = factors(model)               # exp(μ), shape (100, 5)
F_uncertainty = factor_uncertainty(model)  # σ, shape (100, 5)
F_samples = factor_samples(model, n_samples=100)  # (100, 100, 5)

# === Conditional Inference ===
# New data
X_new = np.random.rand(20, 50)

# Option 1: Learn new W conditioned on F
W_new = transform_W(X_new, F_log)    # (50, 5)

# Option 2: Learn new F conditioned on W (full VI)
W = get_loadings(model)              # (50, 5)
prior_new = transform_F(X_new, W)    # GaussianPrior object
F_new_log = log_factors_from_prior(prior_new)  # (20, 5)
```

## Notes

- The `factors()` function should compute `exp(μ + σ²/2)` using the Gaussian
  moment-generating function for the true expected value
- For `factor_samples()`, use `qF.rsample()` with reparameterization trick
- Consider adding `return_std` boolean to `factors()` for uncertainty
