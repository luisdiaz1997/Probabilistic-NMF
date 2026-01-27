# Plan: Batched Implementation for PNMF

## Overview

Implement stochastic variational inference (SVI) with mini-batch training for the PNMF library, following patterns established in GPzoo. This will enable scaling to large datasets by processing subsets of data per iteration instead of full-batch.

## Motivation

- Current implementation processes full dataset each iteration (memory intensive, slow for large N)
- GPzoo already has working batched implementation for similar Poisson factorization models
- Enables training on datasets larger than memory
- Faster iterations with stochastic gradient descent

## GPzoo Batching Pattern

**Key insight**: GPzoo uses **dual-dimension batching** for spatial transcriptomics data:

- **X (spatial coordinates)**: N samples, batched with `X_BATCH = 34000`
- **Y (gene expression)**: J genes (features) × N samples, batched with `Y_BATCH = 1000`
- **Batch sampling**: `torch.multinomial(torch.ones(N), num_samples=batch_size, replacement=False)`
- **Data indexing**: `Xb = X[idx]` and `yb = y[idy][:, idx]` - note the double indexing for Y

**Reference**: `/Users/luisfcd/gitclones/GPzoo/gpzoo/training_utilities.py:181-185`

For PNMF, our data is **X (N samples × M features)**, so we can batch on:
- **N (samples)**: Batch rows of the data matrix
- **M (features)**: Batch columns of the data matrix (similar to GPzoo's gene batching)

**Simplification**: For PNMF v1, we'll implement **sample-only batching** (N dimension) since:
- W (loadings) is D×L where D = features, typically manageable
- F (latent factors) is L×N where N = samples, can be very large
- Sample batching follows the standard SVI pattern

## Phase 1: API and Parameter Changes

### 1.1 Add New Parameters to PNMF Class

**File**: `PNMF/models.py` - `PNMF.__init__()`

Add parameters:
- `batch_size: int | None = None` - Size of mini-batches for **X (samples, N dimension)** (None = full batch, current behavior)
- `y_batch_size: int | None = None` - Size of mini-batches for **Y (features, M dimension)** (None = full batch, default)
- `shuffle: bool = True` - Whether to shuffle sample indices between epochs
- `n_epochs: int = 200` - Number of epochs (renamed from `max_iter` for clarity when batching)

**Note**:
- User-facing API: `batch_size` maps to internal `x_batch_size`
- Internally use `x_batch_size` and `y_batch_size` for clarity
- `y_batch_size=None` means all features (M) are included - this is the default

### 1.2 Update Fit Method Signature

**File**: `PNMF/models.py` - `PNMF.fit()`

Add support for:
- Batch iteration instead of full-batch processing
- Progress tracking per epoch (not per iteration)
- Return value compatible with batched training

## Phase 2: Batched Forward Pass Implementation

### 2.1 Implement Batched Rate Computation

**File**: `PNMF/models.py` - `PoissonFactorization.get_rate_batched()`

New method to compute rate for **sample batch indices**:
- Reference existing `GaussianPrior.forward_batched()` pattern from `priors.py:99-126`
- Accept `idx` (sample indices) as input
- Return rate tensor shape (E, D, batch_size) instead of (E, D, N)
- W (loadings) remains full-batch: (D, L) - all features included

### 2.2 Implement Batched Forward Method

**File**: `PNMF/models.py` - `PoissonFactorization.forward_batched_train()`

New method following GPzoo pattern:
- Reference: `/Users/luisfcd/gitclones/GPzoo/gpzoo/likelihoods.py:162-170` (NSF2.forward_batched_train)
- Accept `idx` (sample indices) and `y` (full data matrix for feature indexing)
- Call `qF, pF = self.prior.forward_train(X=None, idx=idx)` to get batched variational distributions
- Sample F: `F = qF.rsample((E,))` - shape (E, L, batch_size)
- Compute rate: `rate = self.W @ F` - shape (E, D, batch_size)
- Return (rate, qF, pF)

**Key GPzoo pattern from `likelihoods.py:164-170`**:
```python
qF, qU, pU = self.prior.forward_train(X=X, idx=idx, verbose=verbose, **kwargs)
F = qF.rsample((E,))
Z = self.get_rate(F, idy=idy)  # idy used for feature batching
```

### 2.3 ELBO Computation for Batches

**File**: `PNMF/elbo.py`

**No new functions needed!** The existing ELBO functions already work with batched inputs - just pass the batched tensors:

```
# Existing functions work as-is with batched inputs:
L1 = expected_log_likelihood(mode, rate_batch, qF_batch, X_batch, W_batch)
kl_batch = kl_divergence(qF_batch, pF_batch)
```

Where:
- `X_batch = X[idy][:, idx]` (batched data)
- `W_batch = W[idy]` if feature batching, else `W` (full)
- `qF_batch`, `pF_batch` from `prior.forward_batched(idx)`

**ELBO scaling formula** (done in training loop, not in elbo.py):
```
# Scale L1 by M if batching features (y_batch_size)
if y_batch_size is not None:
    L1 = L1 * (M / y_batch_size)

# Scale entire ELBO by N (sample batching)
elbo = (L1 - kl_batch) * (N / x_batch_size)
```

**Key insight**:
- `L1` (expected log-likelihood) is scaled by `y_batch_size` if feature batching is used
- The entire `(L1 - KL)` is scaled by `x_batch_size` for sample batching
- This matches GPzoo's pattern at `training_utilities.py:190-193`

## Phase 3: Training Loop Modification

### 3.1 Implement Batch Sampling

**File**: `PNMF/models.py` - New internal method `_get_batch_indices()`

Following GPzoo pattern exactly:
- Reference: `/Users/luisfcd/gitclones/GPzoo/gpzoo/training_utilities.py:181-182`

**Internal naming convention**:
- User's `batch_size` → internal `x_batch_size` (for samples N)
- User's `y_batch_size` → internal `y_batch_size` (for features M)

**GPzoo pattern**:
```python
# Line 181-182 in training_utilities.py
idx = torch.multinomial(torch.ones(N), num_samples=min(x_batch_size, N), replacement=False)
idy = torch.multinomial(torch.ones(J), num_samples=min(y_batch_size, J), replacement=False)
```

For PNMF:
- Always compute `idx` (sample indices)
- Compute `idy` (feature indices) only if `y_batch_size is not None`

### 3.2 Update Training Loop

**File**: `PNMF/models.py` - `PNMF.fit()`

Restructure from:
```
for iteration in range(max_iter):
    full forward pass
    full backward pass
```

To:
```
for epoch in range(max_epochs):
    for batch in batches:
        batch forward pass
        batch backward pass
    compute full ELBO for convergence check
```

Reference: `/Users/luisfcd/gitclones/GPzoo/gpzoo/training_utilities.py:188` (model call in loop)

## Phase 4: Optimizer Integration

### 4.1 Update Dual Optimizer Support

**File**: `PNMF/models.py` - Natural gradient mode with batching

Current dual optimizer setup (NGD for variational, Adam for W) needs:
- Ensure both optimizers work with batched gradients
- Reference existing NGD implementation
- No changes to NGD itself, just gradient accumulation pattern

### 4.2 Gradient Accumulation

If `batch_size` is smaller than dataset:
- Standard SGD: gradients naturally accumulate across batches
- NGD: ensure natural parameters update correctly with batched gradients

## Phase 5: Testing

### 5.1 Unit Tests

**File**: `tests/test_pnmf.py`

Add test classes:
- `TestBatchedPNMF` - Basic batched training tests
  - Test batch_size parameter works
  - Test final ELBO similar to full-batch
  - Test shuffle vs no-shuffle
- `TestBatchedELBOConvergence` - Verify batched converges to similar solution
  - Compare batched vs full-batch final ELBO
  - Verify reconstruction quality

### 5.2 Benchmark

**File**: `benchmarks/batched_vs_full.py`

New benchmark comparing:
- Training time per iteration
- Memory usage
- Final ELBO comparison
- Convergence speed

## Phase 6: Documentation

### 6.1 Update API Documentation

**File**: `docs/api.rst`

Document new parameters:
- `batch_size` usage and recommendations
- `shuffle` parameter
- `epoch_size` parameter

### 6.2 Update Usage Examples

**File**: `docs/examples.rst`

Add batched training example:
```python
# Large dataset batched training
model = PNMF(n_components=10, batch_size=1000, shuffle=True)
model.fit(X_large)
```

## Implementation Order

1. Phase 1: API changes (add parameters, backward compatible)
2. Phase 2: Batched forward methods (core computation)
3. Phase 3: Training loop (batch iteration)
4. Phase 5: Testing (verify correctness)
5. Phase 4: Optimizer integration (if needed, likely works out of box)
6. Phase 6: Documentation (update docs)

## Key Design Decisions

1. **Backward compatibility**: `batch_size=None` means full-batch (current behavior)
2. **API naming**: User's `batch_size` → internal `x_batch_size`, user's `y_batch_size` → internal `y_batch_size`
3. **Default batching**: `y_batch_size=None` (default) means all features included - only samples batched by default
4. **Optional feature batching**: `y_batch_size` enables feature batching for very wide datasets
5. **ELBO scaling**:
   - `L1` (expected log-likelihood) scaled by `M / y_batch_size` if batching features
   - Entire `(L1 - KL)` scaled by `N / x_batch_size` for sample batching
   - Formula: `elbo = (L1 * (M/y_batch_size if y_batch_size else 1) - KL) * (N / x_batch_size)`
6. **KL divergence**: Compute KL on batched qF vs pF (samples are independent, unlike GPzoo's GP prior)
7. **Convergence check**: Compute full ELBO periodically (not every batch iteration)
8. **Epoch vs iteration**: Report progress in epochs for batched mode
9. **API consistency**: Keep sklearn-compatible interface

## References

- GPzoo batched training: `/Users/luisfcd/gitclones/GPzoo/gpzoo/training_utilities.py:129`
- GPzoo batched likelihood: `/Users/luisfcd/gitclones/GPzoo/gpzoo/likelihoods.py:162-170`
- GPzoo batch sampling: `/Users/luisfcd/gitclones/GPzoo/gpzoo/training_utilities.py:181-182`
- Existing GaussianPrior.forward_batched(): `PNMF/priors.py:99-126`
- GPzoo SlideSeq config (batch sizes): `/Users/luisfcd/gitclones/GPzoo/gpzoo/datasets/slideseq/config.py:23-24`
