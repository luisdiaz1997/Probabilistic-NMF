# PLAN: SVGP Prior for PNMF

## Overview

Add **SVGP (Sparse Variational Gaussian Process)** as an alternative prior to `GaussianPrior` in PNMF. When `spatial=True`, the latent factors F are modeled by a GP over spatial coordinates instead of an independent Gaussian. We use the **MGGP** (Multi-Group GP) variant with `batched_MGGP_Matern32` kernel from GPzoo, enabling group-aware spatial smoothing.

**Key idea**: Replace `GaussianPrior` (independent per-sample) with `MGGP_SVGP` (spatially correlated, group-aware) while keeping the same `PoissonFactorization` likelihood and ELBO framework.

---

## New API

### Default (non-spatial, unchanged)

```python
from PNMF import PNMF

model = PNMF(n_components=10, random_state=42)
transformed = model.fit_transform(X)
```

### Spatial mode

```python
from PNMF import PNMF

model = PNMF(
    n_components=10,
    spatial=True,                # Enables GP prior
    prior='SVGP',               # Default when spatial=True
    kernel='Matern32',          # Default kernel
    multigroup=True,            # Default: True (uses MGGP)
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
    random_state=42,
    verbose=True,
    batch_size=1000,            # Required for spatial (large N)
    y_batch_size=500,           # Feature batching
)

# spatial=True requires coordinates and groups
history, model = model.fit(
    X,                          # (N, D) count matrix
    coordinates=coordinates,    # (N, 2) spatial coordinates - REQUIRED when spatial=True
    groups=groups,              # (N,) integer group codes - REQUIRED when multigroup=True
    return_history=True,
)
```

### Parameter Summary

| Parameter | Default | When | Description |
|-----------|---------|------|-------------|
| `spatial` | `False` | always | Enable spatial GP prior |
| `prior` | `'GaussianPrior'` | always | Prior type. `'SVGP'` when `spatial=True` |
| `kernel` | `'Matern32'` | `spatial=True` | Kernel function |
| `multigroup` | `True` | `spatial=True` | Use MGGP (multi-group GP) |
| `num_inducing` | `3000` | `spatial=True` | Number of inducing points M |
| `lengthscale` | `1.0` | `spatial=True` | Kernel lengthscale |
| `sigma` | `1.0` | `spatial=True` | Kernel output scale |
| `group_diff_param` | `10.0` | `multigroup=True` | Group difference scaling |
| `jitter` | `1e-5` | `spatial=True` | Numerical stability |
| `train_lengthscale` | `False` | `spatial=True` | Whether to train kernel lengthscale |
| `cholesky_mode` | `'exp'` | `spatial=True` | Cholesky diagonal constraint |
| `diagonal_only` | `False` | `spatial=True` | Diagonal-only variational covariance |
| `inducing_allocation` | `'proportional'` | `multigroup=True` | How to distribute inducing points across groups |

### Validation Rules

- `spatial=True` requires `coordinates` in `fit()`
- `multigroup=True` requires `groups` in `fit()`
- `prior='SVGP'` requires `spatial=True`
- When `spatial=True` and `prior` not explicitly set, defaults to `'SVGP'`
- When `spatial=True` and `multigroup` not explicitly set, defaults to `True`

---

## Architecture

### Current Flow (non-spatial)

```
fit(X):
  X (N, D) -> transpose to Y (D, N)
  -> GaussianPrior(Y, L)                    # independent q(F) ~ N(mu, sigma)
  -> PoissonFactorization(prior, Y, L)
  -> Training loop:
       qF, pF = prior()                     # (L, N) distributions
       F ~ qF.rsample((E,))                 # (E, L, N)
       rate = W @ exp(F)                     # (E, D, N)
       ELBO = E[log p(Y|F)] - KL(q||p)
```

### New Flow (spatial)

```
fit(X, coordinates, groups):
  X (N, D) -> transpose to Y (D, N)
  coords (N, 2) -> torch tensor
  groups (N,) -> torch LongTensor

  -> Create kernel: batched_MGGP_Matern32(sigma, lengthscale, group_diff_param, n_groups)
  -> Create GP:     MGGP_SVGP(kernel, dim=2, M=num_inducing, n_groups=n_groups, jitter=jitter)
  -> Set inducing:  gp.Z = Z, gp.groupsZ = groupsZ  (via kmeans or random selection)
  -> Init Lu:       CholeskyParameter batched (L, M, M), initialized from kernel
  -> Init mu:       gp.mu (L, M) from Lu decomposition

  -> PoissonFactorization(gp, Y, L)          # same likelihood, different prior!

  -> Training loop:
       qF, qU, pU = gp(X=coords_batch, groupsX=groups_batch)
                                              # qF: (L, N_batch) from GP predictive
       F ~ qF.rsample((E,))                  # (E, L, N_batch)
       rate = W @ exp(F)                      # (E, D, N_batch)
       exp_ll = E[log p(Y|F)]                # same ELBO modes (simple/expanded/lower-bound)
       kl = gp.kl_divergence(qU, pU)         # GP KL, NOT Gaussian KL
       loss = kl - exp_ll
```

### Key Architectural Difference

| Aspect | GaussianPrior | MGGP_SVGP |
|--------|--------------|-----------|
| Parameters | mu (L,N), sigma (L,N) | mu (L,M), Lu (L,M,M), Z (M,2), kernel params |
| Forward input | None (or idx) | X (coords), groupsX |
| Forward output | (qF, pF) | (qF, qU, pU) |
| qF shape | (L, N) | (L, N_batch) via GP predictive equations |
| KL divergence | `kl_divergence(qF, pF)` analytic Gaussian | `gp.kl_divergence(qU, pU)` whitened KL |
| Spatial structure | None (independent) | Matern32 kernel with group modulation |
| Mini-batch | Index into mu/sigma columns | Pass coordinate subset to GP forward |

---

## Implementation Stages

### Stage 1: Add spatial parameters to PNMF constructor

**File: `PNMF/models.py` (PNMF class `__init__`)**

Add the new parameters to `__init__`:

```python
def __init__(
    self,
    # ... existing params ...
    # New spatial parameters
    spatial: bool = False,
    prior: str = 'GaussianPrior',
    kernel: str = 'Matern32',
    multigroup: bool = True,
    num_inducing: int = 3000,
    lengthscale: float = 1.0,
    sigma: float = 1.0,
    group_diff_param: float = 10.0,
    jitter: float = 1e-5,
    train_lengthscale: bool = False,
    cholesky_mode: str = 'exp',
    diagonal_only: bool = False,
    inducing_allocation: str = 'proportional',
):
```

Add validation rules to `_validate_params()`.

Store all as instance attributes.

**Reference**: Current constructor at `PNMF/models.py:255-291`

---

### Stage 2: Add `coordinates` and `groups` to `fit()`

**File: `PNMF/models.py` (PNMF class `fit` method)**

Update `fit()` signature:

```python
def fit(self, X, y=None, coordinates=None, groups=None, return_history=False):
```

Add validation:
- If `spatial=True`, require `coordinates` (numpy array or tensor, shape (N, 2))
- If `multigroup=True`, require `groups` (numpy array or tensor, shape (N,))
- Convert both to torch tensors on correct device

Store as `self._coordinates`, `self._groups`.

**Reference**: Current fit() method at `PNMF/models.py:349+`

---

### Stage 3: Create the GP prior (new method `_create_spatial_prior`)

**File: `PNMF/models.py` (new method on PNMF class)**

This is the core new code. We need a method that:

1. Creates the kernel
2. Selects inducing points (and inducing groups) via kmeans - **this is the critical init**
3. Creates the GP with those inducing points
4. Freezes lengthscale if `train_lengthscale=False`
5. Returns the GP object (ready to be used as `prior` in `PoissonFactorization`)

**mu and Lu use random init** (same as current PNMF's GaussianPrior). W uses the existing `init` parameter (default `'random'`). The only thing that matters for initialization is getting good inducing point locations and their group assignments.

**Code to borrow from GPzoo:**

| Step | GPzoo Source | What to copy |
|------|-------------|-------------|
| Kernel creation | `gpzoo/kernels.py:115-171` | `batched_MGGP_Matern32` class (import, don't copy) |
| GP creation | `gpzoo/gp.py:924-987` | `MGGP_SVGP = MGGPWrapper(SVGP)` (import) |
| Inducing point selection | `gpzoo/model_utilities.py:182-287` | `mggp_kmeans_inducing_points()` (import) |

**Pseudocode:**

```python
def _create_spatial_prior(self, Y, coordinates, groups):
    """Create MGGP_SVGP prior for spatial mode."""
    from gpzoo.kernels import batched_MGGP_Matern32, batched_Matern32
    from gpzoo.gp import MGGP_SVGP, SVGP
    from gpzoo.modules import CholeskyParameter
    from gpzoo.model_utilities import mggp_kmeans_inducing_points

    D, N = Y.shape
    L = self.n_components
    coords = coordinates  # (N, 2) tensor
    n_groups = int(groups.max().item() + 1)

    # 1. Create kernel
    if self.multigroup:
        kernel = batched_MGGP_Matern32(
            sigma=self.sigma, lengthscale=self.lengthscale,
            group_diff_param=self.group_diff_param, n_groups=n_groups
        )
    else:
        kernel = batched_Matern32(sigma=self.sigma, lengthscale=self.lengthscale)

    # 2. Select inducing points (THE critical initialization)
    M = min(self.num_inducing, N)
    if self.multigroup:
        Z, groupsZ = mggp_kmeans_inducing_points(
            coords, groups, M, seed=self.random_state or 123,
            allocation=self.inducing_allocation,
        )
    else:
        Z = coords[:M].clone()
        groupsZ = None

    # 3. Create GP (mu and Lu stay at random defaults from SVGP.__init__)
    if self.multigroup:
        gp = MGGP_SVGP(
            kernel, dim=2, M=M, n_groups=n_groups,
            jitter=self.jitter, cholesky_mode=self.cholesky_mode,
            diagonal_only=self.diagonal_only
        )
        gp.Z = nn.Parameter(Z, requires_grad=False)
        gp.groupsZ = nn.Parameter(groupsZ, requires_grad=False)
    else:
        gp = SVGP(
            kernel, dim=2, M=M,
            jitter=self.jitter, cholesky_mode=self.cholesky_mode,
            diagonal_only=self.diagonal_only
        )
        gp.Z = nn.Parameter(Z, requires_grad=False)

    # 4. Batch mu and Lu for L factors
    #    SVGP.__init__ creates mu (M,) and Lu (M, M) for a single output.
    #    We need (L, M) and (L, M, M) for L latent factors.
    del gp.Lu
    gp.Lu = CholeskyParameter((L, M), mode=self.cholesky_mode, diagonal_only=self.diagonal_only)
    gp.mu = nn.Parameter(torch.randn(L, M) * 0.01)

    # 5. Freeze kernel lengthscale if not training it
    if not self.train_lengthscale:
        kernel.lengthscale.requires_grad = False

    return gp
```

**Reference**: Pattern from `gpzoo/models/nsf.py:376-443` (`MGGP_SVGP_NSF.__init__`)

---

### Stage 4: Modify `fit()` to branch on `spatial`

**File: `PNMF/models.py` (PNMF class `fit` method)**

The fit method currently:
1. Creates `GaussianPrior`
2. Creates `PoissonFactorization`
3. Runs training loop

We need to branch at step 1:

```python
# In fit():
if self.spatial:
    self._prior = self._create_spatial_prior(Y_torch, coords_torch, groups_torch)
else:
    self._prior = GaussianPrior(Y_torch, L=self.n_components, ...)
```

The `PoissonFactorization` already accepts any prior via `prior` argument (it just calls `prior()` or `prior.forward_batched(idx)`). **But** for SVGP, the forward signature is different:

- `GaussianPrior.forward()` returns `(qF, pF)`
- `SVGP.forward(X=coords)` returns `(qF, qU, pU)`

This means `PoissonFactorization.forward()` needs to handle both cases.

---

### Stage 5: Adapt PoissonFactorization for GP priors

**File: `PNMF/models.py` (PoissonFactorization class)**

The current `PoissonFactorization.forward()` calls:
```python
qF, pF = self.prior.forward_batched(idx)  # or self.prior()
```

For SVGP, the call is:
```python
qF, qU, pU = self.prior(X=coords_batch, groupsX=groups_batch)
```

**Options:**

**Option A (recommended):** Add a `spatial` flag and coordinate/group arguments to `PoissonFactorization.forward()`:

```python
def forward(self, idx=None, idy=None, E=10, X=None, coordinates=None, groups=None, spatial=False):
    if spatial:
        if hasattr(self.prior, 'forward_train'):
            # VNNGP/LCGP path (not needed for SVGP but future-proof)
            qF, qU, pU = self.prior.forward_train(X=coordinates, idx=idx, groupsX=groups)
        else:
            # SVGP path
            qF, qU, pU = self.prior(X=coordinates, groupsX=groups)
        # Return qU, pU for KL computation later
    else:
        # Existing path
        if idx is not None:
            qF, pF = self.prior.forward_batched(idx)
        else:
            qF, pF = self.prior()
```

**Option B:** Create a wrapper/adapter that makes SVGP look like GaussianPrior. More complex, less transparent.

We'll go with **Option A** since it's more explicit.

**Key change**: The return value changes from `(terms, qF, pF)` to `(terms, qF, qU_or_pF, pU_or_None)` when spatial, or we can always return a consistent tuple with `None` for non-spatial.

---

### Stage 6: Adapt the training loop for GP KL divergence

**File: `PNMF/models.py` (PNMF class `fit` method, training loop)**

The current training loop does:
```python
terms, qF, pF = self._model.forward(idx, idy, E=self.E, X=X_batch)
exp_ll, kl = compute_elbo(self.mode, terms, qF, pF, X_batch)
```

For spatial mode, KL is computed differently:
- **Non-spatial**: `KL(q(F) || p(F))` via `torch.distributions.kl_divergence`
- **Spatial (SVGP)**: Whitened KL on inducing points via `gp.kl_divergence(qU, pU)`

**Reference**: The whitened KL is computed in `gpzoo/gp.py` (BaseVGP.kl_divergence) and `gpzoo/utilities.py:254` (`whitened_KL`)

```python
# In training loop:
if self.spatial:
    terms, qF, qU, pU = self._model.forward(
        idx, idy, E=self.E, X=X_batch,
        coordinates=coords_batch, groups=groups_batch, spatial=True
    )
    exp_ll = expected_log_likelihood(self.mode, terms, X_batch)
    kl = self._prior.kl_divergence(qU, pU)
else:
    terms, qF, pF = self._model.forward(idx, idy, E=self.E, X=X_batch)
    exp_ll, kl = compute_elbo(self.mode, terms, qF, pF, X_batch)
```

---

### Stage 7: Handle mini-batching for spatial mode

**File: `PNMF/models.py` (PNMF class `fit` method)**

For spatial mode, mini-batching works differently:
- `idx` selects a subset of N samples -> need to pass `coordinates[idx]` and `groups[idx]` to GP
- `idy` selects a subset of D features -> same as before (index into W and Y)

```python
# In training loop, when spatial=True:
if self.batch_size is not None:
    idx = torch.randperm(N)[:self.batch_size].to(device)
    coords_batch = self._coordinates[idx]
    groups_batch = self._groups[idx]
    X_batch = Y_torch[:, idx] if idy is None else Y_torch[idy][:, idx]
else:
    coords_batch = self._coordinates
    groups_batch = self._groups
    X_batch = Y_torch
```

**Note**: For SVGP (not VNNGP), mini-batching of samples is straightforward because the GP predictive equations work for any subset of coordinates. We don't need `forward_train` or precomputed KNN - just `gp(X=coords_batch, groupsX=groups_batch)`.

---

### Stage 8: Handle `transform()` for spatial mode

**File: `PNMF/models.py` or `PNMF/transforms.py`**

For spatial models, `transform(X_new)` needs new coordinates too:

```python
def transform(self, X, coordinates=None, groups=None):
    if self.spatial:
        # Use GP predictive equations for new coordinates
        with torch.no_grad():
            qF, _, _ = self._prior(X=coordinates, groupsX=groups)
            return torch.exp(qF.mean).T.cpu().numpy()  # (N_new, L)
    else:
        # Existing NNLS-based transform
        ...
```

**Reference**: `get_groupwise_factors()` pattern from `GPzoo/notebooks/liver_mggp_healthy_matern32_umap_init.ipynb` (cell 41)

---

### Stage 9: Factor extraction for spatial mode

**File: `PNMF/transforms.py`**

Functions like `log_factors()`, `get_factors()`, `factor_uncertainty()` need to work with GP priors.

For `GaussianPrior`, factors come from `prior.mean` (L, N).
For `SVGP`, factors come from `gp(X=coordinates)` which returns `qF` with `.mean` and `.scale`.

```python
def log_factors(model, coordinates=None, groups=None, return_tensor=False):
    if model.spatial:
        with torch.no_grad():
            qF, _, _ = model._prior(X=coordinates, groupsX=groups)
            mu = qF.mean  # (L, N)
        result = mu.T  # (N, L)
    else:
        # Existing code
        ...
```

Alternatively, we could cache the last forward pass results so factor extraction doesn't require re-running the GP.

---

### Stage 10: Update `__init__.py` exports

**File: `PNMF/__init__.py`**

No new exports needed initially since all changes are behind parameters on existing classes. But we may want to export:
- The GP prior object for advanced users
- Any new utility functions

---

### Stage 11: Update `_validate_params()`

**File: `PNMF/models.py`**

Add validation for:
```python
if self.spatial:
    if self.prior not in ['SVGP']:
        raise ValueError("When spatial=True, prior must be 'SVGP'")
    if self.kernel not in ['Matern32']:
        raise ValueError("kernel must be 'Matern32'")
    if self.training_mode == 'natural':
        raise ValueError("Natural gradient training not supported with spatial priors")
    if self.inducing_allocation not in ['proportional', 'equal']:
        raise ValueError("inducing_allocation must be 'proportional' or 'equal'")
```

---

### Stage 12: Tests

**File: `tests/test_spatial.py` (new file)**

```python
class TestSpatialPNMF:
    """Test PNMF with spatial=True and SVGP prior."""

    def test_fit_spatial(self):
        """Basic spatial fit."""
        X = np.random.poisson(5, (100, 50)).astype(np.float32)
        coords = np.random.randn(100, 2).astype(np.float32)
        groups = np.random.randint(0, 3, 100)

        model = PNMF(n_components=3, spatial=True, max_iter=5, num_inducing=50, lengthscale=1.0)
        model.fit(X, coordinates=coords, groups=groups)

        assert model.components_.shape == (3, 50)
        assert model.elbo_ is not None

    def test_spatial_requires_coordinates(self):
        """spatial=True without coordinates should raise."""
        model = PNMF(spatial=True)
        with pytest.raises(ValueError, match="coordinates"):
            model.fit(X)

    def test_multigroup_requires_groups(self):
        """multigroup=True without groups should raise."""
        model = PNMF(spatial=True, multigroup=True)
        with pytest.raises(ValueError, match="groups"):
            model.fit(X, coordinates=coords)

    def test_transform_spatial(self):
        """Transform new data with spatial prior."""
        model = PNMF(n_components=3, spatial=True, max_iter=5, num_inducing=50, lengthscale=1.0)
        model.fit(X_train, coordinates=coords_train, groups=groups_train)
        transformed = model.transform(X_test, coordinates=coords_test, groups=groups_test)
        assert transformed.shape == (n_test, 3)

    def test_factor_extraction_spatial(self):
        """Factor extraction functions work with spatial models."""
        model = PNMF(n_components=3, spatial=True, max_iter=5, num_inducing=50, lengthscale=1.0)
        model.fit(X, coordinates=coords, groups=groups)
        F = log_factors(model, coordinates=coords, groups=groups)
        assert F.shape == (100, 3)

    def test_spatial_with_batching(self):
        """Spatial mode with mini-batching."""
        model = PNMF(
            n_components=3, spatial=True, max_iter=5,
            num_inducing=50, batch_size=30, y_batch_size=20, lengthscale=1.0
        )
        model.fit(X, coordinates=coords, groups=groups)
        assert model.components_.shape == (3, 50)
```

---

## Code References

### GPzoo Files to Import From

| What | Path | Import |
|------|------|--------|
| MGGP_SVGP | `gpzoo/gp.py:985` | `from gpzoo.gp import MGGP_SVGP` |
| SVGP | `gpzoo/gp.py:453` | `from gpzoo.gp import SVGP` |
| batched_MGGP_Matern32 | `gpzoo/kernels.py:115` | `from gpzoo.kernels import batched_MGGP_Matern32` |
| batched_Matern32 | `gpzoo/kernels.py:29` | `from gpzoo.kernels import batched_Matern32` |
| CholeskyParameter | `gpzoo/modules.py:194` | `from gpzoo.modules import CholeskyParameter` |
| mggp_kmeans_inducing_points | `gpzoo/model_utilities.py:182` | `from gpzoo.model_utilities import mggp_kmeans_inducing_points` |
| whitened_KL | `gpzoo/utilities.py:254` | Used internally by `gp.kl_divergence()` |

### GPzoo Patterns to Follow

| Pattern | Source | Notes |
|---------|--------|-------|
| MGGP_SVGP_NSF constructor | `gpzoo/models/nsf.py:376-443` | Reference for MGGP+SVGP+NSF setup (we simplify by using random mu/Lu) |
| Inducing point selection | `gpzoo/model_utilities.py:182-287` | `mggp_kmeans_inducing_points()` - critical init |
| NSF2 forward (spatial) | `gpzoo/likelihoods.py:142-149` | `gp(X=coords, groupsX=groups)` -> `qF.rsample((E,))` |
| NSF2 forward_batched | `gpzoo/likelihoods.py:152-160` | Batched spatial forward |
| MGGP forward_kernels | `gpzoo/gp.py:946-972` | Group-aware kernel computation |
| Groupwise factors | Liver notebook cell 41 | `get_groupwise_factors()` |

### PNMF Files to Modify

| File | What Changes |
|------|-------------|
| `PNMF/models.py` | PNMF `__init__`, `fit()`, `transform()`, `_validate_params()`, `PoissonFactorization.forward()` |
| `PNMF/transforms.py` | `log_factors()`, `get_factors()`, `factor_uncertainty()`, `transform_F()` |
| `PNMF/__init__.py` | Possibly new exports |
| `tests/test_spatial.py` | New test file |

---

## Data Flow Diagram

### Non-spatial (current)

```
X (N, D) numpy
    |
    v
Y = X.T (D, N) torch
    |
    v
GaussianPrior(Y, L=10)
  mu: (L, N) nn.Parameter
  scale: (L, N) PositiveParameter
    |
    v
prior() -> (qF, pF)
  qF = N(mu, scale)    shape (L, N)
  pF = N(0, 1)         shape (L, N)
    |
    v
PoissonFactorization
  W: (D, L) PositiveParameter
  rate = W @ exp(F)    shape (E, D, N)
    |
    v
ELBO = E[log p(Y|F)] - KL(qF || pF)
```

### Spatial (new)

```
X (N, D) numpy          coordinates (N, 2) numpy       groups (N,) numpy
    |                         |                              |
    v                         v                              v
Y = X.T (D, N) torch    coords (N, 2) torch            groups (N,) LongTensor
    |                         |                              |
    |                    +----|------------------------------+
    |                    |
    |                    v
    |              batched_MGGP_Matern32(sigma, lengthscale, group_diff_param, n_groups)
    |                    |
    |                    v
    |              MGGP_SVGP(kernel, dim=2, M=3000, n_groups)
    |                Z: (M, 2)        inducing locations  (kmeans init - critical)
    |                groupsZ: (M,)    inducing groups     (kmeans init - critical)
    |                mu: (L, M)       variational mean    (random init)
    |                Lu: (L, M, M)    variational Cholesky (random init)
    |                    |
    |                    v
    |              gp(X=coords, groupsX=groups) -> (qF, qU, pU)
    |                qF = N(mu_pred, sigma_pred)    shape (L, N)
    |                qU = MVN(mu_u, Lu @ Lu.T)      shape (L, M)
    |                pU = MVN(0, Kzz)               shape (L, M)
    |                    |
    v                    v
PoissonFactorization
  W: (D, L) PositiveParameter
  rate = W @ exp(F)    shape (E, D, N)
    |
    v
exp_ll = E[log p(Y|F)]          (same three modes: simple/expanded/lower-bound)
kl     = KL_whitened(qU || pU)   (whitened KL on inducing points, from gpzoo)
ELBO   = exp_ll - kl
```

---

## GP Forward Pass Details

### What `MGGP_SVGP.forward(X, groupsX)` does internally

```
Input: X (N, 2), groupsX (N,)

1. Compute kernels (batched via vmap):
   Kxx = kernel(X, X, groupsX, groupsX, diag=True)     # (L, N) diagonal
   Kzx = kernel(Z, X, groupsZ, groupsX)                 # (L, M, N)
   Kzz = kernel(Z, Z, groupsZ, groupsZ)                 # (L, M, M)

2. Cholesky decomposition:
   L_zz = cholesky(Kzz + jitter * I)                     # (L, M, M)

3. Get constrained variational parameters:
   mu, Lu = apply_constraints()                           # (L, M), (L, M, M)

4. Transform (non-whitened SVGP):
   mu_t, Lu_t = transform_variables(mu, Lu, L_zz)
   # Solves: L_zz @ mu_t = mu  =>  mu_t = L_zz^{-1} @ mu
   # Same for Lu_t

5. Compute predictive distribution:
   W = solve(L_zz, Kzx)                                  # (L, M, N) weights
   mean = W.T @ mu_t                                      # (L, N)
   var  = Kxx - sum(W^2, dim=M) + sum((W.T @ Lu_t)^2)   # (L, N)
   qF = Normal(mean, sqrt(var))

6. Compute inducing distribution:
   qU = MultivariateNormal(mu_t, Lu_t @ Lu_t.T)
   pU = None  (whitened prior is implicit)

Return: (qF, qU, pU)
```

### KL Divergence for SVGP

The whitened KL is:
```
KL = 0.5 * (-2 * sum(log(diag(Lu))) + sum(Lu^2) + sum(mu^2) - M)
```

Computed by `BaseVGP.kl_divergence()` in `gpzoo/gp.py`.

This is summed over all L factors.

---

## ELBO Computation

### Non-spatial (unchanged)

```
ELBO = E[log p(Y|F)] - KL(q(F) || p(F))
```

Where `KL(q(F) || p(F))` is the standard Gaussian KL with shape (L, N), summed.

### Spatial

```
ELBO = E[log p(Y|F)] - KL(q(U) || p(U))
```

Where:
- `E[log p(Y|F)]` uses the **same three modes** (simple/expanded/lower-bound) as non-spatial.
  The `qF` distribution from the GP is still Normal, so all the same sampling and MGF computations work.
- `KL(q(U) || p(U))` is the whitened KL on inducing points (shape (L, M), summed).

**This means `elbo.py` does NOT need to change.** The expected log-likelihood functions work with any Normal `qF`. Only the KL computation changes, and that's handled by the GP object.

---

## Mini-batch Scaling for Spatial

For non-spatial PNMF, the current scaling in `fit()` is:
```python
if y_batch_size:
    exp_ll = exp_ll * (D / y_batch_size)
if batch_size:
    exp_ll = exp_ll * (N / batch_size)
    kl = kl * (N / batch_size)
```

For spatial mode:
- `exp_ll` scaling is the same (scales with both D and N batch sizes)
- `kl` does **NOT** scale with N/batch_size because the KL is over inducing points (M), not data points (N). The inducing KL is computed in full regardless of batch size.

```python
if self.spatial:
    if y_batch_size:
        exp_ll = exp_ll * (D / y_batch_size)
    if batch_size:
        exp_ll = exp_ll * (N / batch_size)
    # KL is over inducing points, NOT data - no scaling needed
else:
    # Existing scaling
    ...
```

**Reference**: This matches the NSF training pattern in `GPzoo/gpzoo/datasets/slideseq/svgp_nsf.py` where KL is not scaled by batch size.

---

## Dependencies

The SVGP feature requires GPzoo as a dependency:

```toml
# In pyproject.toml [project.optional-dependencies]
spatial = [
    "gpzoo @ git+https://github.com/luisdiaz1997/GPzoo.git",
]
```

Or for development:
```bash
pip install -e ../GPzoo
```

GPzoo imports should be **lazy** (inside the method that creates the spatial prior) so that the base PNMF package works without GPzoo installed:

```python
def _create_spatial_prior(self, ...):
    try:
        from gpzoo.gp import MGGP_SVGP
        from gpzoo.kernels import batched_MGGP_Matern32
        ...
    except ImportError:
        raise ImportError(
            "GPzoo is required for spatial mode. "
            "Install with: pip install -e path/to/GPzoo"
        )
```

---

## File Checklist

### Files to Modify

- [ ] `PNMF/models.py` - PNMF class: `__init__`, `fit()`, `transform()`, `_validate_params()`, `_create_spatial_prior()` (new)
- [ ] `PNMF/models.py` - PoissonFactorization class: `forward()` (add spatial path)
- [ ] `PNMF/transforms.py` - Update factor extraction functions for spatial
- [ ] `PNMF/__init__.py` - Any new exports
- [ ] `pyproject.toml` - Add optional `spatial` dependency

### Files to Create

- [ ] `tests/test_spatial.py` - Tests for spatial mode

### Files NOT Changed

- [ ] `PNMF/elbo.py` - No changes needed (works with any Normal qF)
- [ ] `PNMF/priors.py` - No changes needed (GaussianPrior unchanged)
- [ ] `PNMF/custom_modules.py` - No changes needed
- [ ] `PNMF/optimizers.py` - No changes needed (NGD not used with spatial)
- [ ] `PNMF/initialization.py` - No changes needed (W uses existing `init` parameter)

---

## Implementation Order

1. **Stage 1-2**: Add parameters to constructor and fit() (scaffold only, no functionality)
2. **Stage 3**: Implement `_create_spatial_prior()` (the core GP setup)
3. **Stage 4-5**: Modify fit() to use GP prior, adapt PoissonFactorization.forward()
4. **Stage 6-7**: Adapt training loop (KL, scaling, batching)
5. **Stage 8-9**: Adapt transform() and factor extraction
6. **Stage 10-12**: Exports, validation, tests

---

## Summary

| What | Non-spatial (current) | Spatial (new) |
|------|-----------------------|--------------|
| Prior | `GaussianPrior` | `MGGP_SVGP` from GPzoo |
| Prior params | mu (L,N), sigma (L,N) | mu (L,M), Lu (L,M,M), Z (M,2), kernel |
| fit() input | X only | X + coordinates + groups |
| Forward | `prior()` -> (qF, pF) | `gp(X=coords, groupsX=groups)` -> (qF, qU, pU) |
| KL | Gaussian KL on (L,N) | Whitened KL on (L,M) |
| KL scaling | Scales with N/batch_size | No scaling (over inducing pts) |
| ELBO modes | All three work | All three work (same qF type) |
| elbo.py changes | None | None |
| transform() | NNLS multiplicative | GP predictive at new coordinates |
| Mini-batch | Index into mu/sigma | Pass coord subset to GP |
| Dependency | PyTorch only | PyTorch + GPzoo |
