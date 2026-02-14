# PLAN: LCGP Integration into PNMF

**Branch:** `LCGP`
**Goal:** Add LCGP (Locally Conditioned GP) as an alternative spatial prior in PNMF, alongside the existing SVGP prior. Support both single-group (`multigroup=False`) and multi-group (`multigroup=True`) variants.

---

## Background

### What is LCGP?

LCGP (Locally Conditioned Gaussian Process) is a scalable GP variant from GPzoo that replaces the standard SVGP's full Cholesky covariance with a **low-rank plus diagonal** parameterization:

```
S = D + VV^T
```

Instead of computing KL divergence over all M inducing points (O(M^3)), LCGP uses a **locally conditioned KL** that only considers K nearest neighbors per point (O(M * K^2)), making it far more scalable for large spatial datasets.

### Why LCGP?

| Aspect | Current SVGP (PNMF) | LCGP (new) |
|--------|---------------------|------------|
| **Inducing points** | Subset via k-means (M << N) | All points (M = N) |
| **Covariance params** | Cholesky: O(L * M^2) | Low-rank+diag: O(L * M * R) |
| **KL computation** | Full whitened KL: O(M^3) | **Local KL: O(M * K^2)** — only K neighbors per point |
| **Scalability** | Good for N < 10,000 | Excellent for N > 10,000 |
| **Spatial resolution** | Limited by M | Full resolution (every point) |
| **Key advantage** | Sparse inducing points | **Locally conditioned** — conditions on K neighbors, so can use all points |
| **multigroup=False** | SVGP + batched_Matern32 | LCGP + batched_Matern32 |
| **multigroup=True** | MGGP_SVGP + batched_MGGP_Matern32 | MGGP_LCGP + batched_MGGP_Matern32 |

### Single-Group vs Multi-Group Support

LCGP will support both variants, controlled by the existing `multigroup` parameter:

| Setting | GPzoo GP Class | GPzoo Kernel | groups arg | Reference File |
|---------|---------------|-------------|------------|----------------|
| `multigroup=False` | `LCGP` | `batched_Matern32` | Not required | `lcgp_nsf.py` |
| `multigroup=True` | `MGGP_LCGP` | `batched_MGGP_Matern32` | Required in `fit()` | `lcgp_mggp_nsf.py` |

- **`multigroup=False`**: Standard LCGP with a single spatial kernel. No group assignments needed. Uses `LCGP` from `gpzoo/gp.py:744` and `batched_Matern32` kernel.
- **`multigroup=True`**: Multi-group LCGP with group-aware kernel modulation via `group_diff_param`. Requires `groups` array in `fit()`. Uses `MGGP_LCGP` from `gpzoo/gp.py:987` (= `MGGPWrapper(LCGP)`) and `batched_MGGP_Matern32` kernel.

This mirrors the existing SVGP behavior where `multigroup=False` uses `SVGP` + `batched_Matern32` and `multigroup=True` uses `MGGP_SVGP` + `batched_MGGP_Matern32`.

### Reference Implementation

The GPzoo codebase already has LCGP fully implemented. The reference usage is:
- **`GPzoo/gpzoo/datasets/slideseq/run_all.sh`** — Launches `lcgp_nsf.py` and `lcgp_mggp_nsf.py`
- **`GPzoo/gpzoo/datasets/slideseq/lcgp_nsf.py`** — Single-group LCGP training script (`multigroup=False` equivalent)
- **`GPzoo/gpzoo/datasets/slideseq/lcgp_mggp_nsf.py`** — Multi-group LCGP training script (`multigroup=True` equivalent)

---

## Detailed Architecture Analysis

### GPzoo Class Hierarchy for LCGP

```
BaseVGP (gp.py)
  └── WSVGP
        └── SVGP
              └── LCGP (gp.py:744-920)
                    └── MGGP_LCGP = MGGPWrapper(LCGP) (gp.py:987)
```

### Key GPzoo Components to Use

1. **`LCGP` class** (`gpzoo/gp.py:744-920`)
   - Extends SVGP with low-rank+diagonal covariance
   - `forward_train()` (line 791): Returns marginal q(U_j) = N(mu_j, sqrt(s_jj)) where s_jj = D_j + ||V_j||^2
   - `kl_divergence_full()` (line 838): Locally conditioned KL using K nearest neighbors
   - `calculate_knn()` (line 770): FAISS-based KNN search

2. **`MGGP_LCGP`** (`gpzoo/gp.py:987`)
   - Created via `MGGPWrapper(LCGP)` — adds multi-group kernel support

3. **`LowRankPlusDiagonal`** (`gpzoo/modules.py:393-575`)
   - Parameterizes S = D + VV^T
   - `D`: PositiveParameter of shape (L, M) — diagonal
   - `V`: LowRankFactor of shape (L, M, R) — low-rank factor
   - `get_precision_components()` (line 455): Woodbury identity for efficient precision
   - `get_conditional_params()` (line 477): Computes tau_tilde_sq and alpha for KL
   - `get_block()` (line 441): Extracts S[neighbors, neighbors] for KL

4. **`LowRankFactor`** (`gpzoo/modules.py:337-391`)
   - Unconstrained factor V with optional semi-orthogonal projection

5. **NSF Model Wrappers** (`gpzoo/models/nsf.py`)
   - `LCGP_NSF` (line 551-674): Single-group wrapper
   - `MGGP_LCGP_NSF` (line 676-799): Multi-group wrapper
   - Both show how to initialize LowRankPlusDiagonal and set up KNN

6. **Training Functions** (`gpzoo/training_utilities.py`)
   - `train_lcgp_batched_with_tracking()` (line 666-803): Single-group training loop
   - `train_mggp_lcgp_with_tracking()` (line 943-1084): Multi-group training loop

---

## Implementation Plan (Simplified)

### Phase 1: Add LCGP Parameters to PNMF Constructor

**File:** `PNMF/models.py` — `PNMF.__init__()` (line 354)

**Changes:**
- Add new constructor parameters alongside existing spatial params:
  - `K`: int, default=50 — Number of nearest neighbors for LCGP local conditioning
  - `rank`: Optional[int], default=None — Rank of low-rank component (defaults to min(M, K+5))
  - `low_rank_mode`: str, default='softplus' — Constraint mode for LowRankPlusDiagonal
  - `precompute_knn`: bool, default=True — Whether to precompute KNN indices at initialization
- Store new params: `self.K`, `self.rank`, `self.low_rank_mode`, `self.precompute_knn`
- Add `self._knn_idx = None` to stored attributes

**Reference:** `GPzoo/gpzoo/datasets/slideseq/lcgp_nsf.py:21-76` (create_model function shows all LCGP params)

### Phase 2: Add Parameter Validation for LCGP

**File:** `PNMF/models.py` — `PNMF._validate_params()` (line 454)

**Changes:**
- **Remove auto-set logic**: `prior='GaussianPrior'` with `spatial=True` should raise ERROR (not auto-correct to SVGP)
- Extend `prior` validation: when `spatial=True`, allow `'SVGP'` or `'LCGP'`
- Add LCGP-specific validation:
  - `K` must be >= 1
  - `rank` must be >= 1 if specified
  - `low_rank_mode` must be in `['softplus', 'exp']`
  - When `local=True`: natural gradient training not supported (same as SVGP)
  - When `local=True`: `num_inducing` is ignored (LCGP uses all points as inducing)
  - When `local=True` and `multigroup=True`: `groups` will be required in `fit()`

### Phase 3: Extend `_create_spatial_prior()` to Handle LCGP

**File:** `PNMF/models.py` — `PNMF._create_spatial_prior()` (line 574)

**Changes:** Add `if self.local:` branch (no new method needed - follow same pattern as SVGP)

**API approach:** Use `local` parameter:
- **Default**:
  - `spatial=False` → `GaussianPrior` (default)
  - `spatial=True, local=False` → `SVGP` (default spatial prior)
  - `spatial=True, local=True` → `LCGP` (locally conditioned)
- **Optional**: `prior` can be explicitly set to override auto-detection

**Implementation approach:**
1. Import from GPzoo (lazy imports, same pattern as `_create_spatial_prior()`):
   ```
   from gpzoo.gp import LCGP, MGGP_LCGP
   from gpzoo.kernels import batched_Matern32, batched_MGGP_Matern32
   from gpzoo.modules import LowRankPlusDiagonal
   ```

2. Create kernel (same as current SVGP path — reuse kernel creation logic from `_create_spatial_prior()`)

3. **Inducing points**: Use ALL data points as inducing points (Z = coordinates). This is the key difference from SVGP.
   - Reference: `GPzoo/gpzoo/models/nsf.py:612-613` — `Z = X.clone()`
   - M = N (number of data points)

4. **Compute rank**: `rank = self.rank if self.rank is not None else min(M, self.K + 5)`
   - Reference: `GPzoo/gpzoo/models/nsf.py:617-618`

5. **Create GP object** (branching on `self.multigroup`):
   - **`multigroup=False`**: `LCGP(kernel, M=M, jitter=self.jitter, K=self.K, rank=rank, low_rank_mode=self.low_rank_mode)`
     - Uses `batched_Matern32` kernel (no group info)
     - Reference: `GPzoo/gpzoo/models/nsf.py:610,620`
   - **`multigroup=True`**: `MGGP_LCGP(kernel, M=M, n_groups=n_groups, jitter=self.jitter, K=self.K, rank=rank, low_rank_mode=self.low_rank_mode)`
     - Uses `batched_MGGP_Matern32` kernel with `group_diff_param` and `n_groups`
     - Reference: `GPzoo/gpzoo/models/nsf.py:738-742,754`

6. **Set inducing points**:
   - Both: `gp.Z = nn.Parameter(Z, requires_grad=False)`
   - **`multigroup=True` only**: `gp.groupsZ = nn.Parameter(groupsZ, requires_grad=False)`
   - Reference: `GPzoo/gpzoo/models/nsf.py:621, 755-756`

7. **Initialize LowRankPlusDiagonal** (batch for L factors):
   - Delete default Lu and replace with batched version
   - Initialize V (low-rank) to small random values `randn(L, M, R) * 0.01`
   - Initialize mu to `randn(L, M) * 0.1` (will be overwritten by data-aware init later)
   - Reference: `GPzoo/gpzoo/models/nsf.py:643-673` (`_init_lu_for_lcgp` method)

8. **Precompute KNN indices**:
   - `knn_idx = gp.calculate_knn(coordinates)[:, :-1]` (exclude self)
   - Store on model: `gp.knn_idx = knn_idx` and `gp.knn_idz = knn_idx` (since Z = X, knn_idx == knn_idz)
   - Reference: `GPzoo/gpzoo/models/nsf.py:638-639`

9. **Freeze kernel hyperparameters** (same logic as SVGP path)

### Phase 4: Modify `fit()` Training Loop for LCGP

**File:** `PNMF/models.py` — `PNMF.fit()` (line 739)

**Changes to the initialization section (~lines 828-856):**
- Store the full KNN indices if LCGP: `self._knn_idx = self._prior.knn_idz.clone()`

**Changes to the training loop (lines 982-1010):**

1. **KNN index management per batch** (before forward pass):
   ```python
   if self.prior == 'LCGP':
       knn_idx = self._knn_idx[idx] if idx is not None else self._knn_idx
       self._prior.knn_idx = knn_idx
       self._prior.knn_idz = knn_idx
   ```
   - Reference: `GPzoo/gpzoo/training_utilities.py:710-733`

2. **KL divergence** (replace SVGP's KL for LCGP):
   ```python
   if self.prior == 'LCGP':
       # LCGP's locally conditioned KL (no qU/pU needed, no N/M scaling)
       kl = self._prior.kl_divergence_full(qZ=None, idx=idx)
   else:
       # SVGP's standard KL
       kl = self._prior.kl_divergence(qU, pU).sum() * (N / M)
   ```
   - Reference: `GPzoo/gpzoo/training_utilities.py:740`

3. **No changes needed for:**
   - Forward pass: `self._prior()` works for both SVGP and LCGP
   - Expected log-likelihood: `compute_log_likelihood_terms()` works with any Normal qF
   - Loss computation: `loss = kl - exp_ll` (same structure)

### Phase 5: Modify `transform()` for LCGP

**File:** `PNMF/models.py` — `PNMF.transform()` (line 1119)

**Changes:** None required — LCGP's `forward()` inherits from SVGP and works identically for prediction.

### Phase 6: Update `transforms.py` for LCGP

**File:** `PNMF/transforms.py`

**Changes:** None required — LCGP's `forward()` returns `(qF, qU, pU)` just like SVGP, same interface.

### Phase 7: Optimizer Setup for LCGP

**File:** `PNMF/models.py` — `PNMF.fit()` optimizer section (~lines 861-933)

**Changes:** None required — use same single LR approach as SVGP. PNMF's OneCycleLR scheduler handles warmup.

**Note:** GPzoo uses separate LRs per param group, but that's optimization, not required for v1.

### Phase 8: Add Tests

**File:** `tests/test_lcgp.py` (new file)

**Test classes to create:**

1. **TestLCGPValidation** — Parameter validation
   - Test that `local=True` with `spatial=True` is accepted
   - Test that `local=True` without `spatial=True` raises error
   - Test that `K < 1` raises error
   - Test that invalid `low_rank_mode` raises error
   - Test that `training_mode='natural'` with LCGP raises error

2. **TestLCGPFit** — Basic fitting
   - Test LCGP fit with `multigroup=False` (no groups needed), small synthetic data (N=50, D=20, L=3, K=5)
   - Test LCGP fit with `multigroup=True` (groups required), same data + groups array
   - Test that ELBO is finite and not NaN for both variants
   - Test that components_ has correct shape for both variants
   - Test with batched training (batch_size, y_batch_size) for both variants
   - Test all three ELBO modes (simple, expanded, lower-bound) for both variants

3. **TestLCGPTransform** — Transform and fit_transform
   - Test transform returns correct shape with `multigroup=False`
   - Test transform returns correct shape with `multigroup=True`
   - Test fit_transform returns correct shape for both variants
   - Test transform at new coordinates for both variants

4. **TestLCGPFactorExtraction** — Integration with transforms.py
   - Test `log_factors()` with LCGP model
   - Test `get_factors()` with LCGP model
   - Test `factor_uncertainty()` with LCGP model
   - Test `factor_samples()` with LCGP model

**Test data generation:**
```python
N, D, L = 50, 20, 3
X = np.random.poisson(5, size=(N, D)).astype(np.float32)
coordinates = np.random.rand(N, 2).astype(np.float32) * 50
groups = np.random.randint(0, 3, size=N)
```

**Reference:** Existing `tests/test_spatial.py` and `tests/test_spatial_training.py` for patterns.

### Phase 9: Update Documentation

**Files:** `CLAUDE.md`

**Changes:**
- Add LCGP section to "Spatial Mode" documentation
- Document new parameters: `K`, `rank`, `low_rank_mode`, `precompute_knn`
- Add LCGP vs SVGP comparison table
- Document the LCGP API usage example

**Note:** No changes needed for `pyproject.toml` (dependencies already present) or `__init__.py` (no new exports).

**File:** `PNMF/__init__.py`

**No changes expected** — the LCGP functionality is accessed through the existing `PNMF` class with `spatial=True, local=True`. No new public classes or functions need exporting.

### Phase 11: Update CLAUDE.md

**File:** `CLAUDE.md`

**Changes:**
- Add LCGP section to "Spatial Mode" documentation
- Document new parameters: `K`, `rank`, `low_rank_mode`, `precompute_knn`
- Add LCGP vs SVGP comparison table
- Document the LCGP API usage example
- Add architecture differences table (LCGP vs SVGP vs GaussianPrior)

---

## Key Design Decisions

### 1. LCGP as a prior, not a separate class

LCGP is added as `local=True` within the existing `PNMF` class, not as a new top-level class. This keeps the API simple:

```python
# SVGP (existing spatial prior)
model = PNMF(spatial=True, local=False, num_inducing=3000)

# LCGP (new spatial prior - locally conditioned)
model = PNMF(spatial=True, local=True, K=50, rank=55)
```

### 2. multigroup=True vs multigroup=False

Both variants are supported, reusing the existing `multigroup` parameter from the SVGP implementation:

- **`multigroup=False`** (default): Creates `LCGP` GP with `batched_Matern32` kernel. No `groups` argument needed in `fit()` or `transform()`. This is the simpler case — pure spatial correlation without group structure.
  - Reference: `GPzoo/gpzoo/datasets/slideseq/lcgp_nsf.py` and `GPzoo/gpzoo/models/nsf.py:551-674` (LCGP_NSF)

- **`multigroup=True`**: Creates `MGGP_LCGP` GP with `batched_MGGP_Matern32` kernel. `groups` is required in `fit()` and `transform()`. The kernel modulates spatial correlation by group similarity via `group_diff_param`.
  - Reference: `GPzoo/gpzoo/datasets/slideseq/lcgp_mggp_nsf.py` and `GPzoo/gpzoo/models/nsf.py:676-799` (MGGP_LCGP_NSF)

The branching logic in `_create_lcgp_prior()` mirrors `_create_spatial_prior()` — the only difference is which GP class and kernel are instantiated. Everything downstream (LowRankPlusDiagonal init, KNN precomputation, training loop, KL divergence) is identical for both.

### 3. Inducing points = all data points (both variants)

Unlike SVGP which selects a subset via k-means, LCGP uses **all N data points** as inducing points. This is fundamental to the LCGP approach — the efficiency comes from the low-rank covariance and local KL, not from reducing inducing points.

Reference: `GPzoo/gpzoo/models/nsf.py:612-613`

### 3. forward_train() vs forward() distinction

LCGP has two forward modes:
- **`forward_train()`**: Returns marginal q(U_j) directly from stored parameters. Very fast, O(M*R). Used during training.
- **`forward()`** (inherited from SVGP): Full GP predictive equations. More expensive but works at new coordinates. Used for transform().

Reference: `GPzoo/gpzoo/gp.py:791-836`

### 4. KL divergence is not N-scaled for LCGP

Unlike SVGP where we scale KL by N/M (since KL is over M << N inducing points), LCGP's KL is computed over all M=N points. When mini-batching, we scale by N/batch_size to account for the subset.

Reference: `GPzoo/gpzoo/training_utilities.py:740` — KL is called with `idx=idx` for the batch

### 5. KNN precomputation

KNN indices are computed once at initialization (using FAISS) and stored. During mini-batch training, we index into the stored KNN indices for each batch.

Reference: `GPzoo/gpzoo/training_utilities.py:710` and `GPzoo/gpzoo/models/nsf.py:638-639`

---

## Implementation Order

1. **Phase 1-2**: Constructor params + validation (quick, sets up the interface)
2. **Phase 3**: Extend `_create_spatial_prior()` to handle LCGP (core creation logic)
3. **Phase 4**: Training loop modifications (KNN management, KL computation)
4. **Phase 5-7**: transform(), transforms.py, optimizer setup (no changes needed - LCGP works like SVGP)
5. **Phase 8**: Tests (write alongside implementation)
6. **Phase 9**: Documentation updates (CLAUDE.md)

---

## API Examples

### LCGP without groups (multigroup=False)

```python
from PNMF import PNMF
import numpy as np

# Generate spatial data (no groups needed)
N, D = 1000, 500
X = np.random.poisson(5, size=(N, D)).astype(np.float32)
coordinates = np.random.rand(N, 2).astype(np.float32) * 100

# Single-group LCGP — uses LCGP + batched_Matern32
model = PNMF(
    n_components=10,
    spatial=True,
    local=True,
    multigroup=False,              # No groups (default)
    # LCGP-specific parameters
    K=50,                          # 50 nearest neighbors
    rank=55,                       # Low-rank component rank (default: K+5)
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

### LCGP with groups (multigroup=True)

```python
# Generate spatial data with groups
groups = np.random.randint(0, 4, size=N)

# Multi-group LCGP — uses MGGP_LCGP + batched_MGGP_Matern32
model = PNMF(
    n_components=10,
    spatial=True,
    local=True,
    multigroup=True,               # Enable multi-group
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

### LCGP vs SVGP Comparison

```python
# SVGP without groups: fewer inducing points, full Cholesky covariance
model_svgp = PNMF(spatial=True, local=False, multigroup=False, num_inducing=3000)

# LCGP without groups: all points as inducing, locally conditioned KL
model_lcgp = PNMF(spatial=True, local=True, multigroup=False, K=50)

# SVGP with groups
model_svgp_mg = PNMF(spatial=True, local=False, multigroup=True, num_inducing=3000)

# LCGP with groups: all points as inducing, locally conditioned KL
model_lcgp_mg = PNMF(spatial=True, local=True, multigroup=True, K=50)
```

---

## Risk Assessment

### Low Risk
- Constructor params and validation (Phase 1-2): Straightforward additions
- Tests (Phase 8): Standard test patterns

### Medium Risk
- Extending `_create_spatial_prior()` (Phase 3): Needs careful initialization of `LowRankPlusDiagonal`
- Training loop modifications (Phase 4): KNN index management per batch, correct KL computation

### High Risk
- None — simplified approach minimizes code changes by reusing existing patterns

### Mitigation
- Start with small synthetic data (N=50) for development/debugging
- Compare ELBO trajectories between PNMF LCGP and GPzoo LCGP_NSF on same data
- Use lower-bound mode first (fully analytic, no MC noise) for debugging
