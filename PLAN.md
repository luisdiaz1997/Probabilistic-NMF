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
| **KL computation** | Full whitened KL: O(M^3) | Local KL: O(M * K^2) |
| **Scalability** | Good for N < 10,000 | Excellent for N > 10,000 |
| **Spatial resolution** | Limited by M | Full resolution (every point) |
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

## Implementation Plan

### Phase 1: Add LCGP Parameters to PNMF Constructor

**File:** `PNMF/models.py` — `PNMF.__init__()` (line 354)

**Changes:**
- Add new constructor parameters alongside existing spatial params:
  - `prior`: Extend valid values from `['GaussianPrior', 'SVGP']` to include `'LCGP'`
  - `K`: int, default=50 — Number of nearest neighbors for LCGP local conditioning
  - `rank`: Optional[int], default=None — Rank of low-rank component (defaults to min(M, K+5))
  - `diag_mode`: str, default='softplus' — Diagonal constraint mode for LowRankPlusDiagonal
  - `scale_multiplier`: float, default=1e-6 — Initial scale for diagonal D component
  - `precompute_knn`: bool, default=True — Whether to precompute KNN indices at initialization
- When `spatial=True` and `prior='LCGP'`: auto-set `prior_type = 'LCGP'`
- Store new params: `self.K`, `self.rank`, `self.diag_mode`, `self.scale_multiplier`, `self.precompute_knn`
- Add `self._knn_idx = None` to stored attributes

**Reference:** `GPzoo/gpzoo/datasets/slideseq/lcgp_nsf.py:21-76` (create_model function shows all LCGP params)

### Phase 2: Add Parameter Validation for LCGP

**File:** `PNMF/models.py` — `PNMF._validate_params()` (line 454)

**Changes:**
- Extend `prior_type` validation: when `spatial=True`, allow `'SVGP'` or `'LCGP'`
- Add LCGP-specific validation:
  - `K` must be >= 1
  - `rank` must be >= 1 if specified
  - `diag_mode` must be in `['softplus', 'exp']`
  - `scale_multiplier` must be > 0
  - When `prior='LCGP'`: natural gradient training not supported (same as SVGP)
  - When `prior='LCGP'`: `num_inducing` is ignored (LCGP uses all points as inducing)
  - When `prior='LCGP'` and `multigroup=True`: `groups` will be required in `fit()`

### Phase 3: Create LCGP Prior

**File:** `PNMF/models.py` — New method `_create_lcgp_prior()` (after `_create_spatial_prior()`, ~line 671)

**What it does:** Creates an LCGP or MGGP_LCGP GP object with proper initialization.

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
   - **`multigroup=False`**: `LCGP(kernel, M=M, jitter=self.jitter, K=self.K, rank=rank, diag_mode=self.diag_mode)`
     - Uses `batched_Matern32` kernel (no group info)
     - Reference: `GPzoo/gpzoo/models/nsf.py:610,620`
   - **`multigroup=True`**: `MGGP_LCGP(kernel, M=M, n_groups=n_groups, jitter=self.jitter, K=self.K, rank=rank, diag_mode=self.diag_mode)`
     - Uses `batched_MGGP_Matern32` kernel with `group_diff_param` and `n_groups`
     - Reference: `GPzoo/gpzoo/models/nsf.py:738-742,754`

6. **Set inducing points**:
   - Both: `gp.Z = nn.Parameter(Z, requires_grad=False)`
   - **`multigroup=True` only**: `gp.groupsZ = nn.Parameter(groupsZ, requires_grad=False)`
   - Reference: `GPzoo/gpzoo/models/nsf.py:621, 755-756`

7. **Initialize LowRankPlusDiagonal** (batch for L factors):
   - Delete default Lu and replace with batched version
   - Initialize D (diagonal) to `scale_multiplier`
   - Initialize V (low-rank) to small random values `randn(L, M, R) * 0.01`
   - Initialize mu to `randn(L, M) * 0.1` (will be overwritten by data-aware init later)
   - Reference: `GPzoo/gpzoo/models/nsf.py:643-673` (`_init_lu_for_lcgp` method)

8. **Precompute KNN indices**:
   - `knn_idx = gp.calculate_knn(coordinates)[:, :-1]` (exclude self)
   - Store on model: `gp.knn_idx = knn_idx` and `gp.knn_idz = knn_idx` (since Z = X, knn_idx == knn_idz)
   - Reference: `GPzoo/gpzoo/models/nsf.py:638-639`

9. **Freeze kernel hyperparameters** (same logic as SVGP path)

### Phase 4: Modify fit() for LCGP Training Loop

**File:** `PNMF/models.py` — `PNMF.fit()` (line 739)

**Changes to the initialization section (~lines 828-856):**
- Add branch: `if self.spatial and self.prior_type == 'LCGP':`
  - Call `self._create_lcgp_prior()` instead of `self._create_spatial_prior()`
  - Store the full KNN indices: `self._knn_idx = self._prior.knn_idz.clone()`

**Changes to the training loop (lines 963-1108):**

The training loop needs a new branch for LCGP. Key differences from SVGP:

1. **KNN index management per batch**:
   - Before each forward pass, set `self._prior.knn_idx` and `self._prior.knn_idz` for the current batch
   - When using sample batching (idx is not None): `knn_idx = self._knn_idx[idx]`
   - When full-batch: `knn_idx = self._knn_idx`
   - Reference: `GPzoo/gpzoo/training_utilities.py:710-720` (train_lcgp_batched_with_tracking)

2. **Forward pass**:
   - LCGP uses `forward_train()` instead of the standard SVGP `forward()`:
     - `qF, _, _ = self._prior.forward_train(coordinates_batch, idx=idx)`
   - This returns the marginal distribution q(U_j) = N(mu_j, sqrt(s_jj))
   - Reference: `GPzoo/gpzoo/gp.py:791-836` (LCGP.forward_train)

3. **KL divergence**:
   - Uses `self._prior.kl_divergence_full(qZ=None, idx=idx)` instead of `self._prior.kl_divergence(qU, pU)`
   - The locally conditioned KL is NOT scaled by N/M (unlike SVGP) — it's already summed over all points in the batch
   - When using sample batching: scale KL by N/batch_size (since we only compute KL for batch points)
   - Reference: `GPzoo/gpzoo/training_utilities.py:740` and `GPzoo/gpzoo/gp.py:838-919`

4. **Expected log-likelihood**:
   - Same as SVGP path — `compute_log_likelihood_terms()` works with any Normal qF
   - The `elbo.py` module is unchanged — it just needs `qF.mean` and `qF.scale`

5. **Loss computation**:
   - `loss = kl - exp_ll` (same structure, but KL source differs)

**Concrete changes to PoissonFactorization.forward()** (`PNMF/models.py:91-151`):
- Add a new branch for `spatial='lcgp'` or check `prior_type`:
  - Set `self.prior.knn_idx = knn_idx` and `self.prior.knn_idz = knn_idx`
  - Call `qF, _, _ = self.prior.forward_train(X=coordinates, idx=idx)`
  - Compute log-likelihood terms (same code path)
  - Return `terms, qF, kl_value` (or package differently)

**Alternative approach (simpler):** Keep the LCGP-specific logic in the `PNMF.fit()` training loop directly, rather than threading it through `PoissonFactorization.forward()`. This avoids complicating the forward() signature further.

### Phase 5: Modify transform() for LCGP

**File:** `PNMF/models.py` — `PNMF.transform()` (line 1119)

**Changes:**
- For LCGP spatial models, use the GP's full `forward()` method (not `forward_train()`) for prediction at new coordinates
- The full forward pass uses all M inducing points to compute the predictive distribution
- `qF, _, _ = self._prior(X=coords_t)` or `self._prior(X=coords_t, groupsX=groups_t)` for multi-group
- Return `exp(qF.mean).T.cpu().numpy()`

This is the **same** as the current SVGP transform path. The key difference is that LCGP's `forward()` inherits from SVGP and works identically for prediction — local conditioning only affects training KL.

### Phase 6: Update transforms.py for LCGP

**File:** `PNMF/transforms.py`

**Changes to `_get_spatial_qF()` (line 24):**
- LCGP's `forward()` returns `(qF, qU, pU)` just like SVGP — no changes needed for standard forward
- However, for training coordinates, we might want to use `forward_train()` for efficiency (avoids full GP forward with kernel solves)
- Add optional `use_training_forward=False` parameter:
  - If True and model uses LCGP: call `self._prior.forward_train(coords, idx=None)` which returns marginal distribution directly
  - If False: call `self._prior(X=coords, ...)` which does full GP predictive (more expensive but works at new coordinates)

**No changes needed for:**
- `log_factors()`, `get_factors()`, `factor_uncertainty()`, `factor_samples()` — they all call `_get_spatial_qF()` which returns a Normal distribution regardless of prior type

### Phase 7: Optimizer Setup for LCGP

**File:** `PNMF/models.py` — `PNMF.fit()` optimizer section (~lines 861-933)

**LCGP needs parameter-group-aware optimization** (reference: `GPzoo/gpzoo/datasets/slideseq/lcgp_nsf.py:155-175`):

The GPzoo training scripts use separate learning rates for different parameter groups:
- **mu (mean) params**: `lr = LR_MEAN` (0.01)
- **W (loading) params**: `lr = LR_LOADING` (0.001)
- **Lu (scale) params**: `lr = 0.0` initially, unfrozen to `LR_SCALE` (0.01) after step 1
- **Other params**: `lr = LR` (0.01)

**Options:**
1. **Simple approach (recommended for v1):** Use a single optimizer with one learning rate for all params (same as current SVGP path). This is simpler and the PNMF API already supports `learning_rate` as a single value.

2. **Advanced approach (later):** Add `lr_mu`, `lr_loadings`, `lr_scale` parameters and build param groups. Could also add `scale_unfreeze_step` for staged unfreezing.

**Recommendation:** Start with Option 1 (single LR). Add staged unfreezing and per-group LRs as a follow-up if needed. The PNMF scheduler (OneCycleLR) should handle warmup adequately.

### Phase 8: Add Tests

**File:** `tests/test_lcgp.py` (new file)

**Test classes to create:**

1. **TestLCGPValidation** — Parameter validation
   - Test that `prior='LCGP'` with `spatial=True` is accepted
   - Test that `prior='LCGP'` with `spatial=False` raises error
   - Test that `K < 1` raises error
   - Test that invalid `diag_mode` raises error
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

### Phase 9: Update pyproject.toml

**File:** `pyproject.toml`

**Changes:**
- The `faiss-cpu` and `gpzoo` dependencies are already in the main dependencies (line 32-33), so no changes are needed for LCGP since it uses the same GPzoo package.

### Phase 10: Update __init__.py (if needed)

**File:** `PNMF/__init__.py`

**No changes expected** — the LCGP functionality is accessed through the existing `PNMF` class with `spatial=True, prior='LCGP'`. No new public classes or functions need exporting.

### Phase 11: Update CLAUDE.md

**File:** `CLAUDE.md`

**Changes:**
- Add LCGP section to "Spatial Mode" documentation
- Document new parameters: `K`, `rank`, `diag_mode`, `scale_multiplier`, `precompute_knn`
- Add LCGP vs SVGP comparison table
- Document the LCGP API usage example
- Add architecture differences table (LCGP vs SVGP vs GaussianPrior)

---

## Key Design Decisions

### 1. LCGP as a prior_type, not a separate class

LCGP is added as `prior='LCGP'` within the existing `PNMF` class, not as a new top-level class. This keeps the API simple:

```python
# SVGP (existing)
model = PNMF(spatial=True, prior='SVGP', num_inducing=3000)

# LCGP (new)
model = PNMF(spatial=True, prior='LCGP', K=50, rank=55)
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
2. **Phase 3**: `_create_lcgp_prior()` (core creation logic)
3. **Phase 4**: Training loop modifications (most complex phase)
4. **Phase 5-6**: transform() and transforms.py updates
5. **Phase 7**: Optimizer setup
6. **Phase 8**: Tests (write alongside implementation)
7. **Phase 9-11**: Cleanup (pyproject.toml, __init__.py, CLAUDE.md)

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
    prior='LCGP',
    multigroup=False,              # No groups (default)
    # LCGP-specific parameters
    K=50,                          # 50 nearest neighbors
    rank=55,                       # Low-rank component rank (default: K+5)
    diag_mode='softplus',          # Diagonal constraint mode
    scale_multiplier=1e-6,         # Initial diagonal scale
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
    prior='LCGP',
    multigroup=True,               # Enable multi-group
    # LCGP-specific parameters
    K=50,
    rank=55,
    diag_mode='softplus',
    scale_multiplier=1e-6,
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
model_svgp = PNMF(spatial=True, prior='SVGP', multigroup=False, num_inducing=3000)

# LCGP without groups: all points as inducing, low-rank+diagonal covariance
model_lcgp = PNMF(spatial=True, prior='LCGP', multigroup=False, K=50)

# SVGP with groups
model_svgp_mg = PNMF(spatial=True, prior='SVGP', multigroup=True, num_inducing=3000)

# LCGP with groups
model_lcgp_mg = PNMF(spatial=True, prior='LCGP', multigroup=True, K=50)
```

---

## Risk Assessment

### Low Risk
- Constructor params and validation (Phase 1-2): Straightforward additions
- transforms.py updates (Phase 6): Minimal changes, same qF interface
- Tests (Phase 8): Standard test patterns

### Medium Risk
- `_create_lcgp_prior()` (Phase 3): Needs careful initialization of LowRankPlusDiagonal
- Optimizer setup (Phase 7): Single LR may not be optimal; may need parameter groups later

### High Risk
- Training loop modifications (Phase 4): Most complex change
  - KNN index management per batch
  - Correct KL scaling with mini-batching
  - Correct forward_train() vs forward() usage
  - Testing convergence on real spatial data

### Mitigation
- Start with small synthetic data (N=50) for development/debugging
- Compare ELBO trajectories between PNMF LCGP and GPzoo LCGP_NSF on same data
- Use lower-bound mode first (fully analytic, no MC noise) for debugging
