# Spatial-Factorization Pipeline: Save/Load & Group Embedding Analysis

This document traces how group codes (C), group embeddings, and GP parameters flow
through the `spatial_factorization train → analyze → figures` pipeline, identifying
potential bugs that could cause incorrect qF scales.

---

## 1. Pipeline Overview

```
┌──────────────────────────────────────────────────────────────────┐
│  TRAIN (spatial_factorization/commands/train.py)                 │
│                                                                  │
│  1. load_preprocessed(output_dir)                                │
│     → X (N,2), Y (N,D), C (N,) from .npy files                  │
│                                                                  │
│  2. PNMF(**config.to_pnmf_kwargs()).fit(Y, coordinates=X,       │
│                                         groups=C)                │
│     → _create_spatial_prior():                                   │
│         n_groups = C.max() + 1                                   │
│         kernel = batched_MGGP_Matern32(n_groups=n_groups, ...)   │
│           → embedding = _embed_distance_matrix(uniform_D)        │
│         Z, groupsZ = mggp_kmeans_inducing_points(X, C, M)       │
│         gp = MGGP_SVGP(kernel, Z, groupsZ, mu, Lu)              │
│     → Train via ELBO optimization                                │
│                                                                  │
│  3. _save_model() → model.pth                                   │
│     prior_state_dict = gp.state_dict()                           │
│       keys: kernel.sigma, kernel.lengthscale,                    │
│             kernel.group_diff_param, kernel.embedding,           │
│             Z, groupsZ, mu, Lu._raw, Lu._mask                   │
│     model_state_dict = W.state_dict()                            │
│     hyperparameters = {spatial, prior, mode, loadings_mode, ...} │
│                                                                  │
│  ❌ NOT SAVED: spot groups C (N,), coordinates X (N,2),          │
│     n_groups, jitter, cholesky_mode, diagonal_only               │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  ANALYZE (spatial_factorization/commands/analyze.py)             │
│                                                                  │
│  1. load_preprocessed(output_dir)                                │
│     → X (N,2), Y (N,D), C (N,) from SAME .npy files             │
│                                                                  │
│  2. _load_model(model_dir):                                      │
│     a. Create kernel with DEFAULTS (sigma=1, ls=1, gdp=10)      │
│     b. Create MGGP_SVGP with DEFAULTS (jitter=1e-5, etc.)       │
│     c. gp.load_state_dict(prior_sd)  ← overwrites everything    │
│                                                                  │
│  3. _get_spatial_qF(model, coordinates=X, groups=C)              │
│     → gp.forward(X=coords, groupsX=groups)                      │
│       → forward_kernels uses groupsX AND self.groupsZ            │
│       → Computes W, transforms mu/Lu, builds qF                 │
│                                                                  │
│  4. factors = exp(qF.mean), scales = qF.scale                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 2. Group Embedding Mechanism

### 2.1 Creation (training time)

**File:** `gpzoo/kernels.py:116-121`

```python
class batched_MGGP_Matern32:
    def __init__(self, ..., n_groups=10):
        group_distances = torch.ones(n_groups) - torch.eye(n_groups)  # uniform
        self.embedding = nn.Parameter(
            _embed_distance_matrix(group_distances), requires_grad=False
        )
```

**File:** `gpzoo/utilities.py:700-710`

```python
def _embed_distance_matrix(distance_matrix):
    D2 = distance_matrix ** 2
    C = I - (1/N) * ones
    B = -0.5 * (C @ D2 @ C)         # double-centered
    L, Q = torch.linalg.eigh(B)     # spectral decomposition
    L[L < 0] = 0
    embedding = Q @ diag(sqrt(L))   # MDS embedding
    return embedding                 # shape: (n_groups, d_embed)
```

For the uniform distance matrix (all off-diagonal = 1):
- n_groups=2: embedding ≈ `[[0.5], [-0.5]]`, group_dist = 1.0
- n_groups=3: embedding is 2D, all pairwise distances = 1.0

### 2.2 Usage in kernel (every forward pass)

**File:** `gpzoo/kernels.py:152-170`

```python
def forward(self, X, Z, groupsX, groupsZ, diag=False):
    if diag:
        return sigma² * ones(1, N)    # ← GROUPS IGNORED for diagonal

    group_embeddingsX = self.embedding[groupsX]   # (N, d_embed) — direct indexing!
    group_embeddingsZ = self.embedding[groupsZ]   # (M, d_embed)
    # vmap over all (z,x) pairs with their group embeddings
```

**File:** `gpzoo/kernels.py:129-149` (covariance function)

```python
group_dist = sum((group_embedding1 - group_embedding2) ** 2)
val = 1 / (|group_diff_param| * group_dist + 1)      # v ∈ (0, 1]
val2 = sqrt(3) * dist_scaled * sqrt(val)               # scaled distance
cov = sigma² * (1 + val2) * exp(-val2) * val^((3+p)/2) # attenuated Matern
```

**Key properties:**
- Same group: `group_dist=0` → `val=1` → standard Matern 3/2
- Different group: `group_dist>0` → `val<1` → attenuated covariance
- `group_diff_param` controls attenuation strength (larger = more separation)

### 2.3 How groups affect qF scale

The predictive variance (from SVGP_ANALYSIS.md §6):

$$\sigma^2_{f,l}(x) = \underbrace{\sigma^2}_{\text{A: prior}} - \underbrace{\mathbf{w}(x)^\top \mathbf{w}(x)}_{\text{B: info gain}} + \underbrace{\|\mathbf{w}(x)^\top \mathbf{L}^{tr}_{u,l}\|^2}_{\text{C: variational}}$$

where $\mathbf{w}(x) = \mathbf{L}_{kzz}^{-\top} \mathbf{k}_{zx}(x)$.

**Groups affect Kzx and Kzz, which determine W(x).**

For a data point in group $g$:
- Kzx entries to inducing points in group $g$: **full** Matern covariance
- Kzx entries to inducing points in group $g' \neq g$: **attenuated** covariance

This means $\mathbf{w}(x)$ has **group-dependent structure**: stronger weights toward
same-group inducing points. Consequently, qF.scale inherits group structure.

With `group_diff_param=1.0` (your config): moderate attenuation.
With `group_diff_param=10.0` (loading default): strong attenuation → very different W(x).

---

## 3. Save/Load Analysis: What Could Go Wrong

### 3.1 State dict completeness — ✅ OK

The `prior_state_dict` includes ALL nn.Parameters and buffers:

| Key | Shape | Saved? | Loaded? |
|-----|-------|--------|---------|
| `kernel.sigma` | `(1,)` | ✅ | ✅ via `load_state_dict` |
| `kernel.lengthscale` | `(1,)` | ✅ | ✅ via `load_state_dict` |
| `kernel.group_diff_param` | `(1,)` | ✅ | ✅ via `load_state_dict` |
| `kernel.embedding` | `(n_groups, d)` | ✅ | ✅ via `load_state_dict` |
| `Z` | `(M, 2)` | ✅ | ✅ via `load_state_dict` |
| `groupsZ` | `(M,)` | ✅ | ✅ via `load_state_dict` |
| `mu` | `(L, M)` | ✅ | ✅ via `load_state_dict` |
| `Lu._raw` | `(L, M, M)` | ✅ | ✅ via `load_state_dict` |
| `Lu._mask` | `(M, M)` | ✅ | ✅ buffer via `load_state_dict` |

**Conclusion:** All kernel parameters, including the embedding and group_diff_param,
are correctly saved and restored by `load_state_dict`. The hardcoded defaults
(`sigma=1.0, lengthscale=1.0, group_diff_param=10.0`) at `analyze.py:91-93` are
overwritten before any forward pass.

### 3.2 Spot groups (C) — ⚠️ RISK: NOT SAVED WITH MODEL

**Problem:** Spot group codes `C (N,)` are NOT saved in `model.pth`.
They are reloaded from `preprocessed/C.npy` during analysis.

**File:** `analyze.py:495-496`
```python
coords = data.X.numpy()
groups = data.groups.numpy() if data.groups is not None else None
```

**Impact:** If preprocessing is re-run between training and analysis (different
filtering, different group assignments), the groups would change silently. Since
groups directly index into `kernel.embedding`, mismatched group codes would compute
wrong Kzx → wrong W(x) → wrong qF.scale.

**Risk level:** LOW if you never re-run preprocessing. HIGH if you do.

**Validation check:** Compare `C.max() + 1` against `groupsZ.max() + 1`:
```python
state = torch.load("model.pth", map_location="cpu", weights_only=False)
groupsZ = state["prior_state_dict"]["groupsZ"]
C = np.load("preprocessed/C.npy")
assert int(C.max()) + 1 == int(groupsZ.max().item()) + 1, "Group count mismatch!"
```

### 3.3 Config parameters not persisted in state — ⚠️ MINOR

These are hardcoded during load and NOT verified against training config:

| Parameter | Load default | Config value | Impact |
|-----------|-------------|-------------|--------|
| `jitter` | `1e-5` | likely `1e-5` | Low (only affects Kzz conditioning) |
| `cholesky_mode` | `"exp"` | `"exp"` | Low (loaded _raw is in correct space) |
| `diagonal_only` | `False` | `False` | Low (mask shape matches) |

These are OK for your current config, but would break silently if you changed them.

### 3.4 Coordinates (X) — ⚠️ RISK: NOT SAVED WITH MODEL

Same issue as groups: reloaded from `preprocessed/X.npy`. Must be identical
to training data.

---

## 4. Detailed Forward Pass Trace: Groups Through the Pipeline

### Step 1: `_get_spatial_qF(model, coordinates, groups)`

**File:** `PNMF/transforms.py:24-65`

```python
coords = torch.from_numpy(coordinates.astype(np.float32)).to(device)
grps = torch.from_numpy(groups.astype(np.int64)).to(device)
qF, _, _ = model._prior(X=coords, groupsX=grps)
```

Note: `grps` dtype is `int64` — used for integer indexing into `embedding`.

### Step 2: `MGGP.forward_kernels(X=coords, groupsX=grps)`

**File:** `gpzoo/gp.py:946-972`

```python
groupsZ = self.groupsZ                    # from loaded state dict
X, Z, groupsX, groupsZ = self.reshape_input_data(
    X=coords, Z=self.Z, groupsX=grps, groupsZ=groupsZ
)
# X: (N, 2), Z: (M, 2), groupsX: (N,), groupsZ: (M,)

# Compute three kernel matrices
Kxx = kernel(X, X, groupsX, groupsX, diag=True)     # (1, N) — groups IGNORED
Kzx = kernel(Z, X, groupsZ, groupsX)                 # (M, N) — groups USED
Kzz = kernel(Z, Z, groupsZ, groupsZ)                 # (M, M) — groups USED
```

**Critical:** Kxx diagonal does NOT use groups (just returns σ²). But Kzx and Kzz
DO use groups. This means:
- Term A (prior variance) = σ² for everyone, regardless of group
- Term B (info gain) and Term C (variational) depend on groups through W

### Step 3: `WSVGP.forward()` — Predictive distribution

**File:** `gpzoo/gp.py:325-395`

```python
L_kzz = chol(Kzz)                                    # (M, M) — depends on groups
Wt = solve_triangular(L_kzz, Kzx, upper=False)       # L^{-1} Kzx — (M, N)
W = Wt.T                                              # Kxz L^{-T} — (N, M)

mu_tr, Lu_tr = transform_variables(mu, Lu, L_kzz)    # whitened params

mean = W @ mu_tr                                       # (L, N)
cov_diag = σ² - sum(W², dim=-1) + sum((W @ Lu_tr)², dim=-1)  # (L, N)
qF = Normal(mean, sqrt(cov_diag))
```

### Step 4: Where groups matter most

The weight vector w(x) for data point x in group g:

$$w_m(x) = \sum_j [L_{kzz}^{-1}]_{mj} \cdot k(z_j, x, \text{group}(z_j), g)$$

- If `group(z_j) == g`: full Matern covariance → strong weight
- If `group(z_j) != g`: attenuated covariance → weak weight (how weak depends on `group_diff_param`)

**With `group_diff_param=1.0`** (training): moderate cross-group coupling
**With `group_diff_param=10.0`** (initial default at load): very weak cross-group coupling

If `load_state_dict` fails to overwrite `group_diff_param`, the loaded model
would have dramatically different W(x) and thus wrong qF.scale.

---

## 5. Diagnostic: Verify Save/Load Correctness

### Test 1: Verify kernel parameters after loading

```python
import torch
from pathlib import Path

model_dir = Path("outputs/slideseq/svgp")
state = torch.load(model_dir / "model.pth", map_location="cpu", weights_only=False)
prior_sd = state["prior_state_dict"]

# Check what was saved
print("=== Saved kernel parameters ===")
for k, v in prior_sd.items():
    if "kernel" in k:
        print(f"  {k}: shape={v.shape}, value={v.item() if v.numel()==1 else v[:3]}")

# Load model and check what was restored
from spatial_factorization.commands.analyze import _load_model
model = _load_model(model_dir)
kernel = model._prior.kernel

print("\n=== Loaded kernel parameters ===")
print(f"  sigma:          {kernel.sigma.item():.4f}")
print(f"  lengthscale:    {kernel.lengthscale.item():.4f}")
print(f"  group_diff_param: {kernel.group_diff_param.item():.4f}")
print(f"  embedding shape: {kernel.embedding.shape}")

# Verify they match
assert torch.allclose(kernel.sigma.data, prior_sd["kernel.sigma"])
assert torch.allclose(kernel.lengthscale.data, prior_sd["kernel.lengthscale"])
assert torch.allclose(kernel.group_diff_param.data, prior_sd["kernel.group_diff_param"])
assert torch.allclose(kernel.embedding.data, prior_sd["kernel.embedding"])
print("\n✓ All kernel parameters match saved values")
```

### Test 2: Compare qF from fresh model vs loaded model

```python
import torch
import numpy as np
from PNMF.transforms import _get_spatial_qF

# Load preprocessed data
from spatial_factorization.datasets.base import load_preprocessed
data = load_preprocessed("outputs/slideseq")

coords = data.X.numpy()
groups = data.groups.numpy()

# Method A: Load from saved state
from spatial_factorization.commands.analyze import _load_model
model_loaded = _load_model(Path("outputs/slideseq/svgp"))

with torch.no_grad():
    qF_loaded = _get_spatial_qF(model_loaded, coordinates=coords, groups=groups)
    mean_loaded = qF_loaded.mean.cpu()
    scale_loaded = qF_loaded.scale.cpu()

# Method B: If you saved the pickle (and it works)
import pickle
try:
    with open("outputs/slideseq/svgp/model.pkl", "rb") as f:
        model_pkl = pickle.load(f)
    with torch.no_grad():
        qF_pkl = _get_spatial_qF(model_pkl, coordinates=coords, groups=groups)
        mean_pkl = qF_pkl.mean.cpu()
        scale_pkl = qF_pkl.scale.cpu()
    print(f"Mean diff:  {(mean_loaded - mean_pkl).abs().max().item():.2e}")
    print(f"Scale diff: {(scale_loaded - scale_pkl).abs().max().item():.2e}")
except:
    print("Pickle not available, skipping comparison")

print(f"\nqF.mean  range: [{mean_loaded.min():.4f}, {mean_loaded.max():.4f}]")
print(f"qF.scale range: [{scale_loaded.min():.4f}, {scale_loaded.max():.4f}]")
```

### Test 3: Variance decomposition (same as SVGP_ANALYSIS.md §9)

```python
with torch.no_grad():
    gp = model_loaded._prior
    device = next(gp.parameters()).device

    coords_t = torch.from_numpy(coords.astype(np.float32)).to(device)
    groups_t = torch.from_numpy(groups.astype(np.int64)).to(device)

    mu, Lu = gp.apply_constraints()
    Kxx, Kzx, Kzz = gp.forward_kernels(X=coords_t, groupsX=groups_t, diag=True)

    from gpzoo.utilities import add_jitter
    Kzz = add_jitter(Kzz.clone(), gp.jitter)
    L_kzz = torch.linalg.cholesky(Kzz)

    Wt = torch.linalg.solve_triangular(L_kzz, Kzx, upper=False)
    W = Wt.transpose(-2, -1)

    mu_tr, Lu_tr = gp.transform_variables(mu, Lu, L_kzz)

    term_A = Kxx
    term_B = torch.sum(W**2, dim=-1)
    residual = torch.clamp(term_A - term_B, min=0.0)
    term_C = torch.sum((W @ Lu_tr)**2, dim=-1)
    total = residual + term_C

    print("=== Variance decomposition ===")
    print(f"Term A  (σ²)     : {term_A.min().item():.4f} to {term_A.max().item():.4f}")
    print(f"Term B  (info)   : {term_B.min().item():.4f} to {term_B.max().item():.4f}")
    print(f"Residual (A-B≥0) : {residual.min().item():.6f} to {residual.max().item():.6f}")
    print(f"Term C  (var)    : {term_C.min().item():.4f} to {term_C.max().item():.4f}")
    print(f"Total var        : {total.min().item():.4f} to {total.max().item():.4f}")
    print(f"qF.scale         : {total.sqrt().min().item():.4f} to {total.sqrt().max().item():.4f}")

    # Per-group analysis
    unique_groups = groups_t.unique()
    for g in unique_groups:
        mask = groups_t == g
        N_g = mask.sum().item()
        tC_g = term_C[:, mask]
        print(f"\nGroup {g.item()} (N={N_g}):")
        print(f"  Term C: [{tC_g.min().item():.4f}, {tC_g.max().item():.4f}], mean={tC_g.mean().item():.4f}")
        print(f"  Scale:  [{(residual[:, mask] + tC_g).sqrt().min().item():.4f}, "
              f"{(residual[:, mask] + tC_g).sqrt().max().item():.4f}]")

    # Whitened covariance diagnostic
    for l in range(min(3, Lu_tr.shape[0])):
        Sw = Lu_tr[l] @ Lu_tr[l].T
        d = Sw.diagonal()
        print(f"\nFactor {l}: whitened S diag [{d.min():.4f}, {d.max():.4f}]  (1.0 = prior)")
```

### Test 4: Group-specific W structure

```python
# Check that W has group-dependent structure
with torch.no_grad():
    unique_groups = groups_t.unique()
    groupsZ = gp.groupsZ

    for g in unique_groups:
        data_mask = groups_t == g
        inducing_mask = groupsZ == g

        # W for data points in group g
        W_g = W[data_mask]  # (N_g, M)

        # Average weight to same-group vs other-group inducing points
        w_same = W_g[:, inducing_mask].abs().mean().item()
        w_other = W_g[:, ~inducing_mask].abs().mean().item()

        print(f"Group {g.item()}: |w_same|={w_same:.4f}, |w_other|={w_other:.4f}, "
              f"ratio={w_same/w_other:.2f}x")
```

---

## 6. Potential Failure Modes

### 6.1 ❌ `load_state_dict` key mismatch

**Scenario:** The state dict keys don't match the reconstructed GP architecture.

**How to detect:**
```python
# This would raise RuntimeError if keys don't match
gp.load_state_dict(prior_sd, strict=True)  # default is strict=True
```

If loading succeeds without error, this is ruled out.

### 6.2 ❌ Group code permutation between train and analyze

**Scenario:** Preprocessing assigns different integer codes to groups.
E.g., training: hippocampus=0, cortex=1. Analysis: hippocampus=1, cortex=0.

**Impact:** `embedding[0]` and `embedding[1]` would be swapped, changing all
cross-covariances. qF.scale would be wrong.

**How to detect:**
```python
# At training time, save group info
import json
group_info = {
    "n_groups": int(C.max() + 1),
    "group_counts": {int(g): int((C==g).sum()) for g in np.unique(C)},
    "group_names": data.group_names,
}
# Compare with analysis-time groups
```

### 6.3 ⚠️ `group_diff_param` not loaded correctly

**Scenario:** `load_state_dict` loads group_diff_param as 1.0 (training) but
something goes wrong and it stays at 10.0 (initialization default).

**Impact:** With gdp=10.0, cross-group attenuation is ~10x stronger.
W(x) for cross-group inducing points would be near zero.
qF.scale would be dominated by same-group inducing points only.

**How to detect:**
```python
print(f"group_diff_param: {model._prior.kernel.group_diff_param.item()}")
# Should be 1.0 (from config), not 10.0 (from loading default)
```

### 6.4 ⚠️ Embedding eigenvector sign flip

**Scenario:** `_embed_distance_matrix` uses `torch.linalg.eigh` which can return
eigenvectors with flipped signs across different PyTorch versions or platforms.
The saved embedding is loaded via `load_state_dict`, so the LOADED embedding is
correct. But if you compare against a freshly-computed embedding, signs may differ.

**Impact:** None for normal use (loaded embedding is exact). Only matters if you
bypass `load_state_dict` and recompute the embedding.

### 6.5 ⚠️ CholeskyParameter raw ↔ constrained round-trip

**Scenario:** The `_to_unconstrained` / `_to_constrained` round-trip introduces
numerical error, especially for very small diagonal values.

For `mode='exp'`:
- Save: `_raw = log(Lu_diag)` — if `Lu_diag ≈ 0`, `_raw → -∞`
- Load: `Lu_diag = exp(_raw)` — recovers correctly unless `_raw` overflowed

**Impact:** Extremely unlikely for well-trained models. Could matter if Lu
diagonals collapsed during training.

---

## 7. Summary: Most Likely Explanations for Wrong qF.scale

qF.scale should be inverse proportional to qF.mean (confirmed in standalone
notebooks). The pipeline is introducing a bug somewhere. In order of likelihood:

### 7A. `group_diff_param` mismatch (CHECK THIS FIRST)

The loading code initializes with `group_diff_param=10.0` but training used `1.0`.
If `load_state_dict` fails to overwrite this, W(x) would be very different —
cross-group attenuation would be 10x stronger, completely changing the weight
structure and thus qF.scale.

**Quick check:**
```python
print(model._prior.kernel.group_diff_param.item())  # Should be 1.0
```

### 7B. Groups reloaded differently

If preprocessing was re-run, group codes could be permuted.

**Quick check:**
```python
C_loaded = np.load("preprocessed/C.npy")
groupsZ = state["prior_state_dict"]["groupsZ"].numpy()
print(f"Data groups: {np.unique(C_loaded)}, counts: {np.bincount(C_loaded)}")
print(f"Inducing groups: {np.unique(groupsZ)}, counts: {np.bincount(groupsZ)}")
```

---

## 8. Recommendations

1. **Save group info with model:**
   ```python
   state["group_info"] = {
       "n_groups": int(C.max() + 1),
       "group_counts": np.bincount(C).tolist(),
   }
   ```

2. **Save config params in state dict:**
   ```python
   state["hyperparameters"]["jitter"] = config.model.get("jitter", 1e-5)
   state["hyperparameters"]["cholesky_mode"] = config.model.get("cholesky_mode", "exp")
   state["hyperparameters"]["group_diff_param"] = config.model.get("group_diff_param", 1.0)
   ```

3. **Use saved config during load (already exists as `model_dir/config.yaml`):**
   ```python
   # In _load_model, load config.yaml and use its values instead of hardcoded defaults
   saved_config = Config.from_yaml(model_dir / "config.yaml")
   ```

4. **Validate groups at analysis time:**
   ```python
   loaded_n_groups = int(groups.max()) + 1
   saved_n_groups = int(groupsZ.max().item()) + 1
   if loaded_n_groups != saved_n_groups:
       raise ValueError(f"Group count mismatch: data has {loaded_n_groups}, "
                        f"model has {saved_n_groups}")
   ```
