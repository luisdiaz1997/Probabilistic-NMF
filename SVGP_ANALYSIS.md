# SVGP Mathematical Analysis & Code Audit

This document traces the full forward pass, predictive distribution, and KL divergence
for WSVGP, SVGP, and MGGP_SVGP in GPzoo, identifying the exact lines to inspect
when debugging the qF scale issue.

---

## 1. Class Hierarchy

```
BaseVGP          ← reshape_parameters, apply_constraints, forward_kernels
  └─ WSVGP       ← forward(), transform_variables = identity, CholeskyParameter Lu
       └─ SVGP    ← transform_variables = L^{-1} @ [mu, Lu]

MGGP             ← forward_kernels (group-aware), reshape_input_data (adds groups)

MGGP_SVGP = MGGPWrapper(SVGP)   ← MRO: MGGP, SVGP, WSVGP, BaseVGP
```

**Key file paths:**
- `GPzoo/gpzoo/gp.py` — all GP classes
- `GPzoo/gpzoo/kernels.py` — kernel functions
- `GPzoo/gpzoo/utilities.py` — `whitened_KL`, `add_jitter`, `svgp_forward`
- `GPzoo/gpzoo/modules.py` — `CholeskyParameter`
- `Probabilistic-NMF/PNMF/models.py` — `_create_spatial_prior`, training loop

---

## 2. Stored Parameters (Non-Whitened for SVGP)

For SVGP/MGGP_SVGP with L latent factors and M inducing points:

| Parameter | Shape | Meaning |
|-----------|-------|---------|
| `gp.mu` | `(L, M)` | Non-whitened inducing means |
| `gp.Lu.data` | `(L, M, M)` | Non-whitened Cholesky factor: `S = Lu @ Lu^T` |
| `gp.Z` | `(M, 2)` | Inducing point locations (frozen) |
| `gp.groupsZ` | `(M,)` | Inducing point group labels (frozen, MGGP only) |

The variational distribution at inducing points is:

$$q(\mathbf{u}_l) = \mathcal{N}(\boldsymbol{\mu}_l,\; \mathbf{L}_{u,l}\,\mathbf{L}_{u,l}^\top)$$

where $\boldsymbol{\mu}_l$ = `gp.mu[l]` and $\mathbf{L}_{u,l}$ = `gp.Lu.data[l]`.

The marginal variance at inducing point $m$ for factor $l$ is:

$$\text{Var}(u_{l,m}) = \sum_{k=0}^{m} L_{u,l}[m,k]^2$$

> **Note:** `Lu.sum(axis=-1).sqrt()` computes $\sqrt{\sum_k L_{u}[m,k]}$ (sum of elements),
> **NOT** $\sqrt{\sum_k L_{u}[m,k]^2}$ (standard deviation). The correct inducing std is
> `(Lu**2).sum(axis=-1).sqrt()` or equivalently `(Lu @ Lu.T).diagonal(dim1=-2, dim2=-1).sqrt()`.

---

## 3. Forward Pass: WSVGP.forward()

**File:** `gpzoo/gp.py:325–395`

### Step 1: Compute kernels

```python
# gp.py:329
covariances = self.forward_kernels(X, diag=diag, **kwargs)
```

For MGGP_SVGP, this dispatches to `MGGP.forward_kernels` (`gp.py:946–972`),
which calls the MGGP kernel with group labels via `vmap`.

Output shapes (no batch dims):
- `Kxx`: `(1, N)` — see note below
- `Kzx`: `(M, N)`
- `Kzz`: `(M, M)`

> **Shape note:** The kernel `diag=True` branch (`kernels.py:154–156`) returns
> `(self.sigma**2).reshape(-1, 1).expand(1, N)` which is `(1, N)`, not `(N,)`.
> This extra leading dimension broadcasts correctly in all subsequent operations.

### Step 2: Apply constraints

```python
# gp.py:335
mu, Lu = self.apply_constraints()
```

For WSVGP (`gp.py:315–322`):
```python
Lu = self.Lu.data   # CholeskyParameter → constrained lower triangular
mu = self.mu        # raw nn.Parameter
```

`CholeskyParameter._to_constrained` (`modules.py:242–255`):
```python
diag = torch.diagonal(raw, dim1=-2, dim2=-1)
lower = torch.tril(raw, diagonal=-1)
if self.mode == 'exp':
    constrained_diag = torch.exp(diag)      # ← diagonal constraint
else:
    constrained_diag = F.softplus(diag)
return lower + torch.diag_embed(constrained_diag)
```

### Step 3: Cholesky of Kzz and compute W

```python
# gp.py:342–347
L = torch.linalg.cholesky(Kzz)                          # (M, M)
Wt = torch.linalg.solve_triangular(L, Kzx, upper=False) # L^{-1} Kzx,  shape (M, N)
W = Wt.transpose(-2, -1)                                 # Kxz L^{-T},  shape (N, M)
```

So:

$$\mathbf{W} = \mathbf{K}_{xz}\,\mathbf{L}^{-\top}$$

where $\mathbf{L} = \text{chol}(\mathbf{K}_{zz})$.

### Step 4: Transform variables (SVGP only)

```python
# gp.py:354
mu, Lu = self.transform_variables(mu, Lu, L, verbose=verbose)
```

**WSVGP** (`gp.py:49–52`): identity — parameters are already whitened.

**SVGP** (`gp.py:462–513`): un-whitens via $\mathbf{L}^{-1}$:

```python
# gp.py:507–511  (batched case, L.dim() == 3... but for shared kernel L.dim() == 2)
stacked = torch.cat([mu.unsqueeze(-1), Lu], dim=-1)   # (M, L_factors + L_factors*M)
X = torch.linalg.solve_triangular(L, stacked, upper=False)
mu = X[..., 0]     # L^{-1} @ mu
Lu = X[..., 1:]    # L^{-1} @ Lu
```

When `L.dim() == 2` (shared kernel across L factors), the code uses
`flatten_to_solve` / `unflatten_from_solve` (`gp.py:54–80`) to batch-solve:

```python
# gp.py:466–494
mu_stacked, mu_shape = self.flatten_to_solve(mu, keepdim=-1)   # (L,M) → (M, L)
Lu_stacked, Lu_shape = self.flatten_to_solve(Lu, keepdim=-2)   # (L,M,M) → (M, L*M)
stacked = torch.cat([mu_stacked, Lu_stacked], dim=-1)          # (M, L + L*M)
X = torch.linalg.solve_triangular(L, stacked, upper=False)     # (M, L + L*M)
mu = unflatten(X[:, :L])       # → (L, M)
Lu = unflatten(X[:, L:])       # → (L, M, M)
```

After transform, for each factor $l$:

$$\boldsymbol{\mu}^{\text{tr}}_l = \mathbf{L}^{-1}\,\boldsymbol{\mu}_l, \qquad
  \mathbf{L}^{\text{tr}}_{u,l} = \mathbf{L}^{-1}\,\mathbf{L}_{u,l}$$

These are the **whitened** mean and Cholesky.

### Step 5: Predictive mean

```python
# gp.py:360–361
mean = W @ mu.unsqueeze(-1)   # (N,M) @ (L,M,1) → broadcasts to (L,N,1)
mean = mean.squeeze(-1)       # (L, N)
```

$$\mu_{f,l}(x) = \mathbf{w}(x)^\top\,\boldsymbol{\mu}^{\text{tr}}_l
              = \mathbf{k}_{xz}\,\mathbf{K}_{zz}^{-1}\,\boldsymbol{\mu}_l$$

### Step 6: Predictive variance (diagonal)

```python
# gp.py:368–382
cov_diag = Kxx                                           # (1, N)
cov_diag = cov_diag - torch.sum(W**2, dim=-1)           # (1, N) − (N,) → (1, N)
cov_diag = torch.clamp(cov_diag, min=0.0)               # CLAMP ≥ 0
cov_diag = cov_diag + torch.sum((W @ Lu) ** 2, dim=-1)  # (1,N) + (L,N) → (L, N)
cov_diag = torch.clamp(cov_diag, min=0.0, max=100.0)    # CLAMP [0, 100]
qF = distributions.Normal(mean, cov_diag.sqrt())         # scale = sqrt(var)
```

This implements three terms:

$$\sigma^2_{f,l}(x) = \underbrace{k(x,x)}_{\text{Term A}}
  - \underbrace{\mathbf{w}^\top \mathbf{w}}_{\text{Term B}}
  + \underbrace{\|\mathbf{w}^\top \mathbf{L}^{\text{tr}}_{u,l}\|^2}_{\text{Term C}}$$

Expanding with $\mathbf{w} = \mathbf{L}^{-1}\mathbf{k}_{zx}$:

$$\sigma^2_{f,l}(x) = k(x,x)
  - \mathbf{k}_{xz}\,\mathbf{K}_{zz}^{-1}\,\mathbf{k}_{zx}
  + \mathbf{k}_{xz}\,\mathbf{K}_{zz}^{-1}\,\mathbf{S}_l\,\mathbf{K}_{zz}^{-1}\,\mathbf{k}_{zx}$$

where $\mathbf{S}_l = \mathbf{L}_{u,l}\,\mathbf{L}_{u,l}^\top$ is the **non-whitened** inducing covariance.

> **This is mathematically correct** for the standard SVGP predictive equations.

### Step 7: Return

```python
# gp.py:393–395
qZ = distributions.MultivariateNormal(mu, scale_tril=Lu)   # whitened qU
pZ = None
return qF, qZ, pZ
```

Note: `qZ` uses the **transformed** (whitened) mu and Lu, not the stored parameters.

---

## 4. KL Divergence

**File:** `gpzoo/gp.py:123–136` and `gpzoo/utilities.py:254–263`

```python
# gp.py:123–136
def kl_divergence(self, qZ, pZ=None):
    if pZ is None:
        mean, scale_tril = qZ.mean, qZ.scale_tril       # whitened params
        ...
        kl_flat = torch.vmap(whitened_KL)(mean_flat, scale_tril_flat)
```

```python
# utilities.py:254–263
def whitened_KL(mz, Lz):
    Lz_diag = torch.diagonal(Lz)
    log_Lz_diag = torch.log(Lz_diag)
    M = len(mz)
    kl_term = -2*torch.sum(log_Lz_diag) + torch.sum(Lz**2) + torch.sum(mz**2) - M
    return 0.5 * kl_term
```

This computes:

$$\text{KL}\bigl(q_w \,\|\, \mathcal{N}(\mathbf{0}, \mathbf{I})\bigr)
  = \frac{1}{2}\Bigl(
    -2\sum_j \log (L^{\text{tr}}_u)_{jj}
    + \|\mathbf{L}^{\text{tr}}_u\|_F^2
    + \|\boldsymbol{\mu}^{\text{tr}}\|^2
    - M
  \Bigr)$$

Since $\boldsymbol{\mu}^{\text{tr}} = \mathbf{L}^{-1}\boldsymbol{\mu}$ and
$\mathbf{L}^{\text{tr}}_u = \mathbf{L}^{-1}\mathbf{L}_u$, this equals:

$$\text{KL}\bigl(\mathcal{N}(\boldsymbol{\mu}, \mathbf{S}) \,\|\, \mathcal{N}(\mathbf{0}, \mathbf{K}_{zz})\bigr)$$

> **This is correct.**

In PNMF (`models.py:853`):
```python
kl = self._prior.kl_divergence(qU, pU).sum()   # sum over L factors
```

---

## 5. MGGP Kernel

**File:** `gpzoo/kernels.py:115–170` (`batched_MGGP_Matern32`)

### Covariance function (`kernels.py:129–149`)

```python
def covariance(self, x1, x2, group_embedding1, group_embedding2):
    diff = x1 - x2
    dist = torch.sqrt((diff**2).sum())
    dist_scaled = dist / torch.abs(self.lengthscale)
    p = x1.unsqueeze(0).shape[-1]                                # spatial dim
    group_dist = torch.sum((group_embedding1 - group_embedding2) ** 2)
    val = 1 / (torch.abs(self.group_diff_param) * group_dist + 1)
    val2 = (3**0.5) * dist_scaled * (val**0.5)                   # ← multiplicative
    cov = (self.sigma**2) * (1 + val2) * torch.exp(-val2) * (val**((3+p)/2))
    return cov
```

$$k(x_1, x_2, g_1, g_2) = \sigma^2\,(1 + \tilde{d})\,e^{-\tilde{d}}\,v^{(3+p)/2}$$

where:
- $v = \frac{1}{\alpha\,\|e_{g_1} - e_{g_2}\|^2 + 1}$ (group modulation, $v=1$ for same group)
- $\tilde{d} = \frac{\sqrt{3}\,\|x_1 - x_2\|}{\ell}\,\sqrt{v}$ (scaled distance)

### Diagonal shortcut (`kernels.py:154–156`)

```python
if diag:
    return (self.sigma**2).reshape(-1, 1).expand(
        len(self.sigma) if ... else 1, len(X))    # → (1, N)
```

For diagonal: $x_1 = x_2$, same group → $v = 1$, $\tilde{d} = 0$:

$$k(x, x, g, g) = \sigma^2 \cdot 1 \cdot 1 \cdot 1 = \sigma^2$$

The shortcut is correct. Output shape is `(1, N)`, not `(N,)`.

---

## 6. PNMF Initialization of Spatial Prior

**File:** `PNMF/models.py:477–572`

### Inducing points (`models.py:519–531`)
```python
M = min(self.num_inducing, N)
if self.multigroup and groups is not None:
    Z, groupsZ = mggp_kmeans_inducing_points(
        coordinates, groups, M, seed=self.random_state or 123,
        allocation=self.inducing_allocation)
```

### Batched Lu initialization (`models.py:552–560`)
```python
del gp.Lu
gp.Lu = CholeskyParameter(
    (L, M), mode=self.cholesky_mode, diagonal_only=self.diagonal_only)  # creates default raw

# Then overwrite:
Lu_init = torch.randn(L, M, M) * 1e-2        # off-diagonal: ~N(0, 0.01)
Lu_init = torch.tril(Lu_init)                  # enforce lower triangular
Lu_init[:, range(M), range(M)] = torch.rand(L, M)   # diagonal: Uniform[0, 1)
gp.Lu.data = Lu_init                           # calls _to_unconstrained(Lu_init)
```

For `cholesky_mode='exp'`, `_to_unconstrained` (`modules.py:257–273`):
```python
unconstrained_diag = torch.log(diag)    # log(Uniform[0,1)) → (-∞, 0)
```

So the stored raw diagonal is `log(d)` where `d ~ Uniform[0, 1)`.
When constrained back: `exp(log(d)) = d`.

**Consequence:** The non-whitened inducing covariance diagonal is:

$$S_{l}[m,m] \approx d_{l,m}^2 \in [0, 1)$$

while the prior covariance diagonal is $K_{zz}[m,m] = \sigma^2 = 1.0$.

### Mu initialization (`models.py:561`)
```python
gp.mu = nn.Parameter(torch.randn(L, M) * 1.0)
```

---

## 7. Variance Decomposition: Why qF.scale May Look Wrong

### Term-by-term analysis

For a data point $x$ well-covered by M inducing points:

| Term | Expression | Typical value | Code line |
|------|-----------|---------------|-----------|
| A | $k(x,x) = \sigma^2$ | 1.0 | `gp.py:368` |
| B | $\mathbf{k}_{xz}\mathbf{K}_{zz}^{-1}\mathbf{k}_{zx}$ | ≈ 1.0 (good coverage) | `gp.py:371` |
| A − B | (clamped to ≥ 0) | ≈ 0 | `gp.py:374` |
| C | $\mathbf{k}_{xz}\mathbf{K}_{zz}^{-1}\mathbf{S}_l\mathbf{K}_{zz}^{-1}\mathbf{k}_{zx}$ | depends on $\mathbf{S}_l$ | `gp.py:376` |

So:

$$\sigma^2_{f,l}(x) \approx \text{Term C} = \mathbf{w}^\top\,\mathbf{L}^{-1}\mathbf{S}_l\mathbf{L}^{-\top}\,\mathbf{w}$$

The predictive scale is **entirely determined by the whitened inducing covariance**
$\mathbf{S}^w_l = \mathbf{L}^{-1}\mathbf{S}_l\mathbf{L}^{-\top}$ projected through
the weights $\mathbf{w}(x)$.

### Key difference from non-spatial

| Aspect | Non-spatial (GaussianPrior) | Spatial (SVGP) |
|--------|---------------------------|----------------|
| Scale parameters | Free per sample: $\sigma_{l,n}$ (`priors.py:263`) | Derived from $\mathbf{S}_l$ at inducing points |
| Depends on data $Y(x)$? | Yes (gradients flow to $\sigma_{l,n}$) | No (only indirectly through shared $\mathbf{S}_l$) |
| Spatial structure | None (independent per sample) | Smooth function of position |
| Scale pattern after training | "Opposite of exp(mean)" | Smooth spatial field (same for nearby points) |
| Degrees of freedom | $L \times N$ | $L \times M \times (M+1)/2$ (but projected through kernel) |

### Possible collapse scenarios

**Scenario 1: Good inducing coverage + small S**

If $M$ inducing points cover the data well:
- Term A − B ≈ 0
- Term C ∝ eigenvalues of $\mathbf{S}^w_l$
- If initialization has $S_l \ll K_{zz}$, then $S^w_l \ll I$ and Term C is small
- **Result:** qF.scale ≈ 0 everywhere

**Scenario 2: S learned to match Kzz**

If training drives $S_l → K_{zz}$ (the prior):
- $S^w_l → I$
- Term C ≈ Term B ≈ 1.0
- **Result:** $\sigma^2_f \approx 0 + 1.0 = 1.0 = \sigma^2$ (prior uncertainty recovered)

**Scenario 3: S adapts but stays smaller than Kzz**

After training, typically $S_l < K_{zz}$ (data constrains the posterior):
- $S^w_l$ has eigenvalues in $(0, 1)$
- Term C < Term B
- **Result:** $\sigma^2_f \in (0, \sigma^2)$

The scale will be a **smooth spatial function** determined by the inducing point layout
and the learned $S_l$, NOT by per-sample counts.

---

## 8. Lines to Inspect

### Predictive variance computation
- **`gp.py:368–382`** — The three-term variance formula (Term A, B, C)
- **`gp.py:374`** — `torch.clamp(cov_diag, min=0.0)` — floors Term A−B at 0
- **`gp.py:381`** — `torch.clamp(cov_diag, min=0.0, max=100.0)` — caps total variance at 100

### Transform variables (SVGP whitening)
- **`gp.py:462–513`** — `SVGP.transform_variables()` — applies $\mathbf{L}^{-1}$
- **`gp.py:466–494`** — Batched case when `L.dim() == 2` (shared kernel)
- **`gp.py:479`** — `solve_triangular(L, stacked, upper=False)` — the actual $\mathbf{L}^{-1}$ solve

### KL divergence
- **`gp.py:123–136`** — `kl_divergence()` dispatches to `whitened_KL` when `pZ is None`
- **`utilities.py:254–263`** — `whitened_KL` formula on whitened params

### Kernel diagonal
- **`kernels.py:154–156`** — `batched_MGGP_Matern32.forward(diag=True)` returns `(1, N)` not `(N,)`

### PNMF initialization
- **`models.py:552–560`** — Lu initialization: diagonal from `torch.rand` in [0, 1)
- **`models.py:561`** — mu initialization: `randn * 1.0`
- **`models.py:554–556`** — CholeskyParameter created with `mode=self.cholesky_mode`

### PNMF training loop (spatial branch)
- **`models.py:842–845`** — spatial forward pass
- **`models.py:848–853`** — expected log-likelihood + KL computation
- **`models.py:856–861`** — mini-batch scaling (exp_ll only, NOT kl)

---

## 9. Diagnostic: Variance Decomposition

To determine whether the issue is mathematical or an expectation mismatch, decompose
the predictive variance after training:

```python
with torch.no_grad():
    gp = model._prior
    coords = model._coordinates
    groups = model._groups
    mu, Lu = gp.apply_constraints()  # non-whitened: (L,M) and (L,M,M)

    # Kernels
    if groups is not None:
        Kxx, Kzx, Kzz = gp.forward_kernels(X=coords, groupsX=groups, diag=True)
    else:
        Kxx, Kzx, Kzz = gp.forward_kernels(X=coords, diag=True)

    from gpzoo.utilities import add_jitter
    Kzz = add_jitter(Kzz.clone(), gp.jitter)
    L_kzz = torch.linalg.cholesky(Kzz)

    Wt = torch.linalg.solve_triangular(L_kzz, Kzx, upper=False)   # (M, N)
    W = Wt.transpose(-2, -1)                                       # (N, M)

    mu_tr, Lu_tr = gp.transform_variables(mu, Lu, L_kzz)

    # --- Decompose ---
    term_A  = Kxx                                              # (1, N)
    term_B  = torch.sum(W**2, dim=-1)                          # (N,)
    residual = torch.clamp(term_A - term_B, min=0.0)           # (1, N)
    term_C  = torch.sum((W @ Lu_tr)**2, dim=-1)                # (L, N)
    total   = residual + term_C                                 # (L, N)

    print("Term A  (prior var)  :", term_A.min().item(), "to", term_A.max().item())
    print("Term B  (info gain)  :", term_B.min().item(), "to", term_B.max().item())
    print("Residual (A-B, ≥ 0) :", residual.min().item(), "to", residual.max().item())
    print("Term C  (variational):", term_C.min().item(), "to", term_C.max().item())
    print("Total var            :", total.min().item(), "to", total.max().item())
    print("qF.scale             :", total.sqrt().min().item(), "to", total.sqrt().max().item())

    # Whitened covariance diagnostic
    for l in range(min(3, Lu_tr.shape[0])):
        Sw = Lu_tr[l] @ Lu_tr[l].T
        d = Sw.diagonal()
        print(f"Factor {l}: whitened S diag [{d.min():.4f}, {d.max():.4f}] (1.0 = prior)")

    # Inducing std (non-whitened)
    S_diag = (Lu**2).sum(dim=-1)                               # (L, M)
    print("Inducing std range:", S_diag.sqrt().min().item(), "to", S_diag.sqrt().max().item())
```

**What to look for:**

1. If `Residual ≈ 0` and `Term C` is small → scales near zero (initialization or collapse)
2. If whitened S diagonal ≪ 1 → posterior much tighter than prior (may or may not be correct)
3. If whitened S diagonal ≈ 1 → posterior ≈ prior (not learning)
4. Compare `Term C` range to `Inducing std` — the projection through $\mathbf{w}(x)$ can
   dramatically change the effective scale at data points vs at inducing points

---

## 10. Summary of Findings

**The SVGP/MGGP_SVGP math is correct.** The predictive equations, whitening transform,
batched parameter handling, and KL divergence all check out:

- Predictive mean: $\mathbf{k}_{xz}\mathbf{K}_{zz}^{-1}\boldsymbol{\mu}$ ✓
- Predictive var: $k_{xx} - \mathbf{k}_{xz}\mathbf{K}_{zz}^{-1}\mathbf{k}_{zx} + \mathbf{k}_{xz}\mathbf{K}_{zz}^{-1}\mathbf{S}\mathbf{K}_{zz}^{-1}\mathbf{k}_{zx}$ ✓
- KL: $\text{KL}(\mathcal{N}(\boldsymbol{\mu}, \mathbf{S}) \| \mathcal{N}(\mathbf{0}, \mathbf{K}_{zz}))$ via whitened computation ✓
- Batched L-factor: `flatten_to_solve` correctly yields $\mathbf{L}^{-1}\mathbf{L}_{u,l}$ per factor ✓

**The qF.scale "not looking right" is likely because:**

The GP predictive variance is a smooth spatial function that does NOT capture per-sample
count-dependent uncertainty. This is a fundamental modeling difference from the non-spatial
GaussianPrior, not a bug. Run the diagnostic above to confirm the magnitudes are reasonable.
