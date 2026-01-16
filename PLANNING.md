# PNMF Refactoring Plan

## Vision

**GPzoo** = Backend library providing:
- GPs (WSVGP, SVGP, VNNGP, LCGP)
- Priors (GaussianPrior)
- Likelihoods (PoissonFactorization base)
- Constrained modules (PositiveParameter, CholeskyParameter, etc.)
- Custom KL divergences (whitened_KL, kl_divergence_full for VNNGP/LCGP)

**PNMF** = sklearn-compatible interface that:
- Imports building blocks from GPzoo
- Provides PNMF-specific ELBO computation (simple, expanded, lower-bound modes)
- Supports both standard (`spatial=False`) and spatial (`spatial=True`) priors

## Target Architecture

```
GPzoo (pip install gpzoo)
├── modules.py          # ConstrainedParameter, PositiveParameter, CholeskyParameter
│                       # LowRankFactor, LowRankPlusDiagonal
├── gp.py               # GaussianPrior, WSVGP, SVGP, VNNGP, LCGP
│                       # kl_divergence(), kl_divergence_full(), whitened_KL
├── likelihoods.py      # PoissonFactorization (base class only)
├── kernels.py          # RBF, Matern, etc.
└── utilities.py        # whitened_KL, add_jitter, svgp_forward

PNMF (pip install pnmf)
├── models.py           # PNMF sklearn class
│                       # - spatial=False: uses gpzoo.gp.GaussianPrior
│                       # - spatial=True: uses gpzoo.gp.WSVGP/SVGP/VNNGP/LCGP
└── elbo.py             # PNMF-specific ELBO computation
                        # - expected_log_likelihood_{simple,expanded,lower_bound}
                        # - compute_elbo(mode, ..., kl_fn=None)
                        # - kl_fn can be custom KL from GPzoo GP classes
```

## How It Fits Together

```python
# PNMF with standard prior (spatial=False)
from gpzoo.gp import GaussianPrior
from gpzoo.likelihoods import PoissonFactorization
from PNMF.elbo import compute_elbo, kl_divergence

prior = GaussianPrior(y, L=10)
model = PoissonFactorization(prior, y, L=10)
qF, pF = prior()
# ... forward pass ...
loss = compute_elbo(mode, rate, qF, pF, X, W)  # uses standard KL

# PNMF with spatial prior (spatial=True)
from gpzoo.gp import LCGP  # or VNNGP, WSVGP, etc.
from gpzoo.likelihoods import PoissonFactorization
from PNMF.elbo import compute_elbo

gp = LCGP(kernel, dim=2, M=100, K=15)
model = PoissonFactorization(gp, y, L=10)
qF, qU, pU = gp(X)
# ... forward pass ...
loss = compute_elbo(mode, rate, qF, None, X, W,
                    kl_fn=lambda q, p: gp.kl_divergence_full(qU, pU))
```

## Key Changes

### Phase 1: Clean up GPzoo

- [ ] Remove `V` parameter from `PoissonFactorization`
- [ ] Remove `V` parameter from `PNMF` class in GPzoo
- [ ] Remove `V` parameter from `NSF2` class
- [ ] Remove `PNMF` class from GPzoo (sklearn wrapper lives here)
- [ ] Remove `NSF2` class from GPzoo (handled by PNMF with spatial=True)
- [ ] Remove `Hybrid_NSF2`, `Hybrid_NSF_Exact` (or move to examples)
- [ ] Keep `PoissonFactorization` as base class with:
  - `W` (PositiveParameter)
  - `get_rate(prior_samples)`
  - `project_parameters()`

### Phase 2: Refactor PNMF repo

- [ ] Add `gpzoo` as dependency in `pyproject.toml`
- [ ] Replace local imports with GPzoo:
  ```python
  from gpzoo.modules import PositiveParameter
  from gpzoo.gp import GaussianPrior
  from gpzoo.likelihoods import PoissonFactorization
  ```
- [ ] Remove duplicated files:
  - `PNMF/priors.py` → use `gpzoo.gp.GaussianPrior`
  - `PNMF/custom_modules.py` → use `gpzoo.modules`
- [ ] Keep `PNMF/elbo.py` (PNMF-specific)
- [ ] Simplify `PNMF/models.py` to just the sklearn wrapper

### Phase 3: Add spatial support

- [ ] Add `spatial` parameter to `PNMF` class
- [ ] Add `gp_class` parameter for GP type selection (WSVGP, SVGP, VNNGP, LCGP)
- [ ] Add `kernel` parameter for spatial mode
- [ ] Handle coordinate input `X` in fit() when spatial=True

```python
# Future sklearn API
model = PNMF(n_components=10, spatial=False)
model.fit(Y)

model = PNMF(
    n_components=10,
    spatial=True,
    gp_class='LCGP',  # or 'WSVGP', 'VNNGP'
    kernel='rbf',
    n_inducing=100
)
model.fit(Y, X=coordinates)
```

## KL Divergence Integration

GPzoo provides different KL implementations:

| GP Class | KL Method | Notes |
|----------|-----------|-------|
| GaussianPrior | `kl_divergence(qF, pF)` | Standard torch KL |
| WSVGP | `kl_divergence(qZ, pZ=None)` | Uses `whitened_KL` when pZ=None |
| SVGP | `kl_divergence(qZ, pZ)` | Standard |
| VNNGP | `kl_divergence_full(qZ, pZ, idx)` | Local KL approximation |
| LCGP | `kl_divergence_full(qZ, pZ, idx)` | Local KL with low-rank |

PNMF's `compute_elbo()` accepts `kl_fn` parameter:
```python
# From PNMF/elbo.py
def compute_elbo(mode, rate, qF, pF, X, W, kl_fn=None):
    ell = expected_log_likelihood(mode, rate, qF, X, W)
    if kl_fn is not None:
        kl = kl_fn(qF, pF)
    else:
        kl = kl_divergence(qF, pF)
    return -(ell - kl)
```

## Summary

| Component | Lives In | Purpose |
|-----------|----------|---------|
| `PositiveParameter` | GPzoo | Constrained positive params |
| `CholeskyParameter` | GPzoo | Cholesky factor params |
| `GaussianPrior` | GPzoo | Non-spatial prior |
| `WSVGP/SVGP/VNNGP/LCGP` | GPzoo | Spatial priors (GPs) |
| `PoissonFactorization` | GPzoo | Base class (W, get_rate) |
| `expected_log_likelihood_*` | PNMF | Poisson ELBO modes |
| `compute_elbo` | PNMF | ELBO with custom KL support |
| `PNMF` (sklearn class) | PNMF | User-facing API |
