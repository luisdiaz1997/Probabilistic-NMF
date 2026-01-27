"""
Probabilistic Non-negative Matrix Factorization (PNMF)

A PyTorch-based implementation of Probabilistic NMF with variational inference,
compatible with scikit-learn's API.

Example:
    >>> from PNMF import PNMF
    >>> model = PNMF(n_components=5, random_state=0)
    >>> transformed = model.fit_transform(X)
    >>> components = model.components_

For PyTorch-native usage:
    >>> from PNMF import PoissonFactorization, GaussianPrior
    >>> prior = GaussianPrior(y, L=10)
    >>> model = PoissonFactorization(prior, y, L=10)
    >>> pY, qF, pF = model(E=10)

Transform and utility functions:
    >>> from PNMF import log_factors, factors, factor_uncertainty
    >>> from PNMF import transform_F, transform_W, get_loadings
"""

__version__ = "0.1.0"

from .models import PNMF, PoissonFactorization
from .priors import GaussianPrior
from .elbo import (
    compute_elbo,
    expected_log_likelihood,
    kl_divergence,
)
from .transforms import (
    # Factor extraction
    log_factors,
    factors,
    factor_uncertainty,
    factor_samples,
    # Model accessors
    get_loadings,
    get_prior,
    # Conditional inference
    transform_F,
    transform_W,
    # Prior utilities
    log_factors_from_prior,
    factors_from_prior,
    uncertainty_from_prior,
)

__all__ = [
    # Core classes
    "PNMF",
    "PoissonFactorization",
    "GaussianPrior",
    # ELBO functions
    "compute_elbo",
    "expected_log_likelihood",
    "kl_divergence",
    # Factor extraction
    "log_factors",
    "factors",
    "factor_uncertainty",
    "factor_samples",
    # Model accessors
    "get_loadings",
    "get_prior",
    # Conditional inference
    "transform_F",
    "transform_W",
    # Prior utilities
    "log_factors_from_prior",
    "factors_from_prior",
    "uncertainty_from_prior",
]
