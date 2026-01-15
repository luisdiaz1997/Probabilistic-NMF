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
"""

__version__ = "0.1.0"

from .models import PNMF, PoissonFactorization
from .priors import GaussianPrior
from .elbo import (
    compute_elbo,
    compute_expected_log_lik,
    compute_kl_divergence,
)

__all__ = [
    "PNMF",
    "PoissonFactorization",
    "GaussianPrior",
    "compute_elbo",
    "compute_expected_log_lik",
    "compute_kl_divergence",
]
