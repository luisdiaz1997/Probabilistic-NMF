"""
Probabilistic Non-negative Matrix Factorization (PNMF)

A PyTorch-based implementation of Probabilistic NMF with variational inference,
compatible with scikit-learn's API.

Example:
    >>> from PNMF import PNMF
    >>> model = PNMF(n_components=5, random_state=0)
    >>> W = model.fit_transform(X)
    >>> H = model.components_

For PyTorch-native usage:
    >>> from PNMF import PoissonFactorization, GaussianPrior
    >>> prior = GaussianPrior(y, L=10)
    >>> model = PoissonFactorization(prior, y, L=10)
    >>> pY, qF, pF = model(E=10)
"""

__version__ = "0.2.0"

from .models import PNMF, PoissonFactorization
from .priors import GaussianPrior

__all__ = ["PNMF", "PoissonFactorization", "GaussianPrior"]
