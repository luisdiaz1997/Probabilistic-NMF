"""
Probabilistic Non-negative Matrix Factorization (PNMF)

A PyTorch-based implementation of Probabilistic NMF with projected gradient
optimization, compatible with scikit-learn's API.

Example:
    >>> from PNMF import PNMF
    >>> model = PNMF(n_components=2, init='random', random_state=0)
    >>> W = model.fit_transform(X)
    >>> H = model.components_
"""

__version__ = "0.1.0"

from ._core import PNMF

__all__ = ["PNMF"]
