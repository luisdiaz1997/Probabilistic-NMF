"""
Prior distributions for variational inference in PNMF.

This module provides prior classes for Bayesian matrix factorization,
following the GPzoo pattern with variational distributions.
"""

import torch
import torch.nn as nn
from torch import distributions

from .custom_modules import PositiveParameter


class GaussianPrior(nn.Module):
    """
    Gaussian prior for latent factors in variational inference.

    This class represents a variational distribution qF over latent factors F,
    along with a prior distribution pF (standard normal). The variational
    distribution is parameterized by mean and scale parameters.

    Args:
        y: Input data tensor of shape (D, N) where D is features, N is samples
        L: Number of latent components (default: 10)
        scale_pf: Scale parameter for the prior distribution (default: 1.0)

    Attributes:
        mean: Variational mean parameter of shape (L, N)
        scale: PositiveParameter for scale (uses softplus internally) of shape (L, N)
        scale_pf: Fixed scale for the prior distribution

    Example:
        >>> import torch
        >>> from PNMF.priors import GaussianPrior
        >>> y = torch.randn(100, 50)  # 100 features, 50 samples
        >>> prior = GaussianPrior(y, L=10)
        >>> qF, pF = prior()  # Get variational and prior distributions
        >>> F = qF.rsample((5,))  # Sample 5 times using reparameterization
    """

    def __init__(self, y, L=10, scale_pf=1.0):
        super().__init__()
        D, N = y.shape
        self.L = L
        self.N = N

        # Variational parameters
        self.mean = nn.Parameter(torch.randn(L, N))
        # Use PositiveParameter with softplus for scale (ensures positivity)
        self.scale = PositiveParameter((L, N), mode='softplus')

        # Prior hyperparameter (fixed)
        self.scale_pf = scale_pf

    def forward(self):
        """
        Get the variational and prior distributions.

        Returns:
            qF: Variational posterior distribution Normal(mean, scale.data)
            pF: Prior distribution Normal(0, scale_pf)
        """
        # PositiveParameter.data already applies softplus transformation
        qF = distributions.Normal(self.mean, self.scale.data)
        pF = distributions.Normal(
            torch.zeros_like(qF.mean),
            self.scale_pf * torch.ones_like(qF.scale)
        )
        return qF, pF

    def forward_batched(self, idx):
        """
        Get distributions for a batch of samples.

        Args:
            idx: Indices of samples to include in the batch

        Returns:
            qF: Variational distribution for the batch
            pF: Prior distribution for the batch
        """
        # Index into PositiveParameter - .data applies softplus
        scale_batched = self.scale.data[:, idx]
        qF = distributions.Normal(self.mean[:, idx], scale_batched)
        pF = distributions.Normal(
            torch.zeros_like(qF.mean),
            self.scale_pf * torch.ones_like(qF.scale)
        )
        return qF, pF

    def parameters(self):
        """Return parameters for optimization (excludes the prior hyperparameter)."""
        yield self.mean
        yield from self.scale.parameters()
