"""
Prior distributions for variational inference in PNMF.

This module provides prior classes for Bayesian matrix factorization,
following the GPzoo pattern with variational distributions.
"""

import torch
import torch.nn as nn
from torch import distributions

from .custom_modules import PositiveParameter, NaturalToMuS


class GaussianPrior(nn.Module):
    """
    Gaussian prior for latent factors in variational inference.

    This class represents a variational distribution qF over latent factors F,
    along with a prior distribution pF (standard normal). The variational
    distribution can be parameterized either in standard form (mean, scale)
    or in natural parameter form (theta1, theta2) for natural gradient descent.

    Args:
        y: Input data tensor of shape (D, N) where D is features, N is samples
        L: Number of latent components (default: 10)
        scale_pf: Scale parameter for the prior distribution (default: 1.0)
        use_natural_gradients: Use natural parameterization (default: False)

    Attributes:
        mean: Variational mean parameter of shape (L, N) [standard mode]
        scale: PositiveParameter for scale of shape (L, N) [standard mode]
        theta1: Natural parameter θ₁ = μ/s² of shape (L, N) [natural mode]
        theta2: Natural parameter θ₂ = -1/(2s²) of shape (L, N) [natural mode]
        scale_pf: Fixed scale for the prior distribution
        use_natural_gradients: Whether natural parameterization is used

    Example:
        >>> import torch
        >>> from PNMF.priors import GaussianPrior
        >>> y = torch.randn(100, 50)  # 100 features, 50 samples
        >>> prior = GaussianPrior(y, L=10)  # Standard mode
        >>> qF, pF = prior()
        >>> F = qF.rsample((5,))  # Sample 5 times using reparameterization

        >>> # Natural gradient mode
        >>> prior_nat = GaussianPrior(y, L=10, use_natural_gradients=True)
        >>> qF, pF = prior_nat()
    """

    def __init__(self, y, L=10, scale_pf=1.0, use_natural_gradients=False):
        super().__init__()
        D, N = y.shape
        self.L = L
        self.N = N
        self.use_natural_gradients = use_natural_gradients
        self.scale_pf = scale_pf

        if use_natural_gradients:
            # Natural parameters
            # θ₁ = μ/s² initialized to 0 (corresponds to μ=0)
            # θ₂ = -1/(2s²) initialized to -0.5 (corresponds to s²=1)
            self.theta1 = nn.Parameter(torch.zeros(L, N))
            self.theta2 = nn.Parameter(-0.5 * torch.ones(L, N))
        else:
            # Standard parameterization
            # Variational parameters
            self.mean = nn.Parameter(torch.randn(L, N))
            # Use PositiveParameter with softplus for scale (ensures positivity)
            self.scale = PositiveParameter((L, N), mode='softplus')

    def forward(self):
        """
        Get the variational and prior distributions.

        Returns:
            qF: Variational posterior distribution Normal(mean, scale)
            pF: Prior distribution Normal(0, scale_pf)
        """
        if self.use_natural_gradients:
            # Convert natural parameters to (mu, s) using custom autograd
            # The backward pass will compute natural gradients
            mu, s = NaturalToMuS.apply(self.theta1, self.theta2)
            scale_constrained = s.clamp(min=1e-8)
        else:
            # Standard parameterization
            # PositiveParameter.data already applies softplus transformation
            # Add small epsilon to ensure scale > 0 (Normal distribution requires strictly positive)
            mu = self.mean
            scale_constrained = self.scale.data.clamp(min=1e-8)

        qF = distributions.Normal(mu, scale_constrained)
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
        if self.use_natural_gradients:
            # Convert natural parameters to (mu, s)
            mu_batched, s_batched = NaturalToMuS.apply(
                self.theta1[:, idx], self.theta2[:, idx]
            )
            scale_batched = s_batched.clamp(min=1e-8)
        else:
            # Index into PositiveParameter - .data applies softplus
            mu_batched = self.mean[:, idx]
            scale_batched = self.scale.data[:, idx].clamp(min=1e-8)

        qF = distributions.Normal(mu_batched, scale_batched)
        pF = distributions.Normal(
            torch.zeros_like(qF.mean),
            self.scale_pf * torch.ones_like(qF.scale)
        )
        return qF, pF

    def parameters(self):
        """Return parameters for optimization (excludes the prior hyperparameter)."""
        if self.use_natural_gradients:
            yield self.theta1
            yield self.theta2
        else:
            yield self.mean
            yield from self.scale.parameters()

    def natural_parameters(self):
        """Return natural parameters (theta1, theta2) for NGD optimizer."""
        if self.use_natural_gradients:
            return [self.theta1, self.theta2]
        else:
            raise RuntimeError("Natural parameters not available. Set use_natural_gradients=True.")
