"""
Custom optimizers for PNMF.

This module contains specialized optimizers for variational inference,
including natural gradient descent.
"""

import torch


class NaturalGradientDescent(torch.optim.Optimizer):
    """
    Natural Gradient Descent (NGD) optimizer for variational parameters.

    This optimizer implements natural gradient descent using the natural
    parameterization for Gaussian variational distributions. The natural
    gradients are computed via the Fisher information matrix.

    For a Gaussian with natural parameters (θ₁, θ₂), the natural gradient
    update is:
        θ₁ ← θ₁ - ρ * ∂L/∂η₁
        θ₂ ← θ₂ - ρ * ∂L/∂η₂

    where η₁ = μ and η₂ = s² + μ² are the expectation parameters.

    Args:
        params: Iterable of parameters to optimize (natural parameters)
        num_data: Number of data points (unused, kept for backward compatibility)
        lr: Learning rate (default: 0.1)

    Example:
        >>> # Natural parameters for Gaussian variational distribution
        >>> theta1 = nn.Parameter(torch.zeros(10, 50))
        >>> theta2 = nn.Parameter(-0.5 * torch.ones(10, 50))
        >>> optimizer = NaturalGradientDescent(
        ...     [theta1, theta2], num_data=100, lr=0.1
        ... )
        >>> optimizer.zero_grad()
        >>> loss.backward()
        >>> optimizer.step()
    """

    def __init__(self, params, num_data, lr=0.1):
        if num_data <= 0:
            raise ValueError(f"num_data must be positive, got {num_data}")
        if lr <= 0:
            raise ValueError(f"Learning rate must be positive, got {lr}")

        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step.

        Args:
            closure: Optional closure for re-evaluating the loss

        Returns:
            The loss value if closure is provided, else None
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue

                # Natural gradient update: grad is ∂L/∂η (natural gradient for θ)
                # The NaturalToMuS autograd function returns gradients w.r.t. expectation parameters
                p.add_(p.grad, alpha=-lr)

        return loss
