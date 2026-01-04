"""
Utility classes for constrained parameters.

Adapted from GPzoo: https://github.com/luisdiaz1997/GPzoo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod
from typing import Union, Tuple


class NaturalToMuS(torch.autograd.Function):
    """
    Custom autograd function for converting natural parameters to (μ, s).

    Natural parameterization for Gaussian variational distribution:
        θ₁ = μ/s²  (natural parameter for mean)
        θ₂ = -1/(2s²)  (natural parameter for precision)

    Forward pass: converts (θ₁, θ₂) → (μ, s)
    Backward pass: returns gradients w.r.t. expectation parameters (η₁, η₂)
                   where η₁ = μ and η₂ = s² + μ²

    This enables natural gradient descent for variational inference.

    Example:
        >>> theta1 = torch.randn(10, 50, requires_grad=True)
        >>> theta2 = -0.5 * torch.ones(10, 50, requires_grad=True)
        >>> mu, s = NaturalToMuS.apply(theta1, theta2)
    """

    @staticmethod
    def forward(ctx, theta1, theta2, jitter=1e-6):
        """
        Convert natural parameters to (μ, s).

        Natural parameters:
            θ₁ = μ/s²
            θ₂ = -1/(2s²)

        Args:
            ctx: Context object for backward pass
            theta1: Natural parameter θ₁ of shape (L, N)
            theta2: Natural parameter θ₂ of shape (L, N)
            jitter: Small value for numerical stability

        Returns:
            mu: Mean parameter of shape (L, N)
            s: Standard deviation of shape (L, N)
        """
        # s² = -1/(2θ₂)
        s_squared = -1.0 / (2.0 * theta2 + jitter)
        s = torch.sqrt(s_squared.clamp(min=jitter))

        # μ = s² * θ₁
        mu = s_squared * theta1

        ctx.save_for_backward(mu, s)
        return mu, s

    @staticmethod
    def backward(ctx, dout_dmu, dout_ds):
        """
        Compute gradients w.r.t. expectation parameters (η₁, η₂).

        The natural gradients are computed as:
            ∂L/∂η₁ = ∂L/∂μ - 2μ·∂L/∂(s²)
            ∂L/∂η₂ = ∂L/∂(s²)

        where ∂L/∂(s²) = (∂L/∂s) * (∂s/∂s²) = (∂L/∂s) / (2s)

        Args:
            ctx: Context object with saved tensors
            dout_dmu: Gradient of loss w.r.t. μ
            dout_ds: Gradient of loss w.r.t. s

        Returns:
            dout_deta1: Gradient w.r.t. η₁
            dout_deta2: Gradient w.r.t. η₂
            None: For jitter argument
        """
        mu, s = ctx.saved_tensors

        # Convert gradient w.r.t. s to gradient w.r.t. s²
        # ∂s/∂s² = 1/(2s)
        dout_ds_squared = dout_ds / (2.0 * s + 1e-8)

        # Natural gradients (gradients w.r.t. expectation params)
        # η₁ = μ
        # η₂ = s² + μ²
        dout_deta1 = dout_dmu - 2.0 * mu * dout_ds_squared
        dout_deta2 = dout_ds_squared

        return dout_deta1, dout_deta2, None


class ConstrainedParameter(nn.Module):
    """
    Base class for parameters with constraints.

    Subclasses implement _to_constrained() and _to_unconstrained() to define
    the bijective mapping between raw (unconstrained) and constrained spaces.

    The .data property returns the constrained value, and the underlying
    unconstrained parameter is stored in ._raw (an nn.Parameter).

    Args:
        shape: Shape of the constrained parameter.
        mode: Constraint mode (subclass-specific).
    """

    def __init__(self, shape: Union[int, Tuple[int, ...]], mode: str):
        super().__init__()
        self.mode = mode
        self._shape = (shape,) if isinstance(shape, int) else tuple(shape)
        # Subclass should set self._raw in its __init__ after calling super().__init__

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the constrained parameter."""
        return self._shape

    @abstractmethod
    def _to_constrained(self, raw: torch.Tensor) -> torch.Tensor:
        """Convert raw parameter to constrained space."""
        pass

    @abstractmethod
    def _to_unconstrained(self, constrained: torch.Tensor) -> torch.Tensor:
        """Convert constrained parameter to raw space."""
        pass

    @abstractmethod
    def _init_raw(self) -> torch.Tensor:
        """Initialize raw parameter."""
        pass

    # ==================== Common interface ====================

    @property
    def data(self) -> torch.Tensor:
        """Returns the constrained value."""
        return self._to_constrained(self._raw)

    @data.setter
    def data(self, target: torch.Tensor):
        """Sets parameter from a target constrained tensor."""
        self._raw.data = self._to_unconstrained(target)

    @property
    def requires_grad(self) -> bool:
        """Returns requires_grad status of the underlying parameter."""
        return self._raw.requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool):
        """Sets requires_grad on the underlying parameter."""
        self._raw.requires_grad = value

    @property
    def device(self) -> torch.device:
        """Returns the device of the underlying parameter."""
        return self._raw.device

    @property
    def dtype(self) -> torch.dtype:
        """Returns the dtype of the underlying parameter."""
        return self._raw.dtype

    @property
    def raw(self) -> nn.Parameter:
        """Direct access to the underlying unconstrained parameter."""
        return self._raw

    def freeze(self):
        """Freeze the parameter (disable gradient computation)."""
        self._raw.requires_grad = False

    def unfreeze(self):
        """Unfreeze the parameter (enable gradient computation)."""
        self._raw.requires_grad = True

    def project(self):
        """Project parameters to satisfy constraints. Override for projected modes."""
        pass

    # ==================== Tensor-like interface ====================

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        def unwrap(x):
            return x.data if isinstance(x, ConstrainedParameter) else x

        new_args = [unwrap(arg) for arg in args]
        new_kwargs = {k: unwrap(v) for k, v in kwargs.items()}
        return func(*new_args, **new_kwargs)

    def __getitem__(self, idx):
        """Allow subscripting to index into the constrained value."""
        return self.data[idx]

    def __len__(self):
        return self._shape[0] if self._shape else 1

    def dim(self):
        return len(self._shape)

    def size(self, dim=None):
        if dim is None:
            return torch.Size(self._shape)
        return self._shape[dim]

    def numel(self):
        return self._raw.numel()

    def detach(self):
        """Returns a detached copy of the constrained value."""
        return self.data.detach()

    def clone(self):
        """Returns a cloned copy of the constrained value."""
        return self.data.clone()

    def cpu(self):
        """Move to CPU."""
        self._raw.data = self._raw.data.cpu()
        return self

    def cuda(self, device=None):
        """Move to CUDA."""
        self._raw.data = self._raw.data.cuda(device)
        return self

    def to(self, *args, **kwargs):
        """Move to device/dtype."""
        self._raw.data = self._raw.data.to(*args, **kwargs)
        return self

    def float(self):
        """Convert to float32."""
        self._raw.data = self._raw.data.float()
        return self

    def double(self):
        """Convert to float64."""
        self._raw.data = self._raw.data.double()
        return self

    def half(self):
        """Convert to float16."""
        self._raw.data = self._raw.data.half()
        return self

    def contiguous(self):
        """Returns a contiguous copy of the constrained value."""
        return self.data.contiguous()

    def numpy(self):
        """Returns numpy array of the constrained value."""
        return self.data.detach().cpu().numpy()

    @property
    def grad(self):
        """Returns the gradient of the underlying parameter."""
        return self._raw.grad

    @property
    def is_cuda(self):
        """Check if on CUDA."""
        return self._raw.is_cuda

    @property
    def is_leaf(self):
        """Check if leaf tensor."""
        return self._raw.is_leaf


class PositiveParameter(ConstrainedParameter):
    """
    Parameter constrained to be positive.

    Supports any shape - batching works automatically.

    Args:
        shape: int or tuple for the parameter shape.
        mode: 'softplus', 'exp', or 'projected' for ensuring positivity.

    The mode determines how positivity is enforced:
        - 'softplus': Uses softplus(x) = log(1 + exp(x)) transformation
        - 'exp': Uses exp(x) transformation
        - 'projected': Uses projected gradient descent (clamps to >= 0 after each step)
    """

    def __init__(self, shape: Union[int, Tuple[int, ...]], mode: str = 'softplus'):
        if mode not in ['softplus', 'exp', 'projected']:
            raise ValueError(f"Unknown mode: {mode}. Choose 'softplus', 'exp', or 'projected'")
        super().__init__(shape, mode)
        self._raw = nn.Parameter(self._init_raw())

    def _init_raw(self) -> torch.Tensor:
        if self.mode == 'projected':
            return torch.rand(self._shape)
        else:
            return torch.randn(self._shape)

    def _to_constrained(self, raw: torch.Tensor) -> torch.Tensor:
        if self.mode == 'softplus':
            return F.softplus(raw)
        elif self.mode == 'exp':
            return torch.exp(raw)
        else:  # projected
            return raw

    def _to_unconstrained(self, constrained: torch.Tensor) -> torch.Tensor:
        if self.mode == 'softplus':
            # Inverse softplus: log(exp(x) - 1)
            return torch.log(torch.exp(constrained) - 1)
        elif self.mode == 'exp':
            return torch.log(constrained)
        else:  # projected
            return constrained.clamp(min=0.0)

    def project(self):
        """
        Project parameters to satisfy constraints.

        For 'projected' mode, this clamps values to be >= 0.
        Call this after optimizer.step() when using projected gradients.
        """
        if self.mode == 'projected':
            with torch.no_grad():
                self._raw.data.clamp_(min=0.0)

    def __repr__(self):
        grad_str = ", requires_grad=True" if self._raw.requires_grad else ""
        return f"PositiveParameter(shape={self._shape}, mode='{self.mode}'{grad_str})"
