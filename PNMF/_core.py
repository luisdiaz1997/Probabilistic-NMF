"""
Core PNMF implementation with sklearn-like API.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Poisson
from typing import Optional, Union, Tuple

from ._utils import PositiveParameter


class PNMFModel(nn.Module):
    """
    Internal PyTorch model for Probabilistic NMF.

    This model uses Poisson factorization with projected gradient optimization
    to ensure non-negativity constraints on the factor matrices.

    Args:
        n_features: Number of features (rows of input matrix)
        n_components: Number of latent components
        n_samples: Number of samples (columns of input matrix)
        init_mode: Initialization mode ('softplus', 'exp', or 'projected')
    """

    def __init__(self, n_features: int, n_components: int, n_samples: int, init_mode: str = 'softplus'):
        super().__init__()
        self.n_features = n_features
        self.n_components = n_components
        self.n_samples = n_samples

        # Basis matrix W (n_features x n_components) - positive parameters
        self.W = PositiveParameter((n_features, n_components), mode=init_mode)

        # Activation matrix H (n_components x n_samples)
        self.H = PositiveParameter((n_components, n_samples), mode=init_mode)

    def forward(self, E: int = 1) -> Tuple[torch.distributions.Distribution, torch.Tensor]:
        """
        Forward pass generating Poisson rate.

        Args:
            E: Number of samples for Monte Carlo estimation

        Returns:
            pY: Poisson distribution with rate W @ H
            rate: The rate parameter (for direct access)
        """
        # Get positive matrices
        W_pos = self.W.data  # (n_features, n_components)
        H_pos = self.H.data  # (n_components, n_samples)

        # Compute rate: W @ H gives (n_features, n_samples)
        rate = torch.matmul(W_pos, H_pos)

        # Poisson distribution
        pY = Poisson(rate=rate)

        return pY, rate

    def reconstruct(self) -> torch.Tensor:
        """Return the reconstructed matrix."""
        _, rate = self.forward(E=1)
        return rate

    def project_parameters(self):
        """Apply projection to ensure non-negativity (for projected gradient mode)."""
        self.W.project()
        self.H.project()


class PNMF:
    """
    Probabilistic Non-negative Matrix Factorization.

    This class provides a scikit-learn compatible interface for probabilistic
    NMF using Poisson factorization with projected gradient optimization.

    The model factorizes a non-negative matrix X into two non-negative matrices:
        X ≈ W @ H

    where W is the basis matrix and H is the coefficient matrix.

    Parameters
    ----------
    n_components : int, default=2
        Number of latent components (rank of factorization).

    init : {'random', 'custom'}, default='random'
        Initialization method:
        - 'random': Random initialization with positive values
        - 'custom': Use custom provided W and H (must set W and H before fit)

    init_mode : {'softplus', 'exp', 'projected'}, default='projected'
        Method for enforcing non-negativity:
        - 'softplus': Use softplus transformation
        - 'exp': Use exponential transformation
        - 'projected': Use projected gradient descent (clamp after each step)

    max_iter : int, default=200
        Maximum number of iterations.

    tol : float, default=1e-4
        Tolerance for convergence. If the reconstruction error improvement is
        less than tol, optimization stops.

    learning_rate : float, default=0.01
        Learning rate for gradient descent.

    random_state : int, default=None
        Random seed for reproducibility.

    verbose : bool, default=False
        Whether to print progress messages.

    device : {'cpu', 'cuda', 'auto'}, default='auto'
        Device to use for computation.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        The basis matrix W (transposed for sklearn compatibility).

    n_components_ : int
        The number of components.

    n_features_in_ : int
        Number of features seen during fit.

    reconstruction_err_ : float
        Final reconstruction error.

    n_iter_ : int
        Actual number of iterations performed.

    Examples
    --------
    >>> import numpy as np
    >>> from PNMF import PNMF
    >>> X = np.random.rand(100, 50)  # 100 samples, 50 features
    >>> model = PNMF(n_components=5, random_state=42)
    >>> W = model.fit_transform(X)  # W: (100, 5) transformed data
    >>> H = model.components_  # H: (5, 50) components
    >>> X_reconstructed = model.inverse_transform(W)
    """

    def __init__(
        self,
        n_components: int = 2,
        init: str = 'random',
        init_mode: str = 'projected',
        max_iter: int = 200,
        tol: float = 1e-4,
        learning_rate: float = 0.01,
        random_state: Optional[int] = None,
        verbose: bool = False,
        device: str = 'auto'
    ):
        self.n_components = n_components
        self.init = init
        self.init_mode = init_mode
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.verbose = verbose
        self.device = device

        # Attributes set during fit
        self.components_ = None
        self.n_components_ = n_components
        self.n_features_in_ = None
        self.reconstruction_err_ = None
        self.n_iter_ = 0
        self._model = None
        self._H = None

    def _validate_params(self):
        """Validate input parameters."""
        if self.n_components < 1:
            raise ValueError("n_components must be >= 1")

        if self.init not in ['random', 'custom']:
            raise ValueError("init must be 'random' or 'custom'")

        if self.init_mode not in ['softplus', 'exp', 'projected']:
            raise ValueError("init_mode must be 'softplus', 'exp', or 'projected'")

        if self.max_iter < 1:
            raise ValueError("max_iter must be >= 1")

        if self.tol < 0:
            raise ValueError("tol must be >= 0")

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")

    def _get_device(self):
        """Determine the device to use."""
        if self.device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(self.device)

    def _initialize(
        self,
        X: np.ndarray,
        W: Optional[np.ndarray] = None,
        H: Optional[np.ndarray] = None
    ) -> PNMFModel:
        """Initialize the model."""
        n_samples, n_features = X.shape

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        model = PNMFModel(
            n_features=n_features,
            n_components=self.n_components,
            n_samples=n_samples,
            init_mode=self.init_mode
        ).to(self._get_device())

        if self.init == 'custom':
            if W is not None and H is not None:
                model.W.data = torch.from_numpy(W.astype(np.float32)).to(self._get_device())
                model.H.data = torch.from_numpy(H.astype(np.float32)).to(self._get_device())
            else:
                raise ValueError("Custom init requires W and H to be provided")

        return model

    def _poisson_nll(self, X: torch.Tensor, rate: torch.Tensor) -> torch.Tensor:
        """
        Compute Poisson negative log-likelihood.

        Args:
            X: Input data (n_samples, n_features)
            rate: Predicted rate (n_samples, n_features)

        Returns:
            Negative log-likelihood
        """
        # Poisson NLL: rate - X * log(rate) + log(X!)
        # We omit log(X!) as it's constant w.r.t. parameters
        eps = 1e-8
        rate = rate.clamp(min=eps)
        nll = rate - X * torch.log(rate)
        return nll.sum()

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Optional[Union[np.ndarray, torch.Tensor]] = None,
        W: Optional[Union[np.ndarray, torch.Tensor]] = None,
        H: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> 'PNMF':
        """
        Fit the PNMF model to data X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data matrix (non-negative).

        y : Ignored
            Not used, present for scikit-learn compatibility.

        W : array-like of shape (n_features, n_components), optional
            Initial basis matrix for custom initialization.

        H : array-like of shape (n_components, n_samples), optional
            Initial coefficient matrix for custom initialization.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self._validate_params()

        # Convert input to torch tensor
        if isinstance(X, np.ndarray):
            X_np = X
        else:
            X_np = X.detach().cpu().numpy()

        n_samples, n_features = X_np.shape
        self.n_features_in_ = n_features
        self.n_components_ = self.n_components

        # Convert to torch tensor
        X_torch = torch.from_numpy(X_np.astype(np.float32)).to(self._get_device())

        # Convert W and H if provided
        W_torch = None
        H_torch = None
        if W is not None:
            W_torch = torch.from_numpy(np.asarray(W).astype(np.float32)).to(self._get_device())
        if H is not None:
            H_torch = torch.from_numpy(np.asarray(H).astype(np.float32)).to(self._get_device())

        # Initialize model
        self._model = self._initialize(X_np, W=W_torch, H=H_torch)

        # Setup optimizer
        params = list(self._model.W.parameters()) + list(self._model.H.parameters())
        optimizer = torch.optim.Adam(params, lr=self.learning_rate)

        # Training loop
        prev_loss = float('inf')

        for iteration in range(self.max_iter):
            optimizer.zero_grad()

            # Forward pass
            _, rate = self._model.forward(E=1)

            # Compute loss (Poisson negative log-likelihood)
            # Transpose rate to match X shape: rate is (n_features, n_samples), X is (n_samples, n_features)
            rate_T = rate.t()
            loss = self._poisson_nll(X_torch, rate_T)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Project parameters if using projected gradient
            if self.init_mode == 'projected':
                self._model.project_parameters()

            # Check convergence
            loss_value = loss.item()
            if self.verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}: Loss = {loss_value:.6f}")

            if abs(prev_loss - loss_value) < self.tol:
                if self.verbose:
                    print(f"Converged at iteration {iteration}")
                break

            prev_loss = loss_value

        self.n_iter_ = iteration + 1
        self.reconstruction_err_ = prev_loss

        # Store components (W transposed for sklearn compatibility: n_components x n_features)
        self.components_ = self._model.W.data.detach().cpu().numpy().T
        self._H = self._model.H.data.detach().cpu().numpy()

        return self

    def transform(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Transform X using the fitted model.

        Given fixed W (components_), find the optimal H for new X.
        This uses multiplicative update rules for non-negative least squares.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data matrix.

        Returns
        -------
        H : ndarray of shape (n_samples, n_components)
            Transformed data (coefficient matrix).
        """
        if self.components_ is None:
            raise ValueError("Model has not been fitted yet.")

        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()

        X_np = np.asarray(X).astype(np.float32)
        W = self.components_.T  # (n_features, n_components)

        # Initialize H randomly
        n_samples = X_np.shape[0]
        H = np.random.rand(n_samples, self.n_components_).astype(np.float32) * 0.1

        # Multiplicative update for H
        # H <- H * (W^T @ X) / (W^T @ W @ H + eps)
        for _ in range(100):
            numerator = X_np @ W  # (n_samples, n_components)
            denominator = H @ (W.T @ W) + 1e-8  # (n_samples, n_components)
            H = H * numerator / denominator

        return H

    def fit_transform(self, X: Union[np.ndarray, torch.Tensor], **kwargs) -> np.ndarray:
        """
        Fit the model and transform X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data matrix.

        **kwargs : Additional arguments to pass to fit()

        Returns
        -------
        H : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        self.fit(X, **kwargs)
        return self._H.T  # (n_samples, n_components)

    def inverse_transform(self, H: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Transform data back to its original space.

        Parameters
        ----------
        H : array-like of shape (n_samples, n_components)
            Transformed data in component space.

        Returns
        -------
        X_reconstructed : ndarray of shape (n_samples, n_features)
            Reconstructed data in original space.
        """
        if self.components_ is None:
            raise ValueError("Model has not been fitted yet.")

        if isinstance(H, torch.Tensor):
            H = H.detach().cpu().numpy()

        H = np.asarray(H)
        # X = H @ W^T = H @ components_
        return H @ self.components_
