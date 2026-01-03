"""
Core PNMF implementation with variational inference and sklearn-like API.

This module implements Probabilistic Non-negative Matrix Factorization using
variational inference with Gaussian priors, following the GPzoo pattern.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union

from .custom_modules import PositiveParameter
from .priors import GaussianPrior


def _poisson_log_likelihood(X: torch.Tensor, rate: torch.Tensor) -> torch.Tensor:
    """
    Compute Poisson log-likelihood.

    log p(X|rate) = sum(X * log(rate) - rate)

    Note: We omit the log(X!) term as it's constant w.r.t. parameters.

    Args:
        X: Input data tensor (any shape)
        rate: Rate parameter tensor (same shape as X)

    Returns:
        Log-likelihood value (scalar)
    """
    eps = 1e-8
    rate = rate.clamp(min=eps)
    log_lik = (X * torch.log(rate) - rate).sum()
    return log_lik


class PoissonFactorization(nn.Module):
    """
    Poisson Factorization base model with variational inference.

    This model uses Poisson factorization with variational inference to learn
    non-negative factor matrices. The latent factors F are sampled from a
    Gaussian variational distribution, and the loadings W are positive parameters.

    Args:
        prior: A GaussianPrior object providing variational and prior distributions
        y: Input data tensor of shape (D, N) where D is features, N is samples
        L: Number of latent components (default: 10)
        loadings_mode: Mode for enforcing positivity on W ('softplus', 'exp', or 'projected')

    Attributes:
        prior: GaussianPrior for variational inference
        W: PositiveParameter loadings matrix of shape (D, L)
        loadings_mode: Positivity constraint mode
        D: Number of features
        N: Number of samples
        L: Number of latent components

    Example:
        >>> import torch
        >>> from PNMF.priors import GaussianPrior
        >>> from PNMF.models import PoissonFactorization
        >>> y = torch.randn(100, 50)
        >>> prior = GaussianPrior(y, L=10)
        >>> model = PoissonFactorization(prior, y, L=10)
        >>> rate, qF, pF = model(E=10)
    """

    def __init__(self, prior, y, L=10, loadings_mode='softplus'):
        super().__init__()
        self.prior = prior
        self.loadings_mode = loadings_mode
        D, N = y.shape
        self.D = D
        self.N = N
        self.L = L

        # Loadings matrix W (D x L) - positive parameters
        self.W = PositiveParameter((D, L), mode=loadings_mode)

    def get_rate(self, prior_samples):
        """
        Compute the Poisson rate from prior samples.

        Args:
            prior_samples: Samples from the prior of shape (E, L, N)
                          where E is number of samples, L is components, N is samples

        Returns:
            Z: Rate matrix of shape (E, D, N) where D is features
        """
        F = torch.exp(prior_samples)  # shape (E, L, N)
        W = self.W.data  # shape (D, L)
        # Z = W @ F gives (E, D, N)
        Z = torch.matmul(F.transpose(1, 2), W.T).transpose(1, 2)
        return Z

    def forward(self, E=10):
        """
        Forward pass generating Poisson rate.

        Args:
            E: Number of Monte Carlo samples for variational inference

        Returns:
            rate: Poisson rate tensor of shape (E, D, N)
            qF: Variational posterior distribution
            pF: Prior distribution
        """
        qF, pF = self.prior()
        F = qF.rsample((E,))  # Reparameterization trick, shape (E, L, N)
        rate = self.get_rate(F)  # shape (E, D, N)
        return rate, qF, pF

    def project_parameters(self):
        """Apply projection to ensure non-negativity (for projected gradient mode)."""
        self.W.project()


class PNMF:
    """
    Probabilistic Non-negative Matrix Factorization with variational inference.

    This class provides a scikit-learn compatible interface for variational
    PNMF using Poisson factorization with ELBO optimization.

    The model factorizes a non-negative matrix X into:
        X ≈ W @ exp(F)

    where W is the basis matrix (learned) and F is sampled from a
    variational Gaussian distribution.

    Parameters
    ----------
    n_components : int, default=10
        Number of latent components (rank of factorization).

    loadings_mode : {'softplus', 'exp', 'projected'}, default='projected'
        Method for enforcing non-negativity on W:
        - 'softplus': Use softplus transformation
        - 'exp': Use exponential transformation
        - 'projected': Use projected gradient descent (clamp after each step)

    E : int, default=10
        Number of Monte Carlo samples for ELBO estimation.

    max_iter : int, default=200
        Maximum number of iterations.

    tol : float, default=1e-4
        Tolerance for convergence.

    learning_rate : float, default=0.01
        Learning rate for Adam optimizer.

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

    elbo_ : float
        Final ELBO value.

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
        n_components: int = 10,
        loadings_mode: str = 'projected',
        E: int = 3,
        max_iter: int = 200,
        tol: float = 1e-4,
        learning_rate: float = 0.01,
        random_state: Optional[int] = None,
        verbose: bool = False,
        device: str = 'auto'
    ):
        self.n_components = n_components
        self.loadings_mode = loadings_mode
        self.E = E
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
        self.elbo_ = None
        self.n_iter_ = 0
        self._model = None
        self._prior = None

    def _validate_params(self):
        """Validate input parameters."""
        if self.n_components < 1:
            raise ValueError("n_components must be >= 1")

        if self.loadings_mode not in ['softplus', 'exp', 'projected']:
            raise ValueError("loadings_mode must be 'softplus', 'exp', or 'projected'")

        if self.max_iter < 1:
            raise ValueError("max_iter must be >= 1")

        if self.tol < 0:
            raise ValueError("tol must be >= 0")

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")

        if self.E < 1:
            raise ValueError("E must be >= 1")

    def _get_device(self):
        """Determine the device to use."""
        if self.device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(self.device)

    def _elbo(self, rate, qF, pF, X):
        """
        Compute the Evidence Lower BOund (ELBO).

        ELBO = E[log p(Y|F)] - KL[q(F) || p(F)]

        Args:
            rate: Poisson rate tensor of shape (E, D, N)
            qF: Variational posterior distribution
            pF: Prior distribution
            X: Input data tensor of shape (D, N)

        Returns:
            Negative ELBO (to minimize)
        """
        # Expected log likelihood (Monte Carlo)
        # rate has shape (E, D, N), X has shape (D, N)
        E_samples = rate.shape[0]
        X_expanded = X.unsqueeze(0).expand(E_samples, -1, -1)  # (E, D, N)

        # Sum over all dimensions, then mean over E samples
        log_lik = _poisson_log_likelihood(X_expanded, rate)

        # KL divergence
        kl = torch.distributions.kl_divergence(qF, pF).sum()

        # Negative ELBO (for minimization)
        # Note: We want to maximize ELBO, so minimize negative ELBO
        # Since log_lik is already summed over E samples implicitly,
        # we use: KL - E[log p(Y|F)]
        return kl - log_lik / E_samples

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> 'PNMF':
        """
        Fit the PNMF model to data X using variational inference.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data matrix (non-negative).

        y : Ignored
            Not used, present for scikit-learn compatibility.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self._validate_params()

        # Set random seed
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        # Convert input to torch tensor
        if isinstance(X, np.ndarray):
            X_np = X
        else:
            X_np = X.detach().cpu().numpy()

        n_samples, n_features = X_np.shape
        self.n_features_in_ = n_features
        self.n_components_ = self.n_components

        # Convert to torch tensor and transpose for model (D, N)
        X_torch = torch.from_numpy(X_np.T.astype(np.float32)).to(self._get_device())

        # Initialize prior
        self._prior = GaussianPrior(y=X_torch, L=self.n_components).to(self._get_device())

        # Initialize model
        self._model = PoissonFactorization(
            prior=self._prior,
            y=X_torch,
            L=self.n_components,
            loadings_mode=self.loadings_mode
        ).to(self._get_device())

        # Setup optimizer (W parameters + prior parameters)
        params = list(self._model.W.parameters()) + list(self._prior.parameters())
        optimizer = torch.optim.Adam(params, lr=self.learning_rate)

        # Training loop
        prev_elbo = float('-inf')

        for iteration in range(self.max_iter):
            optimizer.zero_grad()

            # Forward pass
            rate, qF, pF = self._model.forward(E=self.E)

            # Compute ELBO loss
            loss = self._elbo(rate, qF, pF, X_torch)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Project parameters if using projected gradient
            if self.loadings_mode == 'projected':
                self._model.project_parameters()

            # Check convergence
            elbo_value = -loss.item()  # Convert back to ELBO
            if self.verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}: ELBO = {elbo_value:.6f}")

            if abs(elbo_value - prev_elbo) < self.tol:
                if self.verbose:
                    print(f"Converged at iteration {iteration}")
                break

            prev_elbo = elbo_value

        self.n_iter_ = iteration + 1
        self.elbo_ = prev_elbo

        # Store components (W transposed for sklearn compatibility: n_components x n_features)
        self.components_ = self._model.W.data.detach().cpu().numpy().T

        return self

    def transform(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Transform X using the fitted model.

        Given fixed W (components_), find the optimal F for new X.

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

        # For new data, use simple NNLS to find coefficients
        n_samples = X_np.shape[0]
        H = np.random.rand(n_samples, self.n_components_).astype(np.float32) * 0.1

        # Multiplicative update for H
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
        return self.transform(X)

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
