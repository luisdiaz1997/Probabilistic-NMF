"""
Core PNMF implementation with variational inference and sklearn-like API.

This module implements Probabilistic Non-negative Matrix Factorization using
variational inference with Gaussian priors, following the GPzoo pattern.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union

from tqdm.auto import tqdm

from .custom_modules import PositiveParameter
from .optimizers import NaturalGradientDescent
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
        mode: ELBO computation mode ('simple', 'expanded', or 'lower-bound')
            - 'simple': Use torch.distributions.Poisson.log_prob() directly
            - 'expanded': Use hybrid Monte Carlo + analytic expectation (default)
            - 'lower-bound': Use Jensen's lower bound (fully analytic, no MC sampling)

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

    def __init__(self, prior, y, L=10, loadings_mode='softplus', mode='expanded'):
        super().__init__()
        self.prior = prior
        self.loadings_mode = loadings_mode
        self.mode = mode
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
        Z = torch.matmul(W, F)  # shape (E, D, N)
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
        X ≈ exp(F) @ W.T

    where F is the latent factor matrix (sample-specific, sampled from a
    variational Gaussian distribution) and W is the loading matrix (learned).

    Note: For sklearn API compatibility, fit_transform returns exp(F) (called
    the "transformed data") and components_ stores W.T (called the "components").

    Parameters
    ----------
    n_components : int, default=10
        Number of latent components (rank of factorization).

    loadings_mode : {'softplus', 'exp', 'projected'}, default='projected'
        Method for enforcing non-negativity on W:
        - 'softplus': Use softplus transformation
        - 'exp': Use exponential transformation
        - 'projected': Use projected gradient descent (clamp after each step)

    mode : {'simple', 'expanded', 'lower-bound'}, default='expanded'
        ELBO computation mode:
        - 'simple': Use torch.distributions.Poisson.log_prob() directly
        - 'expanded': Use hybrid Monte Carlo + analytic expectation (default)
        - 'lower-bound': Use Jensen's lower bound (fully analytic, no MC sampling)

    training_mode : {'standard', 'natural'}, default='standard'
        Training mode for variational parameters:
        - 'standard': Standard gradient descent with Adam/other optimizer
        - 'natural': Natural gradient descent with dual optimizers (NGD for variational, Adam for W)

    E : int, default=10
        Number of Monte Carlo samples for ELBO estimation.

    max_iter : int, default=200
        Maximum number of iterations.

    tol : float, default=1e-4
        Tolerance for convergence.

    learning_rate : float, default=0.01
        Learning rate for the optimizer.

    optimizer : {'Adam', 'AdamW', 'NAdam', 'SGD', 'RMSprop'}, default='Adam'
        Optimizer to use for training (applies to W parameters in natural mode).

    random_state : int, default=None
        Random seed for reproducibility.

    verbose : bool, default=False
        Whether to print progress messages.

    device : {'cpu', 'cuda', 'mps', 'auto'}, default='auto'
        Device to use for computation. 'auto' will select mps (Apple Silicon),
        cuda (NVIDIA), or cpu in that order based on availability.

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
    >>> transformed = model.fit_transform(X)  # exp(F): (100, 5) transformed data
    >>> components = model.components_        # W.T: (5, 50) components
    >>> X_reconstructed = model.inverse_transform(transformed)
    """

    def __init__(
        self,
        n_components: int = 10,
        loadings_mode: str = 'projected',
        mode: str = 'expanded',
        training_mode: str = 'standard',
        E: int = 3,
        max_iter: int = 200,
        tol: float = 1e-4,
        learning_rate: float = 0.01,
        optimizer: str = 'Adam',
        random_state: Optional[int] = None,
        verbose: bool = False,
        device: str = 'auto'
    ):
        self.n_components = n_components
        self.loadings_mode = loadings_mode
        self.mode = mode
        self.training_mode = training_mode
        self.E = E
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        self.optimizer = optimizer
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
        self._optimizer = None
        self._w_optimizer = None

    def _validate_params(self):
        """Validate input parameters."""
        if self.n_components < 1:
            raise ValueError("n_components must be >= 1")

        if self.loadings_mode not in ['softplus', 'exp', 'projected']:
            raise ValueError("loadings_mode must be 'softplus', 'exp', or 'projected'")

        if self.mode not in ['simple', 'expanded', 'lower-bound']:
            raise ValueError("mode must be 'simple', 'expanded', or 'lower-bound'")

        if self.training_mode not in ['standard', 'natural']:
            raise ValueError("training_mode must be 'standard' or 'natural'")

        if self.max_iter < 1:
            raise ValueError("max_iter must be >= 1")

        if self.tol < 0:
            raise ValueError("tol must be >= 0")

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")

        if self.E < 1:
            raise ValueError("E must be >= 1")

        if self.optimizer not in ['Adam', 'AdamW', 'NAdam', 'SGD', 'RMSprop']:
            raise ValueError("optimizer must be 'Adam', 'AdamW', 'NAdam', 'SGD', or 'RMSprop'")

    def _get_device(self):
        """Determine the device to use."""
        if self.device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        return torch.device(self.device)

    def _elbo(self, rate, qF, pF, X):
        """
        Compute the Evidence Lower BOund (ELBO).

        This method dispatches to the appropriate ELBO computation based on self.mode:
        - 'simple': Uses torch.distributions.Poisson.log_prob() directly
        - 'expanded': Uses hybrid Monte Carlo + analytic expectation
        - 'lower-bound': Uses Jensen's lower bound (fully analytic, no MC)

        ELBO = E[log p(Y|F)] - KL[q(F) || p(F)]

        Args:
            rate: Poisson rate tensor of shape (E, D, N) [unused for lower-bound]
            qF: Variational posterior distribution with mean and scale
            pF: Prior distribution
            X: Input data tensor of shape (D, N)

        Returns:
            Negative ELBO (to minimize)
        """
        if self.mode == 'simple':
            return self._elbo_simple(rate, qF, pF, X)
        elif self.mode == 'lower-bound':
            return self._elbo_lower_bound(qF, pF, X)
        else:  # 'expanded'
            return self._elbo_expanded(rate, qF, pF, X)

    def _elbo_simple(self, rate, qF, pF, X):
        """
        Compute ELBO using full Monte Carlo estimation with torch.distributions.Poisson.

        This uses torch.distributions.Poisson.log_prob() for computing the
        log-likelihood, providing a clean and numerically stable implementation.

        ELBO = E[log p(Y|F)] - KL[q(F) || p(F)]

        The Poisson log-likelihood is:
            log p(X|rate) = X * log(rate) - rate - log(X!)

        Args:
            rate: Poisson rate tensor of shape (E, D, N)
            qF: Variational posterior distribution with mean and scale
            pF: Prior distribution
            X: Input data tensor of shape (D, N)

        Returns:
            Negative ELBO (to minimize)
        """
        E_samples = rate.shape[0]

        # Expand X to match rate shape: (D, N) -> (E, D, N)
        X_expanded = X.unsqueeze(0).expand(E_samples, -1, -1)

        # Use torch.distributions.Poisson for clean, numerically stable computation
        poisson_dist = torch.distributions.Poisson(rate=rate)
        log_lik_mc = poisson_dist.log_prob(X_expanded)

        # Expected log likelihood via Monte Carlo: (1/E) * sum_e log p(Y|F_e)
        log_lik = log_lik_mc.sum() / E_samples

        # KL divergence
        kl = torch.distributions.kl_divergence(qF, pF).sum()

        # Negative ELBO (for minimization)
        return kl - log_lik

    def _elbo_lower_bound(self, qF, pF, X):
        """
        Compute ELBO using Jensen's lower bound (fully analytic, no MC sampling).

        Uses Jensen's inequality for the log-sum-exp term:
            E[log Σ W * exp(F)] ≥ log Σ W * exp(E[F])

        For Gaussian variational distribution with mean μ and variance σ²:
            E[F] = μ

        This gives a true lower bound on the ELBO with NO Monte Carlo sampling.

        The expected log-likelihood is computed as:
            E[log p(Y|F)] ≈ Y * log(Σ W * exp(μ)) - Σ W * exp(μ + σ²/2) - log(Y!)

        ELBO = E[log p(Y|F)] - KL[q(F) || p(F)]

        Args:
            qF: Variational posterior distribution with mean and scale
            pF: Prior distribution
            X: Input data tensor of shape (D, N)

        Returns:
            Negative ELBO (to minimize)
        """
        D, N = X.shape

        # Get variational parameters
        mu = qF.mean  # (L, N)
        sigma = qF.scale  # (L, N)

        # --- First term (Jensen lower bound): Y_ij * log(Σ_l W_jl * exp(μ_il)) ---
        # Lower bound: E[log Σ W * exp(F)] ≥ log Σ W * exp(E[F]) = log Σ W * exp(μ)
        exp_mu = torch.exp(mu)  # (L, N)
        W = self._model.W.data  # (D, L)

        # rate_lower_bound has shape (D, N): sum over L of W_jl * exp(μ_il)
        rate_lower_bound = torch.matmul(W, exp_mu)  # (D, N)

        eps = 1e-8
        rate_clamped = rate_lower_bound.clamp(min=eps)

        # Y * log(rate) using the lower bound
        term1_lower = (X * torch.log(rate_clamped)).sum()

        # --- Second term (analytic): Σ_l W_jl * E[exp(F_il)] ---
        # E[exp(F)] = exp(μ + σ²/2)
        exp_expectation = torch.exp(mu + 0.5 * sigma ** 2)  # (L, N)
        term2_analytic = torch.matmul(W, exp_expectation).sum()  # scalar

        # Expected log likelihood (including Poisson normalization)
        log_lik = term1_lower - term2_analytic - torch.lgamma(X + 1).sum()

        # KL divergence
        kl = torch.distributions.kl_divergence(qF, pF).sum()

        # Negative ELBO (for minimization)
        # Note: Since we use a lower bound on the log term, this is a true lower bound on ELBO
        return kl - log_lik

    def _elbo_expanded(self, rate, qF, pF, X):
        """
        Compute the Evidence Lower BOund (ELBO) using the expanded expectation form.

        The expected log-likelihood is computed as:
            E[log p(Y|F)] = Y * E[log(sum(W * exp(F)))] - sum(W * E[exp(F)]) - sum(log(Y!))

        where:
        - The first term uses Monte Carlo estimation (requires log of sum)
        - The second term is computed analytically using E[exp(F)] = exp(mu + sigma^2/2)
        - The third term is the Poisson normalization constant (log factorial)

        ELBO = E[log p(Y|F)] - KL[q(F) || p(F)]

        Args:
            rate: Poisson rate tensor of shape (E, D, N)
            qF: Variational posterior distribution with mean and scale
            pF: Prior distribution
            X: Input data tensor of shape (D, N)

        Returns:
            Negative ELBO (to minimize)
        """
        E_samples = rate.shape[0]
        D, N = X.shape

        # --- First term: Y_ij * E_q[log sum_l W_jl * exp(F_il)] ---
        # This requires Monte Carlo estimation
        eps = 1e-8
        rate_clamped = rate.clamp(min=eps)
        X_expanded = X.unsqueeze(0).expand(E_samples, -1, -1)  # (E, D, N)

        # E[Y * log(rate)] = (1/E) * sum_e Y * log(rate_e)
        term1_mc = (X_expanded * torch.log(rate_clamped)).sum() / E_samples

        # --- Second term: sum_l W_jl * E_q[exp(F_il)] ---
        # Computed analytically using E[exp(F)] = exp(mu + sigma^2/2)
        # qF.mean has shape (L, N), qF.scale has shape (L, N)
        mu = qF.mean  # (L, N)
        sigma = qF.scale  # (L, N)

        # E[exp(F_il)] = exp(mu_il + sigma_il^2 / 2)
        exp_expectation = torch.exp(mu + 0.5 * sigma ** 2)  # (L, N)

        # W has shape (D, L), need to compute: sum_j sum_l W_jl * exp_expectation[l, n]
        W = self._model.W.data  # (D, L)
        term2_analytic = torch.matmul(W, exp_expectation).sum()  # scalar

        # Expected log likelihood (including the Poisson normalization -log(Y!) term)
        # Using lgamma(X+1) = log(X!) for the Poisson PMF normalization
        log_lik = term1_mc - term2_analytic - torch.lgamma(X + 1).sum()

        # KL divergence
        kl = torch.distributions.kl_divergence(qF, pF).sum()

        # Negative ELBO (for minimization)
        return kl - log_lik

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Optional[Union[np.ndarray, torch.Tensor]] = None,
        return_history: bool = False
    ) -> Union['PNMF', tuple[list[float], 'PNMF']]:
        """
        Fit the PNMF model to data X using variational inference.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data matrix (non-negative).

        y : Ignored
            Not used, present for scikit-learn compatibility.

        return_history : bool, default=False
            If True, returns a tuple (history, self) where history is a list
            of ELBO values during training.

        Returns
        -------
        self : object
            Returns the instance itself (or (history, self) if return_history=True).
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

        # Initialize prior with natural gradient mode if specified
        use_natural_gradients = (self.training_mode == 'natural')
        self._prior = GaussianPrior(
            y=X_torch,
            L=self.n_components,
            use_natural_gradients=use_natural_gradients
        ).to(self._get_device())

        # Initialize model
        self._model = PoissonFactorization(
            prior=self._prior,
            y=X_torch,
            L=self.n_components,
            loadings_mode=self.loadings_mode,
            mode=self.mode
        ).to(self._get_device())

        # Setup optimizers based on training mode
        if self.training_mode == 'natural':
            # Natural gradient mode: dual optimizers
            # NGD for variational parameters (natural params)
            # Use smaller learning rate for NGD (0.1x) for stability
            nat_params = self._prior.natural_parameters()
            self._optimizer = NaturalGradientDescent(
                nat_params, num_data=n_samples, lr=self.learning_rate * 0.1
            )

            # Regular optimizer for W parameters
            W_params = list(self._model.W.parameters())
            if self.optimizer == 'Adam':
                self._w_optimizer = torch.optim.Adam(W_params, lr=self.learning_rate)
            elif self.optimizer == 'AdamW':
                self._w_optimizer = torch.optim.AdamW(W_params, lr=self.learning_rate)
            elif self.optimizer == 'NAdam':
                self._w_optimizer = torch.optim.NAdam(W_params, lr=self.learning_rate)
            elif self.optimizer == 'SGD':
                self._w_optimizer = torch.optim.SGD(W_params, lr=self.learning_rate, momentum=0.9)
            elif self.optimizer == 'RMSprop':
                self._w_optimizer = torch.optim.RMSprop(W_params, lr=self.learning_rate)
        else:
            # Standard mode: single optimizer for all parameters
            params = list(self._model.W.parameters()) + list(self._prior.parameters())
            if self.optimizer == 'Adam':
                self._optimizer = torch.optim.Adam(params, lr=self.learning_rate)
            elif self.optimizer == 'AdamW':
                self._optimizer = torch.optim.AdamW(params, lr=self.learning_rate)
            elif self.optimizer == 'NAdam':
                self._optimizer = torch.optim.NAdam(params, lr=self.learning_rate)
            elif self.optimizer == 'SGD':
                self._optimizer = torch.optim.SGD(params, lr=self.learning_rate, momentum=0.9)
            elif self.optimizer == 'RMSprop':
                self._optimizer = torch.optim.RMSprop(params, lr=self.learning_rate)
            self._w_optimizer = None

        # Training loop
        prev_elbo = float('-inf')
        elbo_history = [] if return_history else None

        # Update progress bar description based on training mode
        mode_desc = f"{self.mode} mode, {self.training_mode} training"
        pbar = tqdm(range(self.max_iter), disable=self.verbose, desc=f"PNMF fitting ({mode_desc})")

        for iteration in pbar:
            # Zero gradients for all optimizers
            self._optimizer.zero_grad()
            if self._w_optimizer is not None:
                self._w_optimizer.zero_grad()

            # Forward pass
            rate, qF, pF = self._model.forward(E=self.E)

            # Compute ELBO loss
            loss = self._elbo(rate, qF, pF, X_torch)

            # Backward pass
            loss.backward()

            # Step optimizers
            if self.training_mode == 'natural':
                # Natural gradient mode: step both optimizers
                self._optimizer.step()  # NGD for variational parameters
                self._w_optimizer.step()  # Adam for W parameters
            else:
                # Standard mode: single optimizer
                self._optimizer.step()

            # Project parameters if using projected gradient
            if self.loadings_mode == 'projected':
                self._model.project_parameters()

            # Check convergence
            elbo_value = -loss.item()  # Convert back to ELBO

            # Track ELBO history
            if return_history:
                elbo_history.append(elbo_value)

            if self.verbose:
                # Use print statements for verbose mode
                if iteration % 10 == 0:
                    print(f"Iteration {iteration}: ELBO = {elbo_value:.6f}")
            else:
                # Update tqdm progress bar with ELBO
                pbar.set_postfix({"ELBO": f"{elbo_value:.6f}"})

            if abs(elbo_value - prev_elbo) < self.tol:
                if self.verbose:
                    print(f"Converged at iteration {iteration}")
                else:
                    pbar.set_postfix({"ELBO": f"{elbo_value:.6f}", "status": "converged"})
                    pbar.close()
                break

            prev_elbo = elbo_value

        self.n_iter_ = iteration + 1
        self.elbo_ = prev_elbo

        # Store components (W transposed for sklearn compatibility: n_components x n_features)
        self.components_ = self._model.W.data.detach().cpu().numpy().T

        if return_history:
            return elbo_history, self
        return self

    def transform(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Transform X using the fitted model.

        Given fixed W (stored in components\\_), find the optimal exp(F) for new X.

        Note: This uses a simple NNLS (non-negative least squares) approach.
        For sklearn NMF compatibility, the returned value represents exp(F) in
        our model notation (called W in sklearn NMF's X ≈ W @ H.T notation).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data matrix.

        Returns
        -------
        transformed : ndarray of shape (n_samples, n_components)
            Transformed data (exp(F) in our model notation, corresponding to
            sklearn NMF's W coefficient matrix).
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
        transformed : ndarray of shape (n_samples, n_components)
            Transformed data (exp(F) in our model notation).
        """
        self.fit(X, **kwargs)
        return self.transform(X)

    def inverse_transform(self, transformed: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Transform data back to its original space.

        Reconstructs X from exp(F) using: X ≈ exp(F) @ W.T

        Parameters
        ----------
        transformed : array-like of shape (n_samples, n_components)
            Transformed data (exp(F) in our model notation).

        Returns
        -------
        X_reconstructed : ndarray of shape (n_samples, n_features)
            Reconstructed data in original space.
        """
        if self.components_ is None:
            raise ValueError("Model has not been fitted yet.")

        if isinstance(transformed, torch.Tensor):
            transformed = transformed.detach().cpu().numpy()

        transformed = np.asarray(transformed)
        # X = exp(F) @ W.T = transformed @ components_
        return transformed @ self.components_
