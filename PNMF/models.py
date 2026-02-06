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
from .elbo import compute_elbo
from .optimizers import NaturalGradientDescent
from .priors import GaussianPrior
from . import initialization


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
        # Pass mode (elbo_mode) to PositiveParameter for multiplicative updates
        self.W = PositiveParameter((D, L), mode=loadings_mode, elbo_mode=mode)

    def get_rate(self, prior_samples, idy=None):
        """
        Compute the Poisson rate from prior samples.

        Args:
            prior_samples: Samples from the prior of shape (E, L, N)
                          where E is number of samples, L is components, N is samples
            idy: Feature indices for batching (D dimension), None for full features

        Returns:
            Z: Rate matrix of shape (E, D, N) or (E, D_batch, batch_size)
        """
        F = torch.exp(prior_samples)  # shape (E, L, N)
        W = self.W.data  # shape (D, L)
        if idy is not None:
            W = W[idy]  # (D_batch, L)
        Z = torch.matmul(W, F)  # shape (E, D, N)
        return Z

    def forward(self, idx=None, idy=None, E=10):
        """
        Forward pass generating Poisson rate.

        Supports both full-batch and mini-batch training. When idx and idy are
        None, performs full-batch forward pass. Otherwise, computes on the
        specified batch indices.

        Args:
            idx: Sample indices (for N dimension), None for full samples
            idy: Feature indices (for D dimension), None for full features
            E: Number of Monte Carlo samples for variational inference

        Returns:
            rate: Poisson rate tensor of shape (E, D_batch, N_batch)
            qF: Variational posterior distribution (batched if idx provided)
            pF: Prior distribution (batched if idx provided)
        """
        # Get variational distributions (batched on samples if idx provided)
        if idx is not None:
            qF, pF = self.prior.forward_batched(idx)
        else:
            qF, pF = self.prior()

        # Sample F using reparameterization trick: shape (E, L, batch_size)
        F = qF.rsample((E,))

        # Compute rate using get_rate (F is the prior samples, not exp(F))
        rate = self.get_rate(F, idy=idy)  # shape (E, D, N)

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

    init : {'random', 'nndsvd', 'nndsvda', 'nndsvdar', 'k-means', None}, default='random'
        Initialization method for W and exp(F):
        - 'random': Non-negative random matrices, scaled with sqrt(X.mean() / n_components) (default).
        - 'nndsvd': Nonnegative Double SVD (better for sparseness).
        - 'nndsvda': NNDSVD with zeros filled with average of X (better for dense data).
        - 'nndsvdar': NNDSVD with zeros filled with small random values (faster dense).
        - 'k-means': K-means clustering based initialization.
        - None: Auto-select 'nndsvda' if n_components <= min(n_samples, n_features),
          otherwise 'random'.

    batch_size : int, default=None
        Size of mini-batches for samples (N dimension). If None, uses full batch.
        Enable mini-batch training for large datasets.

    y_batch_size : int, default=None
        Size of mini-batches for features (M/D dimension). If None, uses all features.
        Enable feature batching for very wide datasets.

    shuffle : bool, default=True
        Whether to shuffle sample indices between iterations (for mini-batch mode).

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

    Mini-batch training for large datasets:

    >>> X_large = np.random.rand(10000, 500)
    >>> model = PNMF(n_components=10, batch_size=1000, shuffle=True)
    >>> model.fit(X_large)
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
        device: str = 'auto',
        init: Optional[str] = 'random',
        batch_size: Optional[int] = None,
        y_batch_size: Optional[int] = None,
        shuffle: bool = True
    ):
        self.n_components = n_components
        self.loadings_mode = loadings_mode
        self.mode = mode
        self.training_mode = training_mode
        self.E = E
        if self.mode in ['lower-bound']:
            self.E = 1
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.random_state = random_state
        self.verbose = verbose
        self.device = device
        self.init = init
        self.batch_size = batch_size
        self.y_batch_size = y_batch_size
        self.shuffle = shuffle

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

        if self.loadings_mode not in ['softplus', 'exp', 'projected', 'multiplicative']:
            raise ValueError("loadings_mode must be 'softplus', 'exp', 'projected', or 'multiplicative'")

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

        valid_init_options = [None, 'random', 'nndsvd', 'nndsvda', 'nndsvdar', 'k-means']
        if self.init not in valid_init_options:
            raise ValueError(
                f"init must be one of {valid_init_options}, got '{self.init}'"
            )

        if self.batch_size is not None and self.batch_size < 1:
            raise ValueError("batch_size must be >= 1 or None")

        if self.y_batch_size is not None and self.y_batch_size < 1:
            raise ValueError("y_batch_size must be >= 1 or None")

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

    def _get_batch_indices(self, N, D, device):
        """
        Sample batch indices for mini-batch training.

        Following GPzoo pattern from training_utilities.py:181-182.

        Args:
            N: Total number of samples
            D: Total number of features
            device: Device to place indices on

        Returns:
            idx: Sample indices (x_batch_size,) or None if full batch
            idy: Feature indices (y_batch_size,) or None if full features
        """
        # Sample batch (N dimension)
        if self.batch_size is not None:
            x_batch_size = min(self.batch_size, N)
            idx = torch.multinomial(
                torch.ones(N, device=device),
                num_samples=x_batch_size,
                replacement=False
            )
        else:
            idx = None

        # Feature batch (D dimension)
        if self.y_batch_size is not None:
            y_batch_size = min(self.y_batch_size, D)
            idy = torch.multinomial(
                torch.ones(D, device=device),
                num_samples=y_batch_size,
                replacement=False
            )
        else:
            idy = None

        return idx, idy

    def _initialize_parameters(self, X_torch: torch.Tensor):
        """
        Initialize W and variational parameters using specified init method.

        Parameters
        ----------
        X_torch : torch.Tensor of shape (D, N)
            Input data tensor (transposed: features x samples).

        Notes
        -----
        This method initializes:
        - W (loadings matrix) via PositiveParameter
        - GaussianPrior mean (μ) based on log of initialized exp(F)
        - GaussianPrior scale (σ) to a small default value
        """
        # Convert back to numpy for initialization (transpose to sklearn format)
        X_np = X_torch.T.cpu().numpy()  # (N, D)

        # Get initializations
        W_init, exp_F_init = initialization.initialize_factors(
            X_np, self.n_components, self.init, self.random_state
        )

        # Initialize W (loadings) - shape (D, L)
        # W_init is (D, L), which matches our internal W shape
        self._model.W.data = torch.from_numpy(W_init.astype(np.float32)).to(self._get_device())

        # Initialize variational mean (μ)
        # F is in log-space, so μ = log(exp_F) = log(initial value)
        # Need to handle zeros: log(exp_F + eps)
        eps = 1e-8
        log_F_init = np.log(exp_F_init + eps)  # (N, L) -> need (L, N)
        mu_init = log_F_init.T  # (L, N)

        if self.training_mode == 'natural':
            # Natural parameterization: θ₁ = μ/s², θ₂ = -1/(2s²)
            # Initialize s² = 0.1 (small uncertainty), so:
            # θ₁ = μ / 0.1 = 10 * μ
            # θ₂ = -1/(2 * 0.1) = -5
            s2_init = 0.1
            self._prior.theta1.data = torch.from_numpy(
                (mu_init / s2_init).astype(np.float32)
            ).to(self._get_device())
            self._prior.theta2.data.fill_(-1.0 / (2.0 * s2_init))
        else:
            # Standard parameterization
            self._prior.mean.data = torch.from_numpy(
                mu_init.astype(np.float32)
            ).to(self._get_device())

            # Initialize scale to small value (we're fairly confident in initialization)
            # For softplus mode: raw parameter such that softplus(raw) ≈ 0.1
            # softplus(x) ≈ 0.1 when x ≈ -2.2
            if hasattr(self._prior.scale, '_raw'):
                self._prior.scale._raw.data.fill_(-2.2)

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

        # Apply custom initialization
        self._initialize_parameters(X_torch)

        # Setup optimizers based on training mode
        # For multiplicative mode, W is updated via multiplicative updates, not gradients
        use_multiplicative_w = (self.loadings_mode == 'multiplicative')

        if self.training_mode == 'natural':
            # Natural gradient mode: dual optimizers
            # NGD for variational parameters (natural params)
            # Use smaller learning rate for NGD (0.1x) for stability
            nat_params = self._prior.natural_parameters()
            self._optimizer = NaturalGradientDescent(
                nat_params, num_data=n_samples, lr=self.learning_rate * 0.1
            )

            if use_multiplicative_w:
                # W uses multiplicative updates, no optimizer needed
                self._w_optimizer = None
            else:
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
            # Standard mode
            if use_multiplicative_w:
                # Only optimize variational parameters, W uses multiplicative updates
                params = list(self._prior.parameters())
            else:
                # Optimize all parameters
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

        # Determine if we're using batched training
        use_batching = self.batch_size is not None or self.y_batch_size is not None
        D, N = X_torch.shape  # D = features, N = samples

        # Calculate batch sizes (for ELBO scaling)
        x_batch_size = self.batch_size if self.batch_size is not None else N
        y_batch_size = self.y_batch_size if self.y_batch_size is not None else D

        # Update progress bar description based on training mode
        mode_desc = f"{self.mode} mode, {self.training_mode} training"
        if use_batching:
            mode_desc += f", batch={x_batch_size}"
            if self.y_batch_size is not None:
                mode_desc += f", y_batch={y_batch_size}"
        pbar = tqdm(range(self.max_iter), disable=self.verbose, desc=f"PNMF fitting ({mode_desc})")

        for iteration in pbar:
            # Zero gradients for all optimizers
            self._optimizer.zero_grad()
            if self._w_optimizer is not None:
                self._w_optimizer.zero_grad()

            # Get batch indices (None for full-batch mode)
            idx, idy = self._get_batch_indices(N, D, self._get_device()) if use_batching else (None, None)

            # Get data batch
            if idx is not None and idy is not None:
                X_batch = X_torch[idy][:, idx]
            elif idx is not None:
                X_batch = X_torch[:, idx]
            elif idy is not None:
                X_batch = X_torch[idy]
            else:
                X_batch = X_torch

            # Forward pass (handles both batched and full modes)
            rate, qF, pF = self._model.forward(idx, idy, E=self.E)

            # Get W (optionally batched)
            W = self._model.W.data
            if idy is not None:
                W = W[idy]

            # Compute ELBO loss
            exp_log_likelihood, kl = compute_elbo(self.mode, rate, qF, pF, X_batch, W)

            # Scale ELBO for mini-batch
            if self.y_batch_size is not None:
                exp_log_likelihood = exp_log_likelihood * (D / min(self.y_batch_size, D))

            loss = kl - exp_log_likelihood
            if self.batch_size is not None:
                loss = loss * (N / min(self.batch_size, N))

            # Backward pass (for variational parameters)
            loss.backward()

            # Step optimizers for variational parameters
            if self.training_mode == 'natural':
                # Natural gradient mode: step NGD for variational parameters
                self._optimizer.step()
                if self._w_optimizer is not None:
                    self._w_optimizer.step()  # Adam for W parameters (if not multiplicative)
            else:
                # Standard mode: single optimizer
                self._optimizer.step()

            # Handle W updates based on loadings_mode
            if self.loadings_mode == 'multiplicative':
                # For lower-bound mode, no samples needed (fully analytic)
                # For expanded/simple, we need F samples
                if self.mode == 'lower-bound':
                    self._model.W.multiplicative_update(X_batch, qF, idy=idy)
                else:
                    # Get F samples for multiplicative update (use same E as forward pass)
                    # We need to re-sample since the forward pass samples are not stored
                    F_samples = qF.rsample((self.E,))  # (E, L, N)
                    self._model.W.multiplicative_update(X_batch, qF, F_samples, idy=idy)
            elif self.loadings_mode == 'projected':
                # Project parameters if using projected gradient
                self._model.project_parameters()

            # Check convergence (using scaled loss for batched mode)
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
