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
from .elbo import compute_elbo, compute_log_likelihood_terms
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

    def forward(self, idx=None, idy=None, E=10, X=None,
                coordinates=None, groups=None, spatial=False):
        """
        Forward pass: compute variational distributions and log-likelihood terms.

        Supports both full-batch and mini-batch training. When idx and idy are
        None, performs full-batch forward pass. Otherwise, computes on the
        specified batch indices.

        Args:
            idx: Sample indices (for N dimension), None for full samples
            idy: Feature indices (for D dimension), None for full features
            E: Number of Monte Carlo samples (ignored for lower-bound mode)
            X: Input data (D, N) or (D_batch, N_batch). Required for
               memory-efficient MC accumulation. If None and MC is needed,
               falls back to full-tensor mode.
            coordinates: Spatial coordinates (N_batch, 2) for GP prior. Required
                when spatial=True.
            groups: Group assignments (N_batch,) for MGGP prior. Required when
                spatial=True and using multi-group GP.
            spatial: Whether to use spatial GP forward pass.

        Returns:
            For non-spatial:
                terms: dict from compute_log_likelihood_terms()
                qF: Variational posterior distribution
                pF: Prior distribution
            For spatial:
                terms: dict from compute_log_likelihood_terms()
                qF: Variational posterior distribution from GP predictive
                qU: Inducing point variational distribution
                pU: Inducing point prior distribution (None for whitened)
        """
        if spatial:
            # Spatial GP forward pass
            # For LCGP, use forward_train() which returns marginal q(U_j) directly
            # from stored parameters - much faster O(M*R) vs full GP predictive O(M^2)
            # forward_train() returns (qF, qU, pU) where qU and pU are None
            if hasattr(self.prior, 'forward_train'):
                # LCGP: use forward_train() for efficient training
                qF, qU, pU = self.prior.forward_train(X=coordinates, groupsX=groups, idx=idx)
            elif groups is not None:
                qF, qU, pU = self.prior(X=coordinates, groupsX=groups)
            else:
                qF, qU, pU = self.prior(X=coordinates)
        else:
            # Standard GaussianPrior forward pass
            if idx is not None:
                qF, pF = self.prior.forward_batched(idx)
            else:
                qF, pF = self.prior()

        # Get W (optionally batched on features)
        W = self.W.data
        if idy is not None:
            W = W[idy]

        # Compute all log-likelihood terms
        # Multiplicative W needs the same MC samples for its update rule
        terms = compute_log_likelihood_terms(
            W=W, qF=qF, X=X, E=E, mode=self.mode,
            return_samples=(self.loadings_mode == 'multiplicative'),
        )

        if spatial:
            return terms, qF, qU, pU
        return terms, qF, pF

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

    scheduler : {'one_cycle', 'plateau', None}, default='one_cycle'
        Learning rate scheduler:
        - 'one_cycle': OneCycleLR with warmup then cosine annealing (default)
        - 'plateau': ReduceLROnPlateau, reduces LR when ELBO plateaus
        - None: No scheduler (constant learning rate)

    scheduler_patience : int, default=200
        Number of iterations with no improvement before reducing LR.
        Only used when scheduler='plateau'.

    scheduler_factor : float, default=0.8
        Factor by which to reduce LR (new_lr = old_lr * factor).
        Only used when scheduler='plateau'.

    scheduler_pct_start : float, default=0.3
        Fraction of total iterations spent in warmup phase (LR ramps up
        from learning_rate/div_factor to learning_rate). Only used when
        scheduler='one_cycle'.

    scheduler_div_factor : float, default=25.0
        Determines initial LR: initial_lr = learning_rate / div_factor.
        Only used when scheduler='one_cycle'.

    scheduler_final_div_factor : float, default=1e4
        Determines final LR: min_lr = initial_lr / final_div_factor.
        Only used when scheduler='one_cycle'.

    min_lr : float, default=1e-5
        Minimum learning rate. Only used when scheduler='plateau'.

    batch_size : int, default=None
        Size of mini-batches for samples (N dimension). If None, uses full batch.
        Enable mini-batch training for large datasets.

    y_batch_size : int, default=None
        Size of mini-batches for features (M/D dimension). If None, uses all features.
        Enable feature batching for very wide datasets.

    shuffle : bool, default=True
        Whether to shuffle sample indices between iterations (for mini-batch mode).

    Spatial GP Parameters
    --------------------
    spatial : bool, default=False
        Enable spatial GP prior instead of independent Gaussian prior.
        Requires coordinates to be provided in fit().

    local : bool, default=False
        Use locally conditioned GP (LCGP) instead of SVGP when spatial=True.
        LCGP uses all points as inducing with low-rank+diagonal covariance.
        Only used when spatial=True.

    kernel : str, default='Matern32'
        Kernel function for spatial GP. Currently only 'Matern32' is supported.

    multigroup : bool, default=False
        Use multi-group GP (MGGP) with group-aware spatial smoothing.
        When False, uses standard GP with batched_Matern32 kernel (no groups needed).

    num_inducing : int, default=3000
        Number of inducing points for SVGP approximation.

    lengthscale : float, default=1.0
        Kernel lengthscale for spatial correlation.

    sigma : float, default=1.0
        Kernel output scale (amplitude).

    group_diff_param : float, default=10.0
        Group difference parameter for MGGP. Higher values = stronger group separation.

    jitter : float, default=1e-5
        Jitter term for numerical stability in Cholesky decomposition.

    train_lengthscale : bool, default=False
        Whether to train the kernel lengthscale (currently not supported).

    cholesky_mode : str, default='exp'
        Cholesky diagonal constraint mode ('exp', 'softplus').

    diagonal_only : bool, default=False
        Use diagonal-only variational covariance for inducing points.

    inducing_allocation : str, default='proportional'
        How to distribute inducing points across groups (SVGP only):
        - 'proportional': Allocate points proportionally to group sizes (default).
        - 'equal': Allocate equal points to each group.
        - 'derived': Run K-means on all data for optimal spatial coverage,
          then use KNN (k=5, distance-weighted) to classify centroids to groups.
          Falls back to proportional for groups with no assigned points.

    LCGP-Specific Parameters
    ------------------------
    K : int, default=50
        Number of nearest neighbors for LCGP local conditioning.
        Only used when local=True.

    precompute_knn : bool, default=True
        Whether to precompute KNN indices at initialization for LCGP.
        Only used when local=True.

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
        scheduler: Optional[str] = 'one_cycle',
        scheduler_patience: int = 200,
        scheduler_factor: float = 0.8,
        scheduler_pct_start: float = 0.3,
        scheduler_div_factor: float = 25.0,
        scheduler_final_div_factor: float = 1e4,
        min_lr: float = 1e-5,
        batch_size: Optional[int] = None,
        y_batch_size: Optional[int] = None,
        shuffle: bool = True,
        # Spatial GP parameters
        spatial: bool = False,
        local: bool = False,
        kernel: str = 'Matern32',
        multigroup: bool = False,
        num_inducing: int = 3000,
        lengthscale: float = 1.0,
        sigma: float = 1.0,
        group_diff_param: float = 10.0,
        jitter: float = 1e-5,
        train_lengthscale: bool = False,
        cholesky_mode: str = 'exp',
        diagonal_only: bool = False,
        inducing_allocation: str = 'proportional',
        # LCGP-specific parameters
        K: int = 50,
        precompute_knn: bool = True,
        # ELBO scaling flags (default True = correct behaviour; False = old/broken for video demos)
        scale_ll_D: bool = True,
        scale_kl_NM: bool = True,
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
        self.scheduler = scheduler
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor
        self.scheduler_pct_start = scheduler_pct_start
        self.scheduler_div_factor = scheduler_div_factor
        self.scheduler_final_div_factor = scheduler_final_div_factor
        self.min_lr = min_lr
        self.batch_size = batch_size
        self.y_batch_size = y_batch_size
        self.shuffle = shuffle

        # Spatial GP parameters
        self.spatial = spatial
        self.local = local
        self.kernel = kernel
        self.multigroup = multigroup
        self.num_inducing = num_inducing
        self.lengthscale = lengthscale
        self.sigma = sigma
        self.group_diff_param = group_diff_param
        self.jitter = jitter
        self.train_lengthscale = train_lengthscale
        self.cholesky_mode = cholesky_mode
        self.diagonal_only = diagonal_only
        self.inducing_allocation = inducing_allocation

        # LCGP-specific parameters
        self.K = K
        self.precompute_knn = precompute_knn

        # ELBO scaling flags
        self.scale_ll_D = scale_ll_D
        self.scale_kl_NM = scale_kl_NM

        # Derive prior type from spatial and local flags
        if self.spatial:
            self.prior_type = 'LCGP' if self.local else 'SVGP'
        else:
            self.prior_type = 'GaussianPrior'

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
        self._scheduler = None
        self._w_scheduler = None
        self._coordinates = None
        self._groups = None
        self._knn_idx = None  # For LCGP: stores KNN indices

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

        if self.scheduler is not None and self.scheduler not in ['one_cycle', 'plateau']:
            raise ValueError("scheduler must be 'one_cycle', 'plateau', or None")

        if self.batch_size is not None and self.batch_size < 1:
            raise ValueError("batch_size must be >= 1 or None")

        if self.y_batch_size is not None and self.y_batch_size < 1:
            raise ValueError("y_batch_size must be >= 1 or None")

        # Spatial parameter validation
        if self.local and not self.spatial:
            raise ValueError("local=True requires spatial=True")

        if self.spatial:
            if self.kernel not in ['Matern32']:
                raise ValueError("kernel must be 'Matern32'")
            if self.training_mode == 'natural':
                raise ValueError("Natural gradient training not supported with spatial priors")
            if self.inducing_allocation not in ['proportional', 'equal', 'derived']:
                raise ValueError("inducing_allocation must be 'proportional', 'equal', or 'derived'")

            # SVGP-specific validation
            if not self.local:
                if self.num_inducing < 1:
                    raise ValueError("num_inducing must be >= 1")

            # LCGP-specific validation
            if self.local:
                if self.K < 1:
                    raise ValueError("K must be >= 1 for LCGP")
                # Warn if num_inducing is set (LCGP ignores it)
                if self.num_inducing != 3000:  # Default value
                    import warnings
                    warnings.warn(
                        f"num_inducing is ignored when local=True (LCGP uses all points as inducing). "
                        f"Current setting: num_inducing={self.num_inducing}"
                    )

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

    def _create_optimizer(self, params):
        """Create optimizer based on self.optimizer setting."""
        if self.optimizer == 'Adam':
            return torch.optim.Adam(params, lr=self.learning_rate)
        elif self.optimizer == 'AdamW':
            return torch.optim.AdamW(params, lr=self.learning_rate)
        elif self.optimizer == 'NAdam':
            return torch.optim.NAdam(params, lr=self.learning_rate)
        elif self.optimizer == 'SGD':
            return torch.optim.SGD(params, lr=self.learning_rate, momentum=0.9)
        elif self.optimizer == 'RMSprop':
            return torch.optim.RMSprop(params, lr=self.learning_rate)

    def _create_spatial_prior(self, Y, coordinates, groups):
        """
        Create spatial GP prior (SVGP or LCGP) for spatial mode.

        Args:
            Y: Data tensor of shape (D, N)
            coordinates: Spatial coordinates tensor of shape (N, 2)
            groups: Group assignments tensor of shape (N,), or None

        Returns:
            gp: GP model (SVGP, MGGP_SVGP, LCGP, or MGGP_LCGP) ready for use as prior
        """
        try:
            from gpzoo.kernels import batched_MGGP_Matern32, batched_Matern32
            from gpzoo.gp import MGGP_SVGP, SVGP, LCGP, MGGP_LCGP
            from gpzoo.modules import CholeskyParameter
            from gpzoo.model_utilities import mggp_kmeans_inducing_points, kmeans_inducing_points
            from gpzoo.utilities import init_Lu as gpzoo_init_Lu, init_Lu_nsf  # noqa: F401
        except ImportError:
            raise ImportError(
                "GPzoo is required for spatial mode. "
                "Install with: pip install -e path/to/GPzoo"
            )

        D, N = Y.shape
        L = self.n_components
        n_groups = int(groups.max().item() + 1) if groups is not None else 1

        # 1. Create kernel (same for SVGP and LCGP)
        if self.multigroup and groups is not None:
            kernel = batched_MGGP_Matern32(
                sigma=self.sigma,
                lengthscale=self.lengthscale,
                group_diff_param=self.group_diff_param,
                n_groups=n_groups,
            )
        else:
            kernel = batched_Matern32(
                sigma=self.sigma,
                lengthscale=self.lengthscale,
            )

        # 2. Branch on prior type: SVGP vs LCGP
        if self.prior_type == 'SVGP':
            # === SVGP: Use subset of inducing points ===
            M = min(self.num_inducing, N)
            if self.multigroup and groups is not None:
                Z, groupsZ = mggp_kmeans_inducing_points(
                    coordinates, groups, M,
                    seed=self.random_state or 123,
                    allocation=self.inducing_allocation,
                )
                # derived allocation may return fewer points than requested
                M = Z.shape[0]
            else:
                Z = kmeans_inducing_points(
                    coordinates, M,
                    seed=self.random_state or 123,
                )
                groupsZ = None

            # 3. Create GP
            if self.multigroup and groups is not None:
                gp = MGGP_SVGP(
                    kernel, dim=coordinates.shape[1], M=M, n_groups=n_groups,
                    jitter=self.jitter, cholesky_mode=self.cholesky_mode,
                    diagonal_only=self.diagonal_only,
                )
                gp.Z = nn.Parameter(Z, requires_grad=False)
                gp.groupsZ = nn.Parameter(groupsZ, requires_grad=False)
            else:
                gp = SVGP(
                    kernel, dim=coordinates.shape[1], M=M,
                    jitter=self.jitter, cholesky_mode=self.cholesky_mode,
                    diagonal_only=self.diagonal_only,
                )
                gp.Z = nn.Parameter(Z, requires_grad=False)

            # 4. Batch mu and Lu for L latent factors
            #    SVGP creates mu (M,) and Lu (M, M) for a single output.
            #    We need (L, M) and (L, M, M) for L latent factors.
            del gp.Lu
            gp.Lu = CholeskyParameter(
                (L, M), mode=self.cholesky_mode, diagonal_only=self.diagonal_only
            )
            Lu_init = torch.randn(L, M, M) * 1e-2
            Lu_init = torch.tril(Lu_init)
            Lu_init[:, range(M), range(M)] = torch.rand(L, M)
            gp.Lu.data = Lu_init
            gp.mu = nn.Parameter(torch.randn(L, M) * 1.0)

        elif self.local:
            # === LCGP: Use ALL points as inducing points ===
            M = N  # LCGP uses all points as inducing points
            Z = coordinates.clone()  # All points are inducing points
            K = self.K
            if self.multigroup and groups is not None:
                groupsZ = groups.clone()
            else:
                groupsZ = None

            # 3. Create GP
            if self.multigroup and groups is not None:
                gp = MGGP_LCGP(
                    kernel, dim=coordinates.shape[1], M=M, n_groups=n_groups,
                    jitter=self.jitter, K=K,
                )
                gp.Z = nn.Parameter(Z, requires_grad=False)
                gp.groupsZ = nn.Parameter(groupsZ, requires_grad=False)
            else:
                gp = LCGP(
                    kernel, dim=coordinates.shape[1], M=M,
                    jitter=self.jitter, K=K,
                )
                gp.Z = nn.Parameter(Z, requires_grad=False)

            # 4. Initialize Lu as raw nn.Parameter (same approach as VNNGP)
            #    Lu shape: (L, M, K) for L latent factors
            del gp.Lu
            gp.Lu = nn.Parameter(torch.randn(L, M, K) * 1e-1)

            # Initialize mu: random N(0, 1)
            gp.mu = nn.Parameter(torch.randn(L, M) * 1.0)

            # 5. Precompute KNN indices (always needed for training)
            knn_idx = gp.calculate_knn(coordinates)[:, :-1]  # Exclude self
            gp.knn_idx = knn_idx
            gp.knn_idz = knn_idx  # For LCGP, knn_idx == knn_idz since Z = X

        else:
            raise ValueError(f"Unknown prior type: {self.prior_type}")

        # 6. Freeze kernel hyperparameters
        #    By default we freeze all kernel params (lengthscale, sigma, group_diff_param).
        #    Only lengthscale can be optionally unfrozen via train_lengthscale=True.
        if not self.train_lengthscale:
            kernel.lengthscale.requires_grad = False
        kernel.sigma.requires_grad = False
        if hasattr(kernel, 'group_diff_param'):
            kernel.group_diff_param.requires_grad = False

        return gp

    def _initialize_W(self, X_torch: torch.Tensor):
        """
        Initialize W (loadings) using the same strategy for spatial and non-spatial.

        Parameters
        ----------
        X_torch : torch.Tensor of shape (D, N)
            Input data tensor (transposed: features x samples).

        Returns
        -------
        W_init : ndarray of shape (D, L)
            Initialized loadings matrix.
        exp_F_init : ndarray of shape (N, L)
            Initialized expected latent factors (exp-space).
        """
        # Convert back to numpy for initialization (transpose to sklearn format)
        X_np = X_torch.T.cpu().numpy()  # (N, D)

        # Get initializations
        W_init, exp_F_init = initialization.initialize_factors(
            X_np, self.n_components, self.init, self.random_state
        )

        # Initialize W (loadings) - shape (D, L)
        device = self._get_device()
        self._model.W.data = torch.from_numpy(W_init.astype(np.float32)).to(device)

        return W_init, exp_F_init

    def _initialize_mu_nonspatial(self, exp_F_init: np.ndarray):
        """
        Initialize variational mean (μ) for non-spatial models.

        Parameters
        ----------
        exp_F_init : ndarray of shape (N, L)
            Initialized expected latent factors from _initialize_W().
        """
        device = self._get_device()
        eps = 1e-8
        log_F_init = np.log(exp_F_init + eps)  # (N, L)
        mu_init = log_F_init.T  # (L, N)

        if self.training_mode == 'natural':
            # Natural parameterization: θ₁ = μ/s², θ₂ = -1/(2s²)
            # Initialize s² = 0.1 (small uncertainty), so:
            # θ₁ = μ / 0.1 = 10 * μ
            # θ₂ = -1/(2 * 0.1) = -5
            s2_init = 0.1
            self._prior.theta1.data = torch.from_numpy(
                (mu_init / s2_init).astype(np.float32)
            ).to(device)
            self._prior.theta2.data.fill_(-1.0 / (2.0 * s2_init))
        else:
            # Standard parameterization
            self._prior.mean.data = torch.from_numpy(
                mu_init.astype(np.float32)
            ).to(device)

            # Initialize scale to small value (we're fairly confident in initialization)
            # For softplus mode: raw parameter such that softplus(raw) ≈ 0.1
            # softplus(x) ≈ 0.1 when x ≈ -2.2
            if hasattr(self._prior.scale, '_raw'):
                self._prior.scale._raw.data.fill_(-2.2)

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Optional[Union[np.ndarray, torch.Tensor]] = None,
        coordinates: Optional[Union[np.ndarray, torch.Tensor]] = None,
        groups: Optional[Union[np.ndarray, torch.Tensor]] = None,
        return_history: bool = False,
        callback=None,
        callback_interval: int = 100,
    ) -> Union['PNMF', tuple[list[float], 'PNMF']]:
        """
        Fit the PNMF model to data X using variational inference.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data matrix (non-negative).

        y : Ignored
            Not used, present for scikit-learn compatibility.

        coordinates : array-like of shape (n_samples, 2), optional
            Spatial coordinates for each sample. Required when spatial=True.

        groups : array-like of shape (n_samples,), optional
            Integer group assignments for each sample. Required when
            spatial=True and multigroup=True.

        return_history : bool, default=False
            If True, returns a tuple (history, self) where history is a list
            of ELBO values during training.

        Returns
        -------
        self : object
            Returns the instance itself (or (history, self) if return_history=True).
        """
        self._validate_params()

        # Validate spatial inputs
        if self.spatial:
            if coordinates is None:
                raise ValueError(
                    "coordinates is required when spatial=True. "
                    "Pass coordinates of shape (n_samples, 2) to fit()."
                )
            if self.multigroup and groups is None:
                raise ValueError(
                    "groups is required when spatial=True and multigroup=True. "
                    "Pass groups of shape (n_samples,) to fit()."
                )

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

        device = self._get_device()

        # Convert to torch tensor and transpose for model (D, N)
        X_torch = torch.from_numpy(X_np.T.astype(np.float32)).to(device)

        # Convert spatial inputs to tensors
        if coordinates is not None:
            if isinstance(coordinates, np.ndarray):
                coords_torch = torch.from_numpy(coordinates.astype(np.float32)).to(device)
            else:
                coords_torch = coordinates.to(device).float()
            self._coordinates = coords_torch
        else:
            coords_torch = None

        if groups is not None:
            if isinstance(groups, np.ndarray):
                groups_torch = torch.from_numpy(groups.astype(np.int64)).to(device)
            else:
                groups_torch = groups.to(device).long()
            self._groups = groups_torch
        else:
            groups_torch = None

        # Initialize prior
        if self.spatial:
            self._prior = self._create_spatial_prior(
                X_torch, coords_torch, groups_torch
            ).to(device)
            # Store KNN indices for LCGP (used during mini-batch training)
            if self.local:
                self._knn_idx = self._prior.knn_idz.clone()
        else:
            use_natural_gradients = (self.training_mode == 'natural')
            self._prior = GaussianPrior(
                y=X_torch,
                L=self.n_components,
                use_natural_gradients=use_natural_gradients
            ).to(device)

        # Initialize model
        self._model = PoissonFactorization(
            prior=self._prior,
            y=X_torch,
            L=self.n_components,
            loadings_mode=self.loadings_mode,
            mode=self.mode
        ).to(device)

        # Initialize W (data-aware, shared by both spatial and non-spatial)
        W_init, exp_F_init = self._initialize_W(X_torch)

        # Initialize variational parameters (different for spatial vs non-spatial)
        if not self.spatial:
            self._initialize_mu_nonspatial(exp_F_init)

        # Setup optimizers based on training mode
        # For multiplicative mode, W is updated via multiplicative updates, not gradients
        use_multiplicative_w = (self.loadings_mode == 'multiplicative')

        if self.spatial:
            # Spatial mode: optimize GP parameters + W parameters together
            gp_params = list(self._prior.parameters())
            if use_multiplicative_w:
                params = gp_params
            else:
                params = list(self._model.W.parameters()) + gp_params

            self._optimizer = self._create_optimizer(params)
            self._w_optimizer = None
        elif self.training_mode == 'natural':
            # Natural gradient mode: dual optimizers
            nat_params = self._prior.natural_parameters()
            self._optimizer = NaturalGradientDescent(
                nat_params, num_data=n_samples, lr=self.learning_rate * 0.1
            )

            if use_multiplicative_w:
                self._w_optimizer = None
            else:
                W_params = list(self._model.W.parameters())
                self._w_optimizer = self._create_optimizer(W_params)
        else:
            # Standard mode
            if use_multiplicative_w:
                params = list(self._prior.parameters())
            else:
                params = list(self._model.W.parameters()) + list(self._prior.parameters())

            self._optimizer = self._create_optimizer(params)
            self._w_optimizer = None

        # Create learning rate schedulers
        if self.scheduler == 'one_cycle':
            # NaturalGradientDescent doesn't support momentum, so disable cycle_momentum
            needs_no_momentum = isinstance(self._optimizer, NaturalGradientDescent)
            self._scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self._optimizer,
                max_lr=self.learning_rate,
                total_steps=self.max_iter,
                pct_start=self.scheduler_pct_start,
                div_factor=self.scheduler_div_factor,
                final_div_factor=self.scheduler_final_div_factor,
                cycle_momentum=not needs_no_momentum,
            )
            if self._w_optimizer is not None:
                self._w_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    self._w_optimizer,
                    max_lr=self.learning_rate,
                    total_steps=self.max_iter,
                    pct_start=self.scheduler_pct_start,
                    div_factor=self.scheduler_div_factor,
                    final_div_factor=self.scheduler_final_div_factor,
                )
        elif self.scheduler == 'plateau':
            self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self._optimizer,
                mode='max',  # maximize ELBO
                factor=self.scheduler_factor,
                patience=self.scheduler_patience,
                min_lr=self.min_lr,
            )
            if self._w_optimizer is not None:
                self._w_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self._w_optimizer,
                    mode='max',
                    factor=self.scheduler_factor,
                    patience=self.scheduler_patience,
                    min_lr=self.min_lr,
                )
        else:
            self._scheduler = None
            self._w_scheduler = None

        # Training loop
        prev_elbo = float('-inf')
        elbo_history = [] if return_history else None
        # Exponential moving average of ELBO for scheduler (smooths noise)
        ema_elbo = None
        ema_alpha = 0.05  # smoothing factor
        prev_lr = self.learning_rate  # track LR for logging reductions

        # Determine if we're using batched training
        use_batching = self.batch_size is not None or self.y_batch_size is not None
        D, N = X_torch.shape  # D = features, N = samples

        # Calculate batch sizes (for ELBO scaling)
        x_batch_size = self.batch_size if self.batch_size is not None else N
        y_batch_size = self.y_batch_size if self.y_batch_size is not None else D

        # Update progress bar description based on training mode
        mode_desc = f"{self.mode} mode"
        if self.spatial:
            mode_desc += ", spatial"
        else:
            mode_desc += f", {self.training_mode} training"
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
            idx, idy = self._get_batch_indices(N, D, device) if use_batching else (None, None)

            # Get data batch
            if idx is not None and idy is not None:
                X_batch = X_torch[idy][:, idx]
            elif idx is not None:
                X_batch = X_torch[:, idx]
            elif idy is not None:
                X_batch = X_torch[idy]
            else:
                X_batch = X_torch

            if self.spatial:
                # Spatial forward pass
                coords_batch = coords_torch[idx] if idx is not None else coords_torch
                groups_batch = groups_torch[idx] if (idx is not None and groups_torch is not None) else groups_torch

                # For LCGP: Set per-batch KNN indices for forward_train()
                # knn_idx = sliced for forward pass; knn_idz = FULL for kl_divergence_full
                if self.local:
                    knn_idx = self._knn_idx[idx.cpu()] if idx is not None else self._knn_idx
                    self._prior.knn_idx = knn_idx
                    # knn_idz stays as full indices (set during _create_spatial_prior)

                # For LCGP: pass idx so forward_train() indexes into mu/Lu
                # For SVGP: idx=None since GP gets coordinate batch directly
                spatial_idx = idx if self.local else None
                terms, qF, qU, pU = self._model.forward(
                    idx=spatial_idx, idy=idy, E=self.E, X=X_batch,
                    coordinates=coords_batch, groups=groups_batch, spatial=True
                )

                # Compute expected log-likelihood
                from .elbo import expected_log_likelihood as exp_ll_fn
                exp_ll = exp_ll_fn(self.mode, terms, X_batch)

                # Compute KL divergence (different for SVGP vs LCGP)
                if self.local:
                    # LCGP: Locally conditioned KL over batch points
                    # Scales with x_batch like non-spatial KL
                    kl = self._prior.kl_divergence_full(qZ=None, idx=idx).sum()
                else:
                    # SVGP: Whitened KL on inducing points (global)
                    # GP returns per-factor KL (shape (L,)), sum over factors
                    # Scale by N/M to match non-spatial KL-to-likelihood ratio
                    M = self._prior.Z.shape[0]
                    if self.scale_kl_NM:
                        kl = self._prior.kl_divergence(qU, pU).sum() * (N / M)
                    else:
                        kl = self._prior.kl_divergence(qU, pU).sum()

                # Scale expected log-likelihood for feature mini-batch
                if self.y_batch_size is not None and self.scale_ll_D:
                    exp_ll = exp_ll * (D / min(self.y_batch_size, D))

                # Scale expected log-likelihood for sample mini-batch
                if self.batch_size is not None:
                    exp_ll = exp_ll * (N / min(self.batch_size, N))

                # LCGP: KL scales with x_batch (like non-spatial), needs N-scaling
                if self.local and self.batch_size is not None:
                    kl = kl * (N / min(self.batch_size, N))
            else:
                # Standard (non-spatial) forward pass
                terms, qF, pF = self._model.forward(idx, idy, E=self.E, X=X_batch)

                # Compute expected log-likelihood and KL separately
                exp_ll, kl = compute_elbo(self.mode, terms, qF, pF, X_batch)

                # Scale expected log-likelihood for feature mini-batch
                if self.y_batch_size is not None and self.scale_ll_D:
                    exp_ll = exp_ll * (D / min(self.y_batch_size, D))

                # Scale expected log-likelihood for sample mini-batch
                if self.batch_size is not None:
                    exp_ll = exp_ll * (N / min(self.batch_size, N))

                # Scale KL for sample mini-batch
                # (KL is over batched q(F) so it needs N-scaling)
                if self.batch_size is not None:
                    kl = kl * (N / min(self.batch_size, N))

            # Loss = -ELBO = KL - E[log p(X|F)]
            loss = kl - exp_ll

            # Backward pass (for variational parameters)
            loss.backward()

            # Step optimizers for variational parameters
            if self.training_mode == 'natural' and not self.spatial:
                # Natural gradient mode: step NGD for variational parameters
                self._optimizer.step()
                if self._w_optimizer is not None:
                    self._w_optimizer.step()
            else:
                # Standard mode or spatial mode: single optimizer
                self._optimizer.step()

            # Handle W updates based on loadings_mode
            if self.loadings_mode == 'multiplicative':
                self._model.W.multiplicative_update(X_batch, terms, idy=idy)
            elif self.loadings_mode == 'projected':
                # Project parameters if using projected gradient
                self._model.project_parameters()

            # Check convergence (using scaled loss for batched mode)
            elbo_value = -loss.item()  # Convert back to ELBO

            # Track ELBO history
            if return_history:
                elbo_history.append(elbo_value)

            # Fire optional callback every callback_interval iterations
            if callback is not None and iteration % callback_interval == 0:
                callback(self, iteration, elbo_value)

            # Update EMA of ELBO and step scheduler
            if ema_elbo is None:
                ema_elbo = elbo_value
            else:
                ema_elbo = ema_alpha * elbo_value + (1 - ema_alpha) * ema_elbo

            if self.scheduler == 'plateau':
                # ReduceLROnPlateau needs the metric
                if self._scheduler is not None:
                    self._scheduler.step(ema_elbo)
                if self._w_scheduler is not None:
                    self._w_scheduler.step(ema_elbo)
            elif self.scheduler == 'one_cycle':
                # OneCycleLR steps every iteration (no metric)
                if self._scheduler is not None:
                    self._scheduler.step()
                if self._w_scheduler is not None:
                    self._w_scheduler.step()

            # Get current learning rate for display
            current_lr = self._optimizer.param_groups[0]['lr']

            # Log LR reduction events (only for plateau scheduler)
            if self.scheduler == 'plateau' and current_lr < prev_lr:
                if self.verbose:
                    print(f"Iteration {iteration}: Reducing lr: {prev_lr:.2e} -> {current_lr:.2e}")
                else:
                    pbar.write(f"Iteration {iteration}: Reducing lr: {prev_lr:.2e} -> {current_lr:.2e}")
            prev_lr = current_lr

            if self.verbose:
                # Use print statements for verbose mode
                if iteration % 10 == 0:
                    print(f"Iteration {iteration}: ELBO = {elbo_value:.6f}")
            else:
                # Update tqdm progress bar with ELBO and LR
                pbar.set_postfix({"ELBO": f"{elbo_value:.6f}", "lr": f"{current_lr:.1e}"})

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

    def transform(
        self,
        X: Union[np.ndarray, torch.Tensor],
        coordinates: Optional[Union[np.ndarray, torch.Tensor]] = None,
        groups: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> np.ndarray:
        """
        Transform X using the fitted model.

        For non-spatial models, uses NNLS multiplicative updates to find exp(F).
        For spatial models, uses GP predictive equations at new coordinates.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data matrix.

        coordinates : array-like of shape (n_samples, 2), optional
            Spatial coordinates for new samples. Required when spatial=True.

        groups : array-like of shape (n_samples,), optional
            Group assignments for new samples. Required when spatial=True
            and multigroup=True.

        Returns
        -------
        transformed : ndarray of shape (n_samples, n_components)
            Transformed data (exp(F) in our model notation).
        """
        if self.components_ is None:
            raise ValueError("Model has not been fitted yet.")

        if self.spatial:
            if coordinates is None:
                raise ValueError("coordinates is required for transform() when spatial=True")
            if self.multigroup and groups is None:
                raise ValueError("groups is required for transform() when spatial=True and multigroup=True")

            device = self._get_device()

            # Convert to tensors
            if isinstance(coordinates, np.ndarray):
                coords_t = torch.from_numpy(coordinates.astype(np.float32)).to(device)
            else:
                coords_t = coordinates.to(device).float()

            if groups is not None:
                if isinstance(groups, np.ndarray):
                    groups_t = torch.from_numpy(groups.astype(np.int64)).to(device)
                else:
                    groups_t = groups.to(device).long()
            else:
                groups_t = None

            with torch.no_grad():
                # For LCGP: set KNN indices before calling forward()
                if self.local:
                    knn_idx = self._prior.calculate_knn(coords_t)[:, :-1]
                    self._prior.knn_idx = knn_idx

                if groups_t is not None:
                    qF, _, _ = self._prior(X=coords_t, groupsX=groups_t)
                else:
                    qF, _, _ = self._prior(X=coords_t)
                # Return exp(mean) as the point estimate
                return torch.exp(qF.mean).T.cpu().numpy()  # (N_new, L)

        # Non-spatial: NNLS multiplicative updates
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

    def fit_transform(
        self,
        X: Union[np.ndarray, torch.Tensor],
        coordinates: Optional[Union[np.ndarray, torch.Tensor]] = None,
        groups: Optional[Union[np.ndarray, torch.Tensor]] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Fit the model and transform X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data matrix.

        coordinates : array-like of shape (n_samples, 2), optional
            Spatial coordinates. Required when spatial=True.

        groups : array-like of shape (n_samples,), optional
            Group assignments. Required when spatial=True and multigroup=True.

        **kwargs : Additional arguments to pass to fit()

        Returns
        -------
        transformed : ndarray of shape (n_samples, n_components)
            Transformed data (exp(F) in our model notation).
        """
        self.fit(X, coordinates=coordinates, groups=groups, **kwargs)
        return self.transform(X, coordinates=coordinates, groups=groups)

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
