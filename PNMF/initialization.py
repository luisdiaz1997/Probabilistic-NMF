"""
Initialization strategies for PNMF.

Provides sklearn-compatible initialization methods for NMF:
- 'random': Random initialization with scaling
- 'nndsvd': Nonnegative Double SVD (sparse)
- 'nndsvda': NNDSVD with average fill (dense)
- 'nndsvdar': NNDSVD with random fill (dense, faster)
- 'k-means': K-means clustering based

Reference:
    C. Boutsidis, E. Gallopoulos (2008)
    "SVD based initialization: A head start for nonnegative matrix factorization"
    Pattern Recognition 41(4):1350-1362
    https://doi.org/10.1016/j.patcog.2007.09.010
"""

import numpy as np
import torch
from typing import Tuple, Optional, Union
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from scipy import sparse


def _norm(x: np.ndarray) -> float:
    """Dot product-based Euclidean norm."""
    return np.sqrt(np.dot(x.ravel(), x.ravel()))


def _scale_log_factors(
    exp_F: np.ndarray,
    target_std: float = 1.0,
    eps: float = 1e-8
) -> np.ndarray:
    """
    Scale exp_F so that log(exp_F) has approximately N(0, target_std^2) distribution.

    This ensures that the variational mean μ = log(exp_F) is on a similar scale
    across different datasets and initialization methods. A standard normal
    distribution has 97.5% of values between -1.96 and 1.96.

    Parameters
    ----------
    exp_F : ndarray of shape (n_samples, n_components)
        Expected latent factors in exp-space (before log transform).
    target_std : float, default=1.0
        Target standard deviation for log(exp_F). Setting to 1.0 means
        μ ~ N(0, 1) approximately.
    eps : float, default=1e-8
        Small value to add before log transform to handle zeros.

    Returns
    -------
    exp_F_scaled : ndarray of same shape as exp_F
        Scaled expected factors such that log(exp_F_scaled) has std ~ target_std.

    Notes
    -----
    The scaling is done by working in log-space directly:
    1. Computing log_F = log(exp_F + eps)
    2. Computing current_mean and current_std of log_F
    3. Standardizing: z = (log_F - current_mean) / current_std
    4. Scaling to target: log_F_scaled = z * target_std
    5. Exponentiating back: exp_F_scaled = exp(log_F_scaled)

    This ensures the log-space values have exactly the target standard deviation
    and approximately zero mean.
    """
    # Compute log-space values
    log_F = np.log(exp_F + eps)

    # Compute current statistics
    current_mean = log_F.mean()
    current_std = log_F.std()

    # If std is very small or zero, don't scale (already very concentrated)
    if current_std < 1e-6:
        return exp_F

    # Standardize and rescale to target standard deviation
    log_F_scaled = (log_F - current_mean) / current_std * target_std

    # Exponentiate back to exp-space
    return np.exp(log_F_scaled)


def _check_init_array(A: np.ndarray, shape: Tuple[int, int], name: str) -> np.ndarray:
    """Validate custom initialization array."""
    A = np.asarray(A)
    if A.shape != shape:
        raise ValueError(f"{name} has wrong shape: {A.shape}, expected {shape}")
    if np.min(A) < 0:
        raise ValueError(f"{name} must be non-negative")
    if np.max(A) == 0:
        raise ValueError(f"{name} cannot be all zeros")
    return A


def _initialize_random(
    X: np.ndarray,
    n_components: int,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Random initialization following sklearn's approach.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data matrix.
    n_components : int
        Number of components.
    random_state : int or None
        Random seed.

    Returns
    -------
    W : ndarray of shape (n_features, n_components)
        Loadings matrix initialization.
    exp_F : ndarray of shape (n_samples, n_components)
        Expected latent factors initialization (exp-space).
    """
    rng = np.random.RandomState(random_state)
    n_samples, n_features = X.shape

    # Scale: sqrt(X.mean() / n_components) as in sklearn
    avg = np.sqrt(X.mean() / n_components)

    # Initialize W (D x L) - corresponds to H.T in sklearn
    W = avg * np.abs(rng.standard_normal(size=(n_features, n_components)))

    # Initialize exp(F) (N x L) - corresponds to W in sklearn
    exp_F = avg * np.abs(rng.standard_normal(size=(n_samples, n_components)))

    return W, exp_F


def _initialize_nndsvd(
    X: np.ndarray,
    n_components: int,
    variant: str = 'nndsvd',
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Nonnegative Double SVD initialization.

    Implements the NNDSVD algorithm from Boutsidis & Gallopoulos (2008)
    using TruncatedSVD for efficiency with large sparse matrices.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Non-negative input data matrix.
    n_components : int
        Number of components (must be <= min(n_samples, n_features)).
    variant : {'nndsvd', 'nndsvda', 'nndsvdar'}
        NNDSVD variant:
        - 'nndsvd': Basic NNDSVD (sparse, many zeros)
        - 'nndsvda': Fill zeros with average of X (dense)
        - 'nndsvdar': Fill zeros with small random values (dense, faster)
    random_state : int or None
        Random seed (used for 'nndsvdar').

    Returns
    -------
    W : ndarray of shape (n_features, n_components)
        Loadings matrix initialization.
    exp_F : ndarray of shape (n_samples, n_components)
        Expected latent factors initialization.

    References
    ----------
    C. Boutsidis, E. Gallopoulos (2008)
    "SVD based initialization: A head start for nonnegative matrix factorization"
    Pattern Recognition 41(4):1350-1362
    """
    if n_components > min(X.shape):
        raise ValueError(
            f"n_components ({n_components}) must be <= min(n_samples, n_features) "
            f"({min(X.shape)}) for NNDSVD initialization"
        )

    rng = np.random.RandomState(random_state)
    n_samples, n_features = X.shape
    eps = 1e-6

    # Convert to sparse matrix for efficiency if X is sparse
    # Use CSR format for efficient arithmetic and row slicing
    X_sparse = sparse.csr_matrix(X)

    # Step 1: Compute Truncated SVD (faster for large/sparse matrices)
    # Use 7 iterations by default (sklearn's default, works well in practice)
    # For very large sparse matrices, this is much faster than full SVD
    svd = TruncatedSVD(
        n_components=n_components,
        random_state=random_state,
        n_iter=7  # sklearn's default, good balance of speed and accuracy
    )
    U = svd.fit_transform(X_sparse)  # (n_samples, n_components)
    S = svd.singular_values_  # (n_components,)
    Vt = svd.components_      # (n_components, n_features)

    # Initialize output matrices
    # Note: In PNMF, we want W (D x L) and exp(F) (N x L)
    # Sklearn returns W_skl (N x L) and H_skl (L x D)
    # Mapping: exp(F) <- W_skl, W <- H_skl.T
    W = np.zeros((n_features, n_components), dtype=X.dtype)  # H.T in sklearn
    exp_F = np.zeros((n_samples, n_components), dtype=X.dtype)  # W in sklearn

    # Step 2: Leading singular triplet (j=0) is already non-negative
    W[:, 0] = np.sqrt(S[0]) * np.abs(Vt[0, :].T)   # First row of Vt
    exp_F[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])   # First column of U

    # Step 3: Process remaining components (j=1 to n_components-1)
    for j in range(1, n_components):
        x = U[:, j]      # j-th left singular vector
        y = Vt[j, :]     # j-th right singular vector

        # Extract positive and negative parts
        x_p = np.maximum(x, 0)
        x_n = np.abs(np.minimum(x, 0))
        y_p = np.maximum(y, 0)
        y_n = np.abs(np.minimum(y, 0))

        # Compute norms
        x_p_nrm = _norm(x_p)
        x_n_nrm = _norm(x_n)
        y_p_nrm = _norm(y_p)
        y_n_nrm = _norm(y_n)

        # Compute magnitudes
        m_p = x_p_nrm * y_p_nrm
        m_n = x_n_nrm * y_n_nrm

        # Choose the better update
        if m_p > m_n:
            u = x_p / (x_p_nrm + 1e-12)
            v = y_p / (y_p_nrm + 1e-12)
            sigma = m_p
        else:
            u = x_n / (x_n_nrm + 1e-12)
            v = y_n / (y_n_nrm + 1e-12)
            sigma = m_n

        lbd = np.sqrt(S[j] * sigma)
        exp_F[:, j] = lbd * u    # W[:, j] in sklearn notation
        W[:, j] = lbd * v.T      # H[j, :].T in sklearn notation

    # Step 4: Truncate small values to zero
    W[W < eps] = 0
    exp_F[exp_F < eps] = 0

    # Step 5: Apply variant-specific zero-filling
    if variant == 'nndsvd':
        pass  # Keep zeros (sparse result)
    elif variant == 'nndsvda':
        # Fill zeros with average of X
        avg = X.mean()
        W[W == 0] = avg
        exp_F[exp_F == 0] = avg
    elif variant == 'nndsvdar':
        # Fill zeros with small random values
        avg = X.mean()
        W_zeros = W == 0
        exp_F_zeros = exp_F == 0

        # Small random values: avg * |N(0,1)| / 100
        W[W_zeros] = np.abs(avg * rng.standard_normal(size=W_zeros.sum()) / 100)
        exp_F[exp_F_zeros] = np.abs(avg * rng.standard_normal(size=exp_F_zeros.sum()) / 100)
    else:
        raise ValueError(f"Unknown NNDSVD variant: {variant}")

    return W, exp_F


def _initialize_kmeans(
    X: np.ndarray,
    n_components: int,
    random_state: Optional[int] = None,
    max_iter: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    K-means based initialization.

    Uses cluster centroids and soft assignments to initialize W and F.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data matrix.
    n_components : int
        Number of clusters/components.
    random_state : int or None
        Random seed for k-means.
    max_iter : int
        Maximum iterations for k-means.

    Returns
    -------
    W : ndarray of shape (n_features, n_components)
        Loadings matrix initialization (cluster centroids transposed).
    exp_F : ndarray of shape (n_samples, n_components)
        Expected latent factors (soft assignment weights).

    Notes
    -----
    This method:
    1. Runs k-means to find cluster centroids
    2. Uses centroids as W (after appropriate scaling)
    3. Uses soft assignments based on distance to centroids for exp(F)
    """
    n_samples, n_features = X.shape

    # Run k-means
    kmeans = KMeans(
        n_clusters=n_components,
        random_state=random_state,
        max_iter=max_iter,
        n_init=10
    )
    labels = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_  # (n_components, n_features)

    # W: transpose of centroids (D x L)
    W = centroids.T  # (n_features, n_components)

    # exp_F: soft assignment weights (N x L)
    # Compute distances to all centroids
    from sklearn.metrics.pairwise import euclidean_distances
    distances = euclidean_distances(X, centroids)  # (n_samples, n_components)

    # Convert distances to weights (closer = higher weight)
    # Using exponential kernel: exp(-distance / mean_distance)
    mean_dist = distances.mean()
    exp_F = np.exp(-distances / (mean_dist + 1e-12))

    # Normalize rows
    row_sums = exp_F.sum(axis=1, keepdims=True)
    exp_F = exp_F / (row_sums + 1e-12)

    # Scale to match magnitude of X
    scale = X.mean() / exp_F.mean()
    exp_F = exp_F * scale
    W = W * scale

    return W, exp_F


def initialize_factors(
    X: Union[np.ndarray, torch.Tensor],
    n_components: int,
    init: Optional[str] = None,
    random_state: Optional[int] = None,
    scale_log_mu: bool = True,
    target_std: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initialize W and exp(F) matrices for PNMF.

    This function provides sklearn-compatible initialization for PNMF.
    The returned matrices can be used to initialize the model parameters.

    Parameters
    ----------
    X : ndarray or torch.Tensor of shape (n_samples, n_features)
        Non-negative input data matrix.
    n_components : int
        Number of latent components.
    init : {None, 'random', 'nndsvd', 'nndsvda', 'nndsvdar', 'k-means'}, default=None
        Initialization method:
        - None: Auto-select 'nndsvda' if n_components <= min(n_samples, n_features),
          otherwise 'random'.
        - 'random': Non-negative random matrices, scaled with sqrt(X.mean() / n_components).
        - 'nndsvd': Nonnegative Double SVD (better for sparseness).
        - 'nndsvda': NNDSVD with zeros filled with average of X (better for dense data).
        - 'nndsvdar': NNDSVD with zeros filled with small random values (faster dense).
        - 'k-means': K-means clustering based initialization.
    random_state : int or None
        Random seed for reproducibility.
    scale_log_mu : bool, default=True
        If True, scales exp_F so that log(exp_F) has approximately N(0, 1) distribution.
        This ensures the variational mean μ is on a consistent scale across different
        datasets and initialization methods.
    target_std : float, default=1.0
        Target standard deviation for log(exp_F) when scale_log_mu=True.

    Returns
    -------
    W : ndarray of shape (n_features, n_components)
        Loadings matrix initialization (for W parameter).
    exp_F : ndarray of shape (n_samples, n_components)
        Expected latent factors initialization (for exp(F) where F is in log-space).

    Examples
    --------
    >>> import numpy as np
    >>> from PNMF.initialization import initialize_factors
    >>> X = np.random.poisson(5, size=(100, 50)).astype(np.float32)
    >>> W, exp_F = initialize_factors(X, n_components=5, init='nndsvda', random_state=42)
    >>> print(W.shape)   # (50, 5)
    >>> print(exp_F.shape)  # (100, 5)

    Notes
    -----
    For PNMF, the model structure is:
        X ≈ W @ exp(F).T

    where:
    - W is (n_features, n_components) - loadings
    - F is the latent factor matrix (n_samples, n_components) in log-space
    - exp(F) is the expected value of the latent factors

    The variational distribution q(F) is Gaussian with mean μ and scale σ.
    To initialize from exp_F:
        μ = log(exp_F)  (or log(exp_F + eps) if exp_F has zeros)
        σ can be initialized to a small value (e.g., 0.1)
    """
    # Convert to numpy if needed
    if isinstance(X, torch.Tensor):
        X_np = X.detach().cpu().numpy()
    else:
        X_np = np.asarray(X)

    # Check for non-negative input
    if np.min(X_np) < 0:
        raise ValueError("Input matrix X must be non-negative for NMF initialization")

    n_samples, n_features = X_np.shape

    # Auto-select initialization if None
    if init is None:
        if n_components <= min(n_samples, n_features):
            init = 'nndsvd'
        else:
            init = 'random'

    # Dispatch to appropriate initialization
    if init == 'random':
        W, exp_F = _initialize_random(X_np, n_components, random_state)
    elif init in ('nndsvd', 'nndsvda', 'nndsvdar'):
        W, exp_F = _initialize_nndsvd(X_np, n_components, init, random_state)
    elif init == 'k-means':
        W, exp_F = _initialize_kmeans(X_np, n_components, random_state)
    else:
        raise ValueError(
            f"Unknown init parameter: '{init}'. "
            f"Valid options are: None, 'random', 'nndsvd', 'nndsvda', 'nndsvdar', 'k-means'"
        )

    # Scale exp_F so that log(exp_F) ~ N(0, target_std^2)
    # This ensures μ is on a consistent scale across different datasets
    if scale_log_mu:
        exp_F_before = exp_F.copy()
        exp_F = _scale_log_factors(exp_F, target_std=target_std)
        # Compute the scaling factor applied to exp_F
        scale_factor = exp_F.mean() / (exp_F_before.mean() + 1e-12)
        # Adjust W inversely to maintain X ≈ W @ exp(F).T approximation
        # If exp_F is scaled by s, W should be scaled by 1/s
        W = W / scale_factor

    return W, exp_F
