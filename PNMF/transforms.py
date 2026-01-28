"""
Transform and utility functions for PNMF.

This module provides:
1. Conditional inference functions (transform_W, transform_F)
2. Factor extraction functions (log_factors, factors, factor_uncertainty, factor_samples)
3. Model accessor functions (get_loadings, get_prior)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Union, Optional
from tqdm.auto import tqdm

from .elbo import compute_elbo
from .priors import GaussianPrior


# =============================================================================
# Factor Extraction Functions
# =============================================================================

def log_factors(model, return_tensor: bool = False) -> Union[np.ndarray, torch.Tensor]:
    """
    Get log-space latent factors (μ from q(F) = Normal(μ, σ²)).

    Parameters
    ----------
    model : PNMF
        A fitted PNMF model.
    return_tensor : bool, default=False
        If True, return a torch.Tensor instead of numpy array.

    Returns
    -------
    F_log : ndarray or Tensor of shape (n_samples, n_components)
        Latent factors in log-space (μ from the variational distribution).

    Examples
    --------
    >>> from PNMF import PNMF
    >>> from PNMF.transforms import log_factors
    >>> model = PNMF(n_components=5).fit(X)
    >>> F_log = log_factors(model)  # Shape: (n_samples, n_components)
    """
    if model._prior is None:
        raise ValueError("Model has not been fitted yet.")

    # Get mean from the variational distribution
    # Prior stores it as (L, N), we transpose to (N, L) for sklearn-style output
    if model._prior.use_natural_gradients:
        # Natural parameterization: convert to mean
        theta1 = model._prior.theta1.detach()
        theta2 = model._prior.theta2.detach()
        # mu = theta1 / (-2 * theta2) = theta1 * s^2
        # s^2 = -1 / (2 * theta2)
        s_sq = -1 / (2 * theta2)
        mu = theta1 * s_sq
    else:
        mu = model._prior.mean.detach()

    # Transpose from (L, N) to (N, L) for sklearn-style
    F_log = mu.T

    if return_tensor:
        return F_log
    return F_log.cpu().numpy()


def get_factors(
    model,
    use_mgf: bool = False,
    return_tensor: bool = False
) -> Union[np.ndarray, torch.Tensor]:
    """
    Get exp-space latent factors.

    By default, computes the expected value E[exp(F)] using the moment-generating
    function of the Gaussian: E[exp(F)] = exp(μ + σ²/2).

    Parameters
    ----------
    model : PNMF
        A fitted PNMF model.
    use_mgf : bool, default=False
        If True, compute E[exp(F)] = exp(μ + σ²/2) using the MGF.
        If False, compute exp(μ) directly.
    return_tensor : bool, default=False
        If True, return a torch.Tensor instead of numpy array.

    Returns
    -------
    F_exp : ndarray or Tensor of shape (n_samples, n_components)
        Latent factors in exp-space.

    Notes
    -----
    The moment-generating function (MGF) of a Gaussian gives:
        E[exp(F)] = exp(μ + σ²/2)

    This is the true expected value under the variational distribution q(F).
    Using use_mgf=False gives exp(μ), which is a biased estimate.

    Examples
    --------
    >>> from PNMF import PNMF
    >>> from PNMF.transforms import factors
    >>> model = PNMF(n_components=5).fit(X)
    >>> F_exp = factors(model)  # E[exp(F)], shape: (n_samples, n_components)
    >>> F_exp_biased = factors(model, use_mgf=False)  # exp(μ)
    """
    if model._prior is None:
        raise ValueError("Model has not been fitted yet.")

    # Get mean and scale from the variational distribution
    if model._prior.use_natural_gradients:
        theta1 = model._prior.theta1.detach()
        theta2 = model._prior.theta2.detach()
        s_sq = -1 / (2 * theta2)
        mu = theta1 * s_sq
        scale_sq = s_sq
    else:
        mu = model._prior.mean.detach()
        scale = model._prior.scale.data.detach().clamp(min=1e-8)
        scale_sq = scale ** 2

    if use_mgf:
        # E[exp(F)] = exp(μ + σ²/2)
        F_exp = torch.exp(mu + scale_sq / 2)
    else:
        # exp(μ) - biased estimate
        F_exp = torch.exp(mu)

    # Transpose from (L, N) to (N, L) for sklearn-style
    F_exp = F_exp.T

    if return_tensor:
        return F_exp
    return F_exp.cpu().numpy()


def factor_uncertainty(
    model,
    return_variance: bool = False,
    return_tensor: bool = False
) -> Union[np.ndarray, torch.Tensor]:
    """
    Get uncertainty in latent factors (σ or σ² from q(F) = Normal(μ, σ²)).

    Parameters
    ----------
    model : PNMF
        A fitted PNMF model.
    return_variance : bool, default=False
        If True, return variance (σ²). If False, return standard deviation (σ).
    return_tensor : bool, default=False
        If True, return a torch.Tensor instead of numpy array.

    Returns
    -------
    uncertainty : ndarray or Tensor of shape (n_samples, n_components)
        Standard deviation (σ) or variance (σ²) of the variational distribution.

    Examples
    --------
    >>> from PNMF import PNMF
    >>> from PNMF.transforms import factor_uncertainty
    >>> model = PNMF(n_components=5).fit(X)
    >>> F_std = factor_uncertainty(model)  # σ, shape: (n_samples, n_components)
    >>> F_var = factor_uncertainty(model, return_variance=True)  # σ²
    """
    if model._prior is None:
        raise ValueError("Model has not been fitted yet.")

    # Get scale from the variational distribution
    if model._prior.use_natural_gradients:
        theta2 = model._prior.theta2.detach()
        s_sq = -1 / (2 * theta2)
        if return_variance:
            result = s_sq
        else:
            result = torch.sqrt(s_sq)
    else:
        scale = model._prior.scale.data.detach().clamp(min=1e-8)
        if return_variance:
            result = scale ** 2
        else:
            result = scale

    # Transpose from (L, N) to (N, L) for sklearn-style
    result = result.T

    if return_tensor:
        return result
    return result.cpu().numpy()


def factor_samples(
    model,
    n_samples: int = 100,
    return_exp: bool = False,
    return_tensor: bool = False
) -> Union[np.ndarray, torch.Tensor]:
    """
    Sample latent factors from the variational posterior q(F).

    Parameters
    ----------
    model : PNMF
        A fitted PNMF model.
    n_samples : int, default=100
        Number of samples to draw.
    return_exp : bool, default=False
        If True, return exp(F) samples. If False, return F samples.
    return_tensor : bool, default=False
        If True, return a torch.Tensor instead of numpy array.

    Returns
    -------
    samples : ndarray or Tensor of shape (n_samples, n_data_samples, n_components)
        Samples from the variational posterior.

    Notes
    -----
    Uses the reparameterization trick for sampling.

    Examples
    --------
    >>> from PNMF import PNMF
    >>> from PNMF.transforms import factor_samples
    >>> model = PNMF(n_components=5).fit(X)
    >>> samples = factor_samples(model, n_samples=100)  # (100, n_data, 5)
    >>> exp_samples = factor_samples(model, n_samples=50, return_exp=True)
    """
    if model._prior is None:
        raise ValueError("Model has not been fitted yet.")

    # Get the variational distribution
    qF, _ = model._prior()

    # Sample using reparameterization trick
    # qF.rsample returns shape (n_samples, L, N)
    samples = qF.rsample((n_samples,))

    if return_exp:
        samples = torch.exp(samples)

    # Transpose from (n_samples, L, N) to (n_samples, N, L) for sklearn-style
    samples = samples.permute(0, 2, 1)

    if return_tensor:
        return samples.detach()
    return samples.detach().cpu().numpy()


# =============================================================================
# Model Accessor Functions
# =============================================================================

def get_loadings(model, return_tensor: bool = False) -> Union[np.ndarray, torch.Tensor]:
    """
    Get the loadings matrix W from a fitted PNMF model.

    Parameters
    ----------
    model : PNMF
        A fitted PNMF model.
    return_tensor : bool, default=False
        If True, return a torch.Tensor instead of numpy array.

    Returns
    -------
    W : ndarray or Tensor of shape (n_features, n_components)
        The loadings matrix W.

    Notes
    -----
    This is equivalent to model.components_.T but provides a consistent
    interface and works with both fitted and unfitted (but initialized) models.

    Examples
    --------
    >>> from PNMF import PNMF
    >>> from PNMF.transforms import get_loadings
    >>> model = PNMF(n_components=5).fit(X)
    >>> W = get_loadings(model)  # Shape: (n_features, n_components)
    """
    if model._model is None:
        raise ValueError("Model has not been fitted yet.")

    W = model._model.W.data.detach()

    if return_tensor:
        return W
    return W.cpu().numpy()


def get_prior(model) -> GaussianPrior:
    """
    Get the GaussianPrior object from a fitted PNMF model.

    Parameters
    ----------
    model : PNMF
        A fitted PNMF model.

    Returns
    -------
    prior : GaussianPrior
        The GaussianPrior object containing the variational distribution.

    Notes
    -----
    This provides access to the full prior for advanced users who want
    to directly manipulate or inspect the variational distribution.

    Examples
    --------
    >>> from PNMF import PNMF
    >>> from PNMF.transforms import get_prior
    >>> model = PNMF(n_components=5).fit(X)
    >>> prior = get_prior(model)
    >>> qF, pF = prior()  # Get distributions directly
    """
    if model._prior is None:
        raise ValueError("Model has not been fitted yet.")

    return model._prior


# =============================================================================
# Conditional Inference Functions
# =============================================================================

def transform_F(
    X: Union[np.ndarray, torch.Tensor],
    W: Union[np.ndarray, torch.Tensor],
    n_components: Optional[int] = None,
    mode: str = 'expanded',
    E: int = 3,
    max_iter: int = 200,
    tol: float = 1e-4,
    learning_rate: float = 0.01,
    verbose: bool = False,
    device: str = 'auto',
    return_prior: bool = False
) -> Union[np.ndarray, GaussianPrior]:
    """
    Learn new latent factors F conditioned on fixed loadings W using variational inference.

    This is a Bayesian alternative to NNLS-based transform. It learns a full
    variational distribution q(F) for the new data given the fixed W.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        New data to transform.
    W : array-like of shape (n_features, n_components)
        Fixed loadings matrix.
    n_components : int, optional
        Number of components. Inferred from W if not specified.
    mode : {'simple', 'expanded', 'lower-bound'}, default='expanded'
        ELBO computation mode.
    E : int, default=3
        Number of Monte Carlo samples for ELBO estimation.
    max_iter : int, default=200
        Maximum number of iterations.
    tol : float, default=1e-4
        Tolerance for convergence.
    learning_rate : float, default=0.01
        Learning rate for the optimizer.
    verbose : bool, default=False
        Whether to print progress messages.
    device : {'cpu', 'cuda', 'mps', 'auto'}, default='auto'
        Device to use for computation.
    return_prior : bool, default=False
        If True, return the GaussianPrior object instead of exp(F).

    Returns
    -------
    F_exp : ndarray of shape (n_samples, n_components)
        Transformed data E[exp(F)]. Returned if return_prior=False.
    prior : GaussianPrior
        The fitted GaussianPrior object. Returned if return_prior=True.

    Examples
    --------
    >>> from PNMF import PNMF
    >>> from PNMF.transforms import transform_F, get_loadings
    >>> # Fit original model
    >>> model = PNMF(n_components=5).fit(X_train)
    >>> W = get_loadings(model)
    >>> # Transform new data with full VI
    >>> F_new = transform_F(X_new, W)  # Shape: (n_new, 5)
    >>> # Or get the full prior for uncertainty quantification
    >>> prior_new = transform_F(X_new, W, return_prior=True)
    """
    # Determine device
    if device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(device)

    # Convert inputs to tensors
    if isinstance(X, np.ndarray):
        X_np = X
    else:
        X_np = X.detach().cpu().numpy()

    if isinstance(W, np.ndarray):
        W_np = W
    else:
        W_np = W.detach().cpu().numpy()

    n_samples, n_features = X_np.shape
    n_features_w, L = W_np.shape

    if n_features != n_features_w:
        raise ValueError(f"Feature dimension mismatch: X has {n_features}, W has {n_features_w}")

    if n_components is None:
        n_components = L
    elif n_components != L:
        raise ValueError(f"n_components ({n_components}) does not match W shape ({L})")

    # Convert to torch tensors (transpose X to (D, N) for internal representation)
    X_torch = torch.from_numpy(X_np.T.astype(np.float32)).to(device)
    W_torch = torch.from_numpy(W_np.astype(np.float32)).to(device)

    # Initialize prior for new data
    prior = GaussianPrior(y=X_torch, L=n_components, use_natural_gradients=False).to(device)

    # Optimizer for variational parameters only
    optimizer = torch.optim.Adam(prior.parameters(), lr=learning_rate)

    # Training loop
    prev_elbo = float('-inf')
    pbar = tqdm(range(max_iter), disable=not verbose, desc="transform_F")

    for iteration in pbar:
        optimizer.zero_grad()

        # Get variational distribution
        qF, pF = prior()

        # Sample F
        F_samples = qF.rsample((E,))  # (E, L, N)

        # Compute rate: W @ exp(F) -> (E, D, N)
        F_exp = torch.exp(F_samples)  # (E, L, N)
        rate = torch.matmul(W_torch, F_exp)  # (E, D, N)

        # Compute ELBO loss
        loss = compute_elbo(mode, rate, qF, pF, X_torch, W_torch)

        # Backward and step
        loss.backward()
        optimizer.step()

        elbo_value = -loss.item()

        if verbose:
            pbar.set_postfix({"ELBO": f"{elbo_value:.6f}"})

        if abs(elbo_value - prev_elbo) < tol:
            if verbose:
                pbar.set_postfix({"ELBO": f"{elbo_value:.6f}", "status": "converged"})
                pbar.close()
            break

        prev_elbo = elbo_value

    if return_prior:
        return prior

    # Return E[exp(F)] = exp(μ + σ²/2)
    mu = prior.mean.detach()
    scale = prior.scale.data.detach().clamp(min=1e-8)
    F_exp = torch.exp(mu + scale ** 2 / 2)

    # Transpose from (L, N) to (N, L) for sklearn-style
    return F_exp.T.cpu().numpy()


def transform_W(
    X: Union[np.ndarray, torch.Tensor],
    F: Union[np.ndarray, torch.Tensor],
    W_init: Optional[Union[np.ndarray, torch.Tensor]] = None,
    n_components: Optional[int] = None,
    max_iter: int = 100,
    tol: float = 1e-4,
    verbose: bool = False,
    device: str = 'auto'
) -> np.ndarray:
    """
    Learn new loadings W conditioned on fixed latent factors F.

    Uses multiplicative updates (NNLS-style) for fast convergence while
    displaying the Poisson negative log-likelihood on the progress bar.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data to fit.
    F : array-like of shape (n_samples, n_components)
        Fixed latent factors in exp-space (exp(F) from the model).
    W_init : array-like of shape (n_features, n_components), optional
        Initial loadings W. If None, uses random initialization.
    n_components : int, optional
        Number of components. Inferred from F if not specified.
    max_iter : int, default=100
        Maximum number of iterations.
    tol : float, default=1e-4
        Tolerance for convergence.
    verbose : bool, default=False
        Whether to print progress messages.
    device : {'cpu', 'cuda', 'mps', 'auto'}, default='auto'
        Device to use for computation.

    Returns
    -------
    W : ndarray of shape (n_features, n_components)
        The learned loadings matrix.

    Notes
    -----
    This function uses multiplicative updates for non-negative matrix
    factorization, which is equivalent to coordinate descent on the
    Poisson log-likelihood.

    The update rule for W is:
        W = W * (X @ H) / (W @ (H.T @ H) + eps)

    where H = F (the fixed factors in exp-space).

    For uncertainty quantification in F, consider using factor_samples() and
    running this function multiple times with different F samples.

    Examples
    --------
    >>> from PNMF import PNMF
    >>> from PNMF.transforms import transform_W, get_factors
    >>> # Fit original model
    >>> model = PNMF(n_components=5).fit(X_train)
    >>> F_exp = get_factors(model)  # exp-space factors
    >>> # Learn new W for different data with same factors
    >>> W_new = transform_W(X_new, F_exp)  # Shape: (n_features, 5)
    """
    # Determine device
    if device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(device)

    # Convert inputs to tensors on device
    if isinstance(X, np.ndarray):
        X_torch = torch.from_numpy(X.astype(np.float32)).to(device)
    else:
        X_torch = X.to(device).float()

    if isinstance(F, np.ndarray):
        F_torch = torch.from_numpy(F.astype(np.float32)).to(device)
    else:
        F_torch = F.to(device).float()

    n_samples, n_features = X_torch.shape
    n_samples_f, L = F_torch.shape

    if n_samples != n_samples_f:
        raise ValueError(f"Sample dimension mismatch: X has {n_samples}, F has {n_samples_f}")

    if n_components is None:
        n_components = L
    elif n_components != L:
        raise ValueError(f"n_components ({n_components}) does not match F shape ({L})")

    # Initialize W: (n_features, n_components)
    if W_init is not None:
        if isinstance(W_init, np.ndarray):
            W = torch.from_numpy(W_init.astype(np.float32)).to(device)
        else:
            W = W_init.to(device).float()
    else:
        W = torch.rand(n_features, n_components, device=device) * 0.1

    # Precompute H.T @ H for multiplicative updates (H = F in our notation)
    # F: (n_samples, n_components), F.T @ F: (n_components, n_components)
    with torch.no_grad():
        FtF = F_torch.T @ F_torch  # (L, L)

        # Training loop with multiplicative updates
        prev_nll = float('inf')
        pbar = tqdm(range(max_iter), disable=not verbose, desc="transform_W")

        for iteration in pbar:
            # Multiplicative update for W
            # W = W * (X.T @ F) / (W @ (F.T @ F) + eps)
            numerator = X_torch.T @ F_torch  # (n_features, n_components)
            denominator = W @ FtF + 1e-8  # (n_features, n_components)
            W = W * numerator / denominator

            # Compute NLL for monitoring (Poisson log-likelihood)
            # rate = F @ W.T -> (n_samples, n_features)
            rate = F_torch @ W.T  # (n_samples, n_features)
            log_rate = torch.log(rate.clamp(min=1e-8))
            nll = -(X_torch * log_rate - rate).sum().item()

            if verbose:
                pbar.set_postfix({"NLL": f"{nll:.6f}"})

            if abs(nll - prev_nll) < tol:
                if verbose:
                    pbar.set_postfix({"NLL": f"{nll:.6f}", "status": "converged"})
                    pbar.close()
                break

            prev_nll = nll

    return W.cpu().numpy()


# =============================================================================
# Utility Functions for Working with Priors
# =============================================================================

def log_factors_from_prior(
    prior: GaussianPrior,
    return_tensor: bool = False
) -> Union[np.ndarray, torch.Tensor]:
    """
    Get log-space latent factors from a GaussianPrior object.

    Parameters
    ----------
    prior : GaussianPrior
        A GaussianPrior object.
    return_tensor : bool, default=False
        If True, return a torch.Tensor instead of numpy array.

    Returns
    -------
    F_log : ndarray or Tensor of shape (n_samples, n_components)
        Latent factors in log-space.

    Examples
    --------
    >>> from PNMF.transforms import transform_F, log_factors_from_prior
    >>> prior = transform_F(X_new, W, return_prior=True)
    >>> F_log = log_factors_from_prior(prior)
    """
    if prior.use_natural_gradients:
        theta1 = prior.theta1.detach()
        theta2 = prior.theta2.detach()
        s_sq = -1 / (2 * theta2)
        mu = theta1 * s_sq
    else:
        mu = prior.mean.detach()

    F_log = mu.T

    if return_tensor:
        return F_log
    return F_log.cpu().numpy()


def factors_from_prior(
    prior: GaussianPrior,
    use_mgf: bool = True,
    return_tensor: bool = False
) -> Union[np.ndarray, torch.Tensor]:
    """
    Get exp-space latent factors from a GaussianPrior object.

    Parameters
    ----------
    prior : GaussianPrior
        A GaussianPrior object.
    use_mgf : bool, default=True
        If True, compute E[exp(F)] = exp(μ + σ²/2).
        If False, compute exp(μ) directly.
    return_tensor : bool, default=False
        If True, return a torch.Tensor instead of numpy array.

    Returns
    -------
    F_exp : ndarray or Tensor of shape (n_samples, n_components)
        Latent factors in exp-space.

    Examples
    --------
    >>> from PNMF.transforms import transform_F, factors_from_prior
    >>> prior = transform_F(X_new, W, return_prior=True)
    >>> F_exp = factors_from_prior(prior)
    """
    if prior.use_natural_gradients:
        theta1 = prior.theta1.detach()
        theta2 = prior.theta2.detach()
        s_sq = -1 / (2 * theta2)
        mu = theta1 * s_sq
        scale_sq = s_sq
    else:
        mu = prior.mean.detach()
        scale = prior.scale.data.detach().clamp(min=1e-8)
        scale_sq = scale ** 2

    if use_mgf:
        F_exp = torch.exp(mu + scale_sq / 2)
    else:
        F_exp = torch.exp(mu)

    F_exp = F_exp.T

    if return_tensor:
        return F_exp
    return F_exp.cpu().numpy()


def uncertainty_from_prior(
    prior: GaussianPrior,
    return_variance: bool = False,
    return_tensor: bool = False
) -> Union[np.ndarray, torch.Tensor]:
    """
    Get uncertainty from a GaussianPrior object.

    Parameters
    ----------
    prior : GaussianPrior
        A GaussianPrior object.
    return_variance : bool, default=False
        If True, return variance (σ²). If False, return standard deviation (σ).
    return_tensor : bool, default=False
        If True, return a torch.Tensor instead of numpy array.

    Returns
    -------
    uncertainty : ndarray or Tensor of shape (n_samples, n_components)
        Standard deviation or variance.

    Examples
    --------
    >>> from PNMF.transforms import transform_F, uncertainty_from_prior
    >>> prior = transform_F(X_new, W, return_prior=True)
    >>> F_std = uncertainty_from_prior(prior)
    """
    if prior.use_natural_gradients:
        theta2 = prior.theta2.detach()
        s_sq = -1 / (2 * theta2)
        if return_variance:
            result = s_sq
        else:
            result = torch.sqrt(s_sq)
    else:
        scale = prior.scale.data.detach().clamp(min=1e-8)
        if return_variance:
            result = scale ** 2
        else:
            result = scale

    result = result.T

    if return_tensor:
        return result
    return result.cpu().numpy()
