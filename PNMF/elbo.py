"""
ELBO computation functions for PNMF.

This module provides different strategies for computing the expected log-likelihood
E[log p(Y|F)] and the full Evidence Lower Bound (ELBO).

Expected log-likelihood modes:
- simple: Full Monte Carlo estimation using torch.distributions.Poisson
- expanded: Hybrid Monte Carlo + analytic expectation (default, lower variance)
- lower-bound: Jensen's lower bound (fully analytic, no MC sampling)

The ELBO is computed as:
    ELBO = E[log p(Y|F)] - KL[q(F) || p(F)]

The KL divergence is computed separately to allow for custom KL implementations.
"""

import torch


def poisson_log_likelihood(X: torch.Tensor, rate: torch.Tensor) -> torch.Tensor:
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
    log_likelihood = (X * torch.log(rate) - rate).sum()
    return log_likelihood


# =============================================================================
# Expected log-likelihood functions (modes)
# =============================================================================


def expected_log_likelihood_simple(rate, X):
    """
    Compute expected log-likelihood using full Monte Carlo estimation.

    Uses torch.distributions.Poisson.log_prob() for computing the
    log-likelihood, providing a clean and numerically stable implementation.

    The Poisson log-likelihood is:
        log p(X|rate) = X * log(rate) - rate - log(X!)

    Args:
        rate: Poisson rate tensor of shape (E, D, N)
        X: Input data tensor of shape (D, N)

    Returns:
        Expected log-likelihood E[log p(Y|F)] (scalar)
    """
    E_samples = rate.shape[0]

    # Expand X to match rate shape: (D, N) -> (E, D, N)
    X_expanded = X.unsqueeze(0).expand(E_samples, -1, -1)

    # Use torch.distributions.Poisson for clean, numerically stable computation
    poisson_dist = torch.distributions.Poisson(rate=rate)
    log_likelihood_mc = poisson_dist.log_prob(X_expanded)

    # Expected log likelihood via Monte Carlo: (1/E) * sum_e log p(Y|F_e)
    expected_log_likelihood = log_likelihood_mc.sum() / E_samples

    return expected_log_likelihood


def expected_log_likelihood_expanded(rate, qF, X, W):
    """
    Compute expected log-likelihood using the expanded expectation form.

    The expected log-likelihood is computed as:
        E[log p(Y|F)] = Y * E[log(sum(W * exp(F)))] - sum(W * E[exp(F)]) - sum(log(Y!))

    where:
    - The first term uses Monte Carlo estimation (requires log of sum)
    - The second term is computed analytically using E[exp(F)] = exp(mu + sigma^2/2)
    - The third term is the Poisson normalization constant (log factorial)

    Args:
        rate: Poisson rate tensor of shape (E, D, N)
        qF: Variational posterior distribution with mean and scale
        X: Input data tensor of shape (D, N)
        W: Loadings matrix tensor of shape (D, L)

    Returns:
        Expected log-likelihood E[log p(Y|F)] (scalar)
    """
    E_samples = rate.shape[0]

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
    term2_analytic = torch.matmul(W, exp_expectation).sum()  # scalar

    # Expected log likelihood (including the Poisson normalization -log(Y!) term)
    # Using lgamma(X+1) = log(X!) for the Poisson PMF normalization
    expected_log_likelihood = term1_mc - term2_analytic - torch.lgamma(X + 1).sum()

    return expected_log_likelihood


def expected_log_likelihood_lower_bound(qF, X, W):
    """
    Compute expected log-likelihood using Jensen's lower bound (fully analytic).

    Uses Jensen's inequality for the log-sum-exp term:
        E[log Σ W * exp(F)] ≥ log Σ W * exp(E[F])

    For Gaussian variational distribution with mean μ and variance σ²:
        E[F] = μ

    This gives a true lower bound on E[log p(Y|F)] with NO Monte Carlo sampling.

    The expected log-likelihood is computed as:
        E[log p(Y|F)] ≈ Y * log(Σ W * exp(μ)) - Σ W * exp(μ + σ²/2) - log(Y!)

    Args:
        qF: Variational posterior distribution with mean and scale
        X: Input data tensor of shape (D, N)
        W: Loadings matrix tensor of shape (D, L)

    Returns:
        Lower bound on expected log-likelihood E[log p(Y|F)] (scalar)
    """
    # Get variational parameters
    mu = qF.mean  # (L, N)
    sigma = qF.scale  # (L, N)

    # --- First term (Jensen lower bound): Y_ij * log(Σ_l W_jl * exp(μ_il)) ---
    # Lower bound: E[log Σ W * exp(F)] ≥ log Σ W * exp(E[F]) = log Σ W * exp(μ)
    exp_mu = torch.exp(mu)  # (L, N)

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
    # Note: Since we use a lower bound on the log term, this is a true lower bound
    expected_log_likelihood = term1_lower - term2_analytic - torch.lgamma(X + 1).sum()

    return expected_log_likelihood


def expected_log_likelihood(mode, rate, qF, X, W):
    """
    Compute expected log-likelihood E[log p(Y|F)].

    This function dispatches to the appropriate computation based on mode:
    - 'simple': Uses torch.distributions.Poisson.log_prob() directly
    - 'expanded': Uses hybrid Monte Carlo + analytic expectation
    - 'lower-bound': Uses Jensen's lower bound (fully analytic, no MC)

    Args:
        mode: Computation mode ('simple', 'expanded', or 'lower-bound')
        rate: Poisson rate tensor of shape (E, D, N) [unused for lower-bound]
        qF: Variational posterior distribution with mean and scale
        X: Input data tensor of shape (D, N)
        W: Loadings matrix tensor of shape (D, L)

    Returns:
        Expected log-likelihood E[log p(Y|F)] (scalar)
    """
    if mode == 'simple':
        return expected_log_likelihood_simple(rate, X)
    elif mode == 'lower-bound':
        return expected_log_likelihood_lower_bound(qF, X, W)
    else:  # 'expanded'
        return expected_log_likelihood_expanded(rate, qF, X, W)


# =============================================================================
# KL divergence
# =============================================================================


def kl_divergence(qF, pF):
    """
    Compute KL divergence between variational posterior and prior.

    KL[q(F) || p(F)]

    Args:
        qF: Variational posterior distribution
        pF: Prior distribution

    Returns:
        KL divergence (scalar)
    """
    return torch.distributions.kl_divergence(qF, pF).sum()


# =============================================================================
# Full ELBO computation
# =============================================================================


def compute_elbo(mode, rate, qF, pF, X, W, kl_fn=None):
    """
    Compute the Evidence Lower BOund (ELBO).

    ELBO = E[log p(Y|F)] - KL[q(F) || p(F)]

    This function computes the expected log-likelihood using the specified mode
    and subtracts the KL divergence. A custom KL function can be provided.

    Args:
        mode: Expected log-likelihood mode ('simple', 'expanded', or 'lower-bound')
        rate: Poisson rate tensor of shape (E, D, N) [unused for lower-bound]
        qF: Variational posterior distribution with mean and scale
        pF: Prior distribution
        X: Input data tensor of shape (D, N)
        W: Loadings matrix tensor of shape (D, L)
        kl_fn: Optional custom KL divergence function. If None, uses standard
               torch.distributions.kl_divergence. Should take (qF, pF) and return scalar.

    Returns:
        Expected log-likelihood and KL divergence
    """
    # Compute expected log-likelihood using the specified mode
    exp_log_likelihood = expected_log_likelihood(mode, rate, qF, X, W)

    # Compute KL divergence (use custom function if provided)
    if kl_fn is not None:
        kl = kl_fn(qF, pF)
    else:
        kl = kl_divergence(qF, pF)

    # return expected log-likelihood and KL divergence
    return exp_log_likelihood, kl
