"""
ELBO computation functions for PNMF.

This module provides different strategies for computing the Evidence Lower Bound (ELBO):
- simple: Full Monte Carlo estimation using torch.distributions.Poisson
- expanded: Hybrid Monte Carlo + analytic expectation (default, lower variance)
- lower-bound: Jensen's lower bound (fully analytic, no MC sampling)
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
    log_lik = (X * torch.log(rate) - rate).sum()
    return log_lik


def compute_elbo_simple(rate, qF, pF, X):
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


def compute_elbo_expanded(rate, qF, pF, X, W):
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
        W: Loadings matrix tensor of shape (D, L)

    Returns:
        Negative ELBO (to minimize)
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
    log_lik = term1_mc - term2_analytic - torch.lgamma(X + 1).sum()

    # KL divergence
    kl = torch.distributions.kl_divergence(qF, pF).sum()

    # Negative ELBO (for minimization)
    return kl - log_lik


def compute_elbo_lower_bound(qF, pF, X, W):
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
        W: Loadings matrix tensor of shape (D, L)

    Returns:
        Negative ELBO (to minimize)
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
    log_lik = term1_lower - term2_analytic - torch.lgamma(X + 1).sum()

    # KL divergence
    kl = torch.distributions.kl_divergence(qF, pF).sum()

    # Negative ELBO (for minimization)
    # Note: Since we use a lower bound on the log term, this is a true lower bound on ELBO
    return kl - log_lik


def compute_elbo(mode, rate, qF, pF, X, W):
    """
    Compute the Evidence Lower BOund (ELBO).

    This function dispatches to the appropriate ELBO computation based on mode:
    - 'simple': Uses torch.distributions.Poisson.log_prob() directly
    - 'expanded': Uses hybrid Monte Carlo + analytic expectation
    - 'lower-bound': Uses Jensen's lower bound (fully analytic, no MC)

    ELBO = E[log p(Y|F)] - KL[q(F) || p(F)]

    Args:
        mode: ELBO computation mode ('simple', 'expanded', or 'lower-bound')
        rate: Poisson rate tensor of shape (E, D, N) [unused for lower-bound]
        qF: Variational posterior distribution with mean and scale
        pF: Prior distribution
        X: Input data tensor of shape (D, N)
        W: Loadings matrix tensor of shape (D, L)

    Returns:
        Negative ELBO (to minimize)
    """
    if mode == 'simple':
        return compute_elbo_simple(rate, qF, pF, X)
    elif mode == 'lower-bound':
        return compute_elbo_lower_bound(qF, pF, X, W)
    else:  # 'expanded'
        return compute_elbo_expanded(rate, qF, pF, X, W)
