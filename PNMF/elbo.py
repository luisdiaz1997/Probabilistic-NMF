"""
ELBO computation functions for PNMF.

This module provides different strategies for computing the expected log-likelihood
E[log p(Y|F)] and the full Evidence Lower Bound (ELBO).

Expected log-likelihood modes:
- simple: Full Monte Carlo estimation (X * log(rate) - rate - log(X!))
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
# Precompute log-likelihood terms
# =============================================================================


def compute_log_likelihood_terms(W, qF, X, E, mode, return_samples=False):
    """
    Precompute all intermediate tensors needed for expected log-likelihood.

    This function centralizes the computation of:
    - exp(μ) and exp(μ + σ²/2) — computed once, reused across modes
    - W @ exp(μ) and W @ exp(μ + σ²/2) — computed once via 2D matmul
    - MC scalar accumulations for the ELBO (term1_mc, term2_mc)
    - Optionally, full sample tensors for downstream consumers that need the
      same MC samples (e.g. multiplicative W updates)

    Args:
        W: Loadings matrix, shape (D, L)
        qF: Variational posterior (torch.distributions.Normal) with .mean and .scale
        X: Input data, shape (D, N). Required for MC modes.
        E: Number of Monte Carlo samples. Ignored for lower-bound mode.
        mode: 'simple', 'expanded', or 'lower-bound'
        return_samples: If True, also store 'exp_F_samples' (E, L, N) in the
                        returned dict so that downstream code (e.g. multiplicative
                        W update) can reuse the same MC samples. Default False.
                        Note: rate_mc (E, D, N) is NOT stored — downstream code
                        should compute W @ exp_F_samples itself if needed.

    Returns:
        dict with keys (only those needed by the given mode):
            Always present:
                'exp_mu':        exp(μ),              shape (L, N)

            When mode in ['expanded', 'lower-bound']:
                'exp_mu_sigma':  exp(μ + σ²/2),       shape (L, N)
                'rate_mgf':      W @ exp(μ + σ²/2),   shape (D, N)

            When mode == 'lower-bound':
                'rate_mean':     W @ exp(μ),           shape (D, N)

            When mode in ['simple', 'expanded']:
                'term1_mc':    (1/E) * Σ_e (X * log(rate_e)).sum(),  scalar
                'term2_mc':    (1/E) * Σ_e rate_e.sum(),             scalar  [simple only]

            When mode in ['simple', 'expanded'] and return_samples=True:
                'exp_F_samples':  exp(F_samples),       shape (E, L, N)
    """
    mu = qF.mean       # (L, N)
    sigma = qF.scale    # (L, N)

    # exp(μ) — always needed (MC loop uses it, lower-bound uses it)
    exp_mu = torch.exp(mu)                          # (L, N)
    terms = {'exp_mu': exp_mu}

    # exp(μ + σ²/2) and W @ exp(μ + σ²/2) — needed by expanded and lower-bound
    if mode in ('expanded', 'lower-bound'):
        exp_mu_sigma = torch.exp(mu + 0.5 * sigma ** 2)  # (L, N)
        terms['exp_mu_sigma'] = exp_mu_sigma
        terms['rate_mgf'] = torch.matmul(W, exp_mu_sigma) # (D, N)

    # W @ exp(μ) — only needed by lower-bound
    if mode == 'lower-bound':
        terms['rate_mean'] = torch.matmul(W, exp_mu)       # (D, N)

    # For lower-bound mode, no MC computation needed
    if mode == 'lower-bound':
        return terms

    # MC modes: simple or expanded
    # Memory-efficient loop — accumulate (D, N) buffers, never materialize (E, D, N)
    # When return_samples=True, also collect exp_F_samples (E, L, N) for downstream reuse
    D, N = X.shape
    log_acc = torch.zeros(D, N, dtype=X.dtype, device=X.device)
    linear_acc = torch.zeros(D, N, dtype=X.dtype, device=X.device) if mode == 'simple' else None
    exp_F_list = [] if return_samples else None

    for e in range(E):
        eps_e = torch.randn_like(mu)                          # (L, N)
        perturbation = torch.exp(eps_e * sigma)                # (L, N)
        exp_F_e = exp_mu * perturbation                        # (L, N)
        rate_e = torch.matmul(W, exp_F_e)                      # (D, N)
        log_acc += X * torch.log(rate_e.clamp(min=1e-8))       # (D, N)
        if linear_acc is not None:
            linear_acc += rate_e                                # (D, N)
        if exp_F_list is not None:
            exp_F_list.append(exp_F_e)                         # (L, N)

    terms['term1_mc'] = log_acc.sum() / E    # scalar
    if mode == 'simple':
        terms['term2_mc'] = linear_acc.sum() / E  # scalar

    if return_samples:
        terms['exp_F_samples'] = torch.stack(exp_F_list)   # (E, L, N)

    return terms


# =============================================================================
# Expected log-likelihood functions (modes)
# =============================================================================


def expected_log_likelihood_simple(terms, X):
    """
    Compute expected log-likelihood using full Monte Carlo estimation.

    E[log p(X|rate)] = (1/E) Σ_e Σ_ij [X_ij * log(rate_e_ij) - rate_e_ij] - Σ_ij log(X_ij!)

    All three Poisson log-PMF terms are estimated via MC samples.

    Args:
        terms: dict from compute_log_likelihood_terms().
               Required keys: 'term1_mc', 'term2_mc'.
        X: Input data tensor of shape (D, N)

    Returns:
        Expected log-likelihood E[log p(Y|F)] (scalar)
    """
    return terms['term1_mc'] - terms['term2_mc'] - torch.lgamma(X + 1).sum()


def expected_log_likelihood_expanded(terms, X):
    """
    Compute expected log-likelihood using the expanded expectation form.

    Term 1 (MC):       (1/E) Σ_e Σ_ij X_ij * log(rate_e_ij)
    Term 2 (analytic):  Σ_ij [W @ exp(μ + σ²/2)]_ij
    Term 3 (constant):  -Σ_ij log(X_ij!)

    Args:
        terms: dict from compute_log_likelihood_terms().
               Required keys: 'term1_mc', 'rate_mgf'.
        X: Input data tensor of shape (D, N)

    Returns:
        Expected log-likelihood E[log p(Y|F)] (scalar)
    """
    term1 = terms['term1_mc']               # MC scalar
    term2 = terms['rate_mgf'].sum()          # analytic scalar
    term3 = torch.lgamma(X + 1).sum()        # constant
    return term1 - term2 - term3


def expected_log_likelihood_lower_bound(terms, X):
    """
    Compute expected log-likelihood using Jensen's lower bound (fully analytic).

    Term 1: Σ_ij X_ij * log([W @ exp(μ)]_ij)
    Term 2: Σ_ij [W @ exp(μ + σ²/2)]_ij
    Term 3: -Σ_ij log(X_ij!)

    Args:
        terms: dict from compute_log_likelihood_terms()
        X: Input data tensor of shape (D, N)

    Returns:
        Lower bound on expected log-likelihood E[log p(Y|F)] (scalar)
    """
    rate_mean = terms['rate_mean'].clamp(min=1e-8)    # (D, N)
    term1 = (X * torch.log(rate_mean)).sum()          # scalar
    term2 = terms['rate_mgf'].sum()                   # scalar
    term3 = torch.lgamma(X + 1).sum()                 # scalar
    return term1 - term2 - term3


def expected_log_likelihood(mode, terms, X):
    """
    Compute expected log-likelihood E[log p(Y|F)].

    This function dispatches to the appropriate computation based on mode:
    - 'simple': Full Monte Carlo (X * log(rate) - rate - log(X!))
    - 'expanded': Hybrid Monte Carlo + analytic expectation
    - 'lower-bound': Jensen's lower bound (fully analytic, no MC)

    Args:
        mode: Computation mode ('simple', 'expanded', or 'lower-bound')
        terms: dict from compute_log_likelihood_terms()
        X: Input data tensor of shape (D, N)

    Returns:
        Expected log-likelihood E[log p(Y|F)] (scalar)
    """
    if mode == 'simple':
        return expected_log_likelihood_simple(terms, X)
    elif mode == 'lower-bound':
        return expected_log_likelihood_lower_bound(terms, X)
    else:  # 'expanded'
        return expected_log_likelihood_expanded(terms, X)


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


def compute_elbo(mode, terms, qF, pF, X, kl_fn=None):
    """
    Compute expected log-likelihood and KL divergence separately.

    ELBO = E[log p(X|F)] - KL[q(F) || p(F)]

    Returns the two terms as separate tensors so the caller can scale them
    independently in the training loop. This is critical because:
    - The log-likelihood term scales by D/y_batch_size and N/batch_size
    - The KL term scales by N/batch_size only when KL is over batched q(F)
    - For global KL sources (e.g., SVGP inducing points), KL should NOT be scaled

    Args:
        mode: Expected log-likelihood mode ('simple', 'expanded', or 'lower-bound')
        terms: dict from compute_log_likelihood_terms()
        qF: Variational posterior distribution with mean and scale
        pF: Prior distribution
        X: Input data tensor of shape (D, N)
        kl_fn: Optional custom KL divergence function. If None, uses standard
               torch.distributions.kl_divergence. Should take (qF, pF) and return scalar.

    Returns:
        (exp_ll, kl): tuple of two scalars
            exp_ll: E_q[log p(X | F)]
            kl:     KL[q(F) || p(F)]
    """
    exp_ll = expected_log_likelihood(mode, terms, X)

    if kl_fn is not None:
        kl = kl_fn(qF, pF)
    else:
        kl = kl_divergence(qF, pF)

    return exp_ll, kl
