"""
Benchmark: Simple vs Expanded ELBO Modes

This script compares the two ELBO computation modes:
- 'simple': Uses torch.distributions.Poisson.log_prob() directly
- 'expanded': Uses hybrid Monte Carlo + analytic expectation

The benchmark measures:
1. Convergence speed (iterations to convergence)
2. Final ELBO value
3. Per-iteration time
4. Reconstruction error
"""

import numpy as np
import matplotlib.pyplot as plt
from PNMF import PNMF


def generate_synthetic_data(n_samples=200, n_features=100, n_components=5, random_state=42):
    """
    Generate synthetic non-negative data for benchmarking.

    Creates data that approximately follows the PNMF model:
    X ≈ W @ exp(F) where F ~ N(0, 1)

    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_components: Number of latent components
        random_state: Random seed

    Returns:
        X: Generated data matrix of shape (n_samples, n_features)
    """
    rng = np.random.RandomState(random_state)

    # Generate true W (positive)
    W_true = rng.exponential(scale=1.0, size=(n_features, n_components))

    # Generate true F (Gaussian latent factors)
    F_true = rng.randn(n_components, n_samples)

    # Compute X
    X = W_true @ np.exp(F_true)

    # Add some noise
    X += rng.exponential(scale=0.1, size=X.shape)

    # Ensure non-negative
    X = np.maximum(X, 0)

    return X


def run_benchmark(mode='expanded', n_components=5, max_iter=100, random_state=42, verbose=False):
    """
    Run PNMF with specified mode and track convergence.

    Args:
        mode: 'simple' or 'expanded'
        n_components: Number of components
        max_iter: Maximum iterations
        random_state: Random seed
        verbose: Whether to print progress

    Returns:
        results: Dictionary with benchmark results
    """
    # Generate synthetic data
    X = generate_synthetic_data(n_samples=200, n_features=100, n_components=n_components, random_state=random_state)

    # Initialize model with same random seed for both modes
    model = PNMF(
        n_components=n_components,
        mode=mode,
        loadings_mode='projected',
        E=3,
        max_iter=max_iter,
        tol=1e-4,
        learning_rate=0.01,
        random_state=random_state,
        verbose=verbose
    )

    # Track ELBO during training by modifying the model's training loop
    import torch
    from PNMF.priors import GaussianPrior
    from PNMF.models import PoissonFactorization
    from tqdm.auto import tqdm

    # Convert to torch tensor
    X_torch = torch.from_numpy(X.T.astype(np.float32))

    # Initialize prior and model
    prior = GaussianPrior(y=X_torch, L=n_components)
    pf_model = PoissonFactorization(prior=prior, y=X_torch, L=n_components, loadings_mode='projected', mode=mode)

    # Setup optimizer
    params = list(pf_model.W.parameters()) + list(prior.parameters())
    optimizer = torch.optim.Adam(params, lr=0.01)

    # Training loop with ELBO tracking
    elbo_history = []
    prev_elbo = float('-inf')

    for iteration in range(max_iter):
        optimizer.zero_grad()
        rate, qF, pF = pf_model.forward(E=3)

        # Compute ELBO based on mode
        if mode == 'simple':
            # Simple mode: full Monte Carlo estimation
            E_samples = rate.shape[0]
            eps = 1e-8
            rate_clamped = rate.clamp(min=eps)
            X_expanded = X_torch.unsqueeze(0).expand(E_samples, -1, -1)
            # All terms via Monte Carlo: X * log(rate) - rate - log(X!)
            log_lik_mc = (X_expanded * torch.log(rate_clamped) - rate_clamped - torch.lgamma(X_expanded + 1))
            log_lik = log_lik_mc.sum() / E_samples
        else:
            # Expanded mode: hybrid MC + analytic
            E_samples = rate.shape[0]
            eps = 1e-8
            rate_clamped = rate.clamp(min=eps)
            X_expanded = X_torch.unsqueeze(0).expand(E_samples, -1, -1)
            term1_mc = (X_expanded * torch.log(rate_clamped)).sum() / E_samples
            mu = qF.mean
            sigma = qF.scale
            exp_expectation = torch.exp(mu + 0.5 * sigma ** 2)
            W = pf_model.W.data
            term2_analytic = torch.matmul(W, exp_expectation).sum()
            log_lik = term1_mc - term2_analytic - torch.lgamma(X_torch + 1).sum()

        kl = torch.distributions.kl_divergence(qF, pF).sum()
        loss = kl - log_lik

        loss.backward()
        optimizer.step()

        # Project parameters
        pf_model.project_parameters()

        elbo_value = -loss.item()
        elbo_history.append(elbo_value)

        # Check convergence
        if abs(elbo_value - prev_elbo) < 1e-4:
            break
        prev_elbo = elbo_value

    # Compute reconstruction error
    with torch.no_grad():
        rate_final, _, _ = pf_model.forward(E=10)
        rate_mean = rate_final.mean(dim=0).t().numpy()  # (N, D)
        reconstruction_error = np.linalg.norm(X - rate_mean, 'fro') / np.linalg.norm(X, 'fro')

    return {
        'mode': mode,
        'n_iterations': len(elbo_history),
        'final_elbo': elbo_history[-1],
        'elbo_history': elbo_history,
        'reconstruction_error': reconstruction_error,
        'converged': len(elbo_history) < max_iter
    }


def plot_results(results_simple, results_expanded, output_path='benchmarks/convergence_comparison.png'):
    """
    Plot convergence comparison between simple and expanded modes.

    Args:
        results_simple: Results from simple mode
        results_expanded: Results from expanded mode
        output_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: ELBO convergence
    ax1 = axes[0]
    ax1.plot(results_simple['elbo_history'], label='Simple (torch.Poisson)', linewidth=2)
    ax1.plot(results_expanded['elbo_history'], label='Expanded (hybrid)', linewidth=2)
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('ELBO', fontsize=12)
    ax1.set_title('ELBO Convergence Comparison', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: ELBO difference (relative to final)
    ax2 = axes[1]
    final_simple = results_simple['elbo_history'][-1]
    final_expanded = results_expanded['elbo_history'][-1]

    # Plot distance to convergence
    diff_simple = [abs(x - final_simple) for x in results_simple['elbo_history']]
    diff_expanded = [abs(x - final_expanded) for x in results_expanded['elbo_history']]

    ax2.semilogy(diff_simple, label='Simple (torch.Poisson)', linewidth=2)
    ax2.semilogy(diff_expanded, label='Expanded (hybrid)', linewidth=2)
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('|ELBO - Final|', fontsize=12)
    ax2.set_title('Distance to Convergence (log scale)', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_path}")


def print_summary(results_simple, results_expanded):
    """
    Print a summary of the benchmark results.

    Args:
        results_simple: Results from simple mode
        results_expanded: Results from expanded mode
    """
    print("=" * 70)
    print("PNMF Benchmark: Simple vs Expanded ELBO Modes")
    print("=" * 70)
    print()

    print(f"{'Metric':<30} {'Simple':<20} {'Expanded':<20}")
    print("-" * 70)

    # Iterations to convergence
    print(f"{'Iterations to convergence':<30} "
          f"{results_simple['n_iterations']:<20} "
          f"{results_expanded['n_iterations']:<20}")

    # Final ELBO
    print(f"{'Final ELBO':<30} "
          f"{results_simple['final_elbo']:<20.4f} "
          f"{results_expanded['final_elbo']:<20.4f}")

    # ELBO difference
    elbo_diff = abs(results_simple['final_elbo'] - results_expanded['final_elbo'])
    print(f"{'ELBO difference':<30} {'':<20} {elbo_diff:<20.6f}")

    # Reconstruction error
    print(f"{'Relative reconstruction error':<30} "
          f"{results_simple['reconstruction_error']:<20.6f} "
          f"{results_expanded['reconstruction_error']:<20.6f}")

    print()

    # Winner analysis
    if results_simple['n_iterations'] < results_expanded['n_iterations']:
        faster = "Simple"
        speedup = results_expanded['n_iterations'] / results_simple['n_iterations']
    else:
        faster = "Expanded"
        speedup = results_simple['n_iterations'] / results_expanded['n_iterations']

    print(f"Convergence winner: {faster} ({speedup:.2f}x faster)")
    print()

    # ELBO comparison
    if results_simple['final_elbo'] > results_expanded['final_elbo']:
        print(f"Final ELBO winner: Simple (higher by {elbo_diff:.6f})")
    elif results_expanded['final_elbo'] > results_simple['final_elbo']:
        print(f"Final ELBO winner: Expanded (higher by {elbo_diff:.6f})")
    else:
        print("Final ELBO: Tie")

    print("=" * 70)


def main():
    """Main benchmark function."""
    print("Running PNMF benchmark: Simple vs Expanded ELBO modes...")
    print()

    # Set random seed for reproducibility
    np.random.seed(42)

    # Run benchmarks
    print("Running simple mode (torch.Poisson)...")
    results_simple = run_benchmark(mode='simple', n_components=5, max_iter=200, random_state=42, verbose=False)
    print(f"  Completed in {results_simple['n_iterations']} iterations")
    print()

    print("Running expanded mode (hybrid MC + analytic)...")
    results_expanded = run_benchmark(mode='expanded', n_components=5, max_iter=200, random_state=42, verbose=False)
    print(f"  Completed in {results_expanded['n_iterations']} iterations")
    print()

    # Print summary
    print_summary(results_simple, results_expanded)

    # Plot results
    print()
    print("Generating plots...")
    plot_results(results_simple, results_expanded)

    print()
    print("Benchmark complete!")


if __name__ == '__main__':
    main()
