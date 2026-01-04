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
    Uses Poisson sampling to generate integer counts.

    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_components: Number of latent components
        random_state: Random seed

    Returns:
        X: Generated data matrix of shape (n_samples, n_features) with integer counts
    """
    rng = np.random.RandomState(random_state)

    # Generate true W (positive)
    W_true = rng.exponential(scale=1.0, size=(n_features, n_components))

    # Generate true F (Gaussian latent factors)
    F_true = rng.randn(n_components, n_samples)

    # Compute rate parameters
    rate = W_true @ np.exp(F_true)

    # Sample from Poisson distribution to get integer counts
    X = rng.poisson(lam=rate)

    return X


def run_benchmark(mode='expanded', n_components=5, max_iter=100, random_state=42, verbose=False, optimizer='Adam', learning_rate=0.01):
    """
    Run PNMF with specified mode and track convergence.

    Args:
        mode: 'simple' or 'expanded'
        n_components: Number of components
        max_iter: Maximum iterations
        random_state: Random seed
        verbose: Whether to print progress
        optimizer: Optimizer to use ('Adam', 'AdamW', 'NAdam', 'SGD', 'RMSprop')
        learning_rate: Learning rate for optimizer

    Returns:
        results: Dictionary with benchmark results
    """
    # Generate synthetic data
    X = generate_synthetic_data(n_samples=200, n_features=100, n_components=n_components, random_state=random_state)

    # Initialize and fit model
    model = PNMF(
        n_components=n_components,
        mode=mode,
        loadings_mode='projected',
        E=3,
        max_iter=max_iter,
        tol=1e-4,
        learning_rate=learning_rate,
        optimizer=optimizer,
        random_state=random_state,
        verbose=verbose
    )

    elbo_history, model = model.fit(X, return_history=True)

    # Compute reconstruction error using exp(qF.mean) @ W.T
    # qF.mean has shape (L, N) = (n_components, n_samples)
    # W has shape (D, L) = (n_features, n_components)
    # Reconstruction: X ≈ exp(F) @ W.T = exp(qF.mean).T @ W
    exp_F_mean = np.exp(model._prior.mean.detach().cpu().numpy().T)  # (N, L)
    W = model.components_.T  # (D, L)
    X_reconstructed = exp_F_mean @ W.T  # (N, D)
    reconstruction_error = np.linalg.norm(X - X_reconstructed, 'fro') / np.linalg.norm(X, 'fro')

    return {
        'mode': mode,
        'optimizer': optimizer,
        'n_iterations': model.n_iter_,
        'final_elbo': model.elbo_,
        'elbo_history': elbo_history,
        'reconstruction_error': reconstruction_error,
        'converged': model.n_iter_ < max_iter
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

    # Convert ELBO to loss (negative ELBO)
    loss_simple = [-x for x in results_simple['elbo_history']]
    loss_expanded = [-x for x in results_expanded['elbo_history']]

    # Plot 1: Loss convergence (log scale)
    ax1 = axes[0]
    ax1.semilogy(loss_simple, label='Simple (torch.Poisson)', linewidth=2)
    ax1.semilogy(loss_expanded, label='Expanded (hybrid)', linewidth=2)
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Loss (-ELBO)', fontsize=12)
    ax1.set_title('Loss Convergence (log scale)', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Loss difference (relative to final)
    ax2 = axes[1]
    final_simple = loss_simple[-1]
    final_expanded = loss_expanded[-1]

    # Plot distance to convergence
    diff_simple = [abs(x - final_simple) for x in loss_simple]
    diff_expanded = [abs(x - final_expanded) for x in loss_expanded]

    ax2.semilogy(diff_simple, label='Simple (torch.Poisson)', linewidth=2)
    ax2.semilogy(diff_expanded, label='Expanded (hybrid)', linewidth=2)
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('|Loss - Final|', fontsize=12)
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
    results_simple = run_benchmark(mode='simple', n_components=5, max_iter=2000, random_state=42, verbose=False, optimizer='Adam')
    print(f"  Completed in {results_simple['n_iterations']} iterations")
    print()

    print("Running expanded mode (hybrid MC + analytic)...")
    results_expanded = run_benchmark(mode='expanded', n_components=5, max_iter=2000, random_state=42, verbose=False, optimizer='Adam')
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
