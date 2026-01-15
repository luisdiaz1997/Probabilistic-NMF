"""
Benchmark: Natural Gradient vs Standard Training Mode

This script compares standard gradient descent with natural gradient descent
across all three ELBO computation modes:
- 'simple': Uses torch.distributions.Poisson.log_prob() directly (full Monte Carlo)
- 'expanded': Uses hybrid Monte Carlo + analytic expectation (default)
- 'lower-bound': Uses Jensen's lower bound (fully analytic, no MC sampling)

The benchmark measures:
1. Convergence speed (iterations to convergence)
2. Final ELBO value
3. Per-iteration time
4. Reconstruction error
5. Improvement from natural gradients
"""

import numpy as np
import matplotlib.pyplot as plt
from PNMF import PNMF


def generate_synthetic_data(n_samples=200, n_features=100, n_components=5, random_state=42):
    """
    Generate synthetic non-negative data for benchmarking.

    Creates data that approximately follows the PNMF model:
    Internal: X (D, N) ≈ W (D, L) @ exp(F) (L, N)
    sklearn API: X (N, D) ≈ exp(F).T (N, L) @ W.T (L, D)
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


def run_benchmark(mode='expanded', training_mode='standard', n_components=5,
                  max_iter=100, random_state=42, verbose=False, learning_rate=0.01):
    """
    Run PNMF with specified mode and training mode and track convergence.

    Args:
        mode: 'simple', 'expanded', or 'lower-bound'
        training_mode: 'standard' or 'natural'
        n_components: Number of components
        max_iter: Maximum iterations
        random_state: Random seed
        verbose: Whether to print progress
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
        training_mode=training_mode,
        loadings_mode='projected',
        E=10,
        max_iter=max_iter,
        tol=1e-4,
        learning_rate=learning_rate,
        optimizer='Adam',
        random_state=random_state,
        verbose=verbose
    )

    elbo_history, model = model.fit(X, return_history=True)

    # Compute reconstruction error using exp(qF.mean) @ W.T
    # qF.mean has shape (L, N) = (n_components, n_samples)
    # W has shape (D, L) = (n_features, n_components)
    # Reconstruction: X ≈ exp(F) @ W.T = exp(qF.mean).T @ W
    # Handle both standard and natural gradient modes
    qF, _ = model._prior()
    exp_F_mean = np.exp(qF.mean.detach().cpu().numpy().T)  # (N, L)
    W = model.components_.T  # (D, L)
    X_reconstructed = exp_F_mean @ W.T  # (N, D)
    reconstruction_error = np.linalg.norm(X - X_reconstructed, 'fro') / np.linalg.norm(X, 'fro')

    return {
        'mode': mode,
        'training_mode': training_mode,
        'n_iterations': model.n_iter_,
        'final_elbo': model.elbo_,
        'elbo_history': elbo_history,
        'reconstruction_error': reconstruction_error,
        'converged': model.n_iter_ < max_iter
    }


def plot_results(results_dict, output_path='benchmarks/natural_gradient_comparison.png'):
    """
    Plot convergence comparison between standard and natural gradient modes.

    Args:
        results_dict: Dictionary with all benchmark results
        output_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    modes = ['simple', 'expanded', 'lower-bound']
    mode_labels = ['Simple (MC)', 'Expanded (Hybrid)', 'Lower Bound (Analytic)']

    for idx, (mode, mode_label) in enumerate(zip(modes, mode_labels)):
        # Standard mode results
        results_std = results_dict[f'{mode}_standard']
        results_nat = results_dict[f'{mode}_natural']

        # Convert ELBO to loss (negative ELBO)
        loss_std = [-x for x in results_std['elbo_history']]
        loss_nat = [-x for x in results_nat['elbo_history']]

        # Plot 1: Loss convergence (log-log scale) - Top row
        ax1 = axes[0, idx]
        iterations_std = range(1, len(loss_std) + 1)
        iterations_nat = range(1, len(loss_nat) + 1)
        ax1.loglog(iterations_std, loss_std, label='Standard', linewidth=2, alpha=0.7)
        ax1.loglog(iterations_nat, loss_nat, label='Natural Gradient', linewidth=2, alpha=0.7)
        ax1.set_xlabel('Iteration', fontsize=12)
        ax1.set_ylabel('Loss (-ELBO)', fontsize=12)
        ax1.set_title(f'{mode_label}: Loss Convergence', fontsize=14)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Loss difference (relative to final) - Bottom row
        ax2 = axes[1, idx]
        final_std = loss_std[-1]
        final_nat = loss_nat[-1]

        # Plot distance to convergence
        diff_std = [abs(x - final_std) for x in loss_std]
        diff_nat = [abs(x - final_nat) for x in loss_nat]

        ax2.loglog(iterations_std, diff_std, label='Standard', linewidth=2, alpha=0.7)
        ax2.loglog(iterations_nat, diff_nat, label='Natural Gradient', linewidth=2, alpha=0.7)
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('|Loss - Final|', fontsize=12)
        ax2.set_title(f'{mode_label}: Distance to Convergence', fontsize=14)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_path}")


def plot_elbo_comparison(results_std_simple, results_std_expanded, results_std_lb,
                         results_nat_simple, results_nat_expanded, results_nat_lb,
                         output_path='benchmarks/natural_gradient_elbo_comparison.png'):
    """
    Plot ELBO comparison across all modes and training methods.

    Args:
        results_std_simple: Standard training, simple mode results
        results_std_expanded: Standard training, expanded mode results
        results_std_lb: Standard training, lower-bound mode results
        results_nat_simple: Natural gradient, simple mode results
        results_nat_expanded: Natural gradient, expanded mode results
        results_nat_lb: Natural gradient, lower-bound mode results
        output_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Convert ELBO to loss (negative ELBO)
    loss_std_simple = [-x for x in results_std_simple['elbo_history']]
    loss_std_expanded = [-x for x in results_std_expanded['elbo_history']]
    loss_std_lb = [-x for x in results_std_lb['elbo_history']]
    loss_nat_simple = [-x for x in results_nat_simple['elbo_history']]
    loss_nat_expanded = [-x for x in results_nat_expanded['elbo_history']]
    loss_nat_lb = [-x for x in results_nat_lb['elbo_history']]

    # Plot 1: All standard modes
    ax1 = axes[0]
    iterations_std_simple = range(1, len(loss_std_simple) + 1)
    iterations_std_expanded = range(1, len(loss_std_expanded) + 1)
    iterations_std_lb = range(1, len(loss_std_lb) + 1)
    ax1.loglog(iterations_std_simple, loss_std_simple, label='Simple', linewidth=2, alpha=0.7)
    ax1.loglog(iterations_std_expanded, loss_std_expanded, label='Expanded', linewidth=2, alpha=0.7)
    ax1.loglog(iterations_std_lb, loss_std_lb, label='Lower Bound', linewidth=2, alpha=0.7)
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Loss (-ELBO)', fontsize=12)
    ax1.set_title('Standard Training: All Modes', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: All natural gradient modes
    ax2 = axes[1]
    iterations_nat_simple = range(1, len(loss_nat_simple) + 1)
    iterations_nat_expanded = range(1, len(loss_nat_expanded) + 1)
    iterations_nat_lb = range(1, len(loss_nat_lb) + 1)
    ax2.loglog(iterations_nat_simple, loss_nat_simple, label='Simple', linewidth=2, alpha=0.7)
    ax2.loglog(iterations_nat_expanded, loss_nat_expanded, label='Expanded', linewidth=2, alpha=0.7)
    ax2.loglog(iterations_nat_lb, loss_nat_lb, label='Lower Bound', linewidth=2, alpha=0.7)
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Loss (-ELBO)', fontsize=12)
    ax2.set_title('Natural Gradient: All Modes', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_path}")


def print_summary(results_dict):
    """
    Print a summary of the benchmark results.

    Args:
        results_dict: Dictionary with all benchmark results
    """
    print("=" * 100)
    print("PNMF Benchmark: Natural Gradient vs Standard Training Mode")
    print("=" * 100)
    print()

    modes = ['simple', 'expanded', 'lower-bound']
    mode_labels = ['Simple (MC)', 'Expanded (Hybrid)', 'Lower Bound (Analytic)']

    for mode, mode_label in zip(modes, mode_labels):
        print(f"{mode_label}")
        print("-" * 100)

        results_std = results_dict[f'{mode}_standard']
        results_nat = results_dict[f'{mode}_natural']

        # Calculate improvement
        elbo_improvement = ((results_nat['final_elbo'] - results_std['final_elbo']) /
                            abs(results_std['final_elbo']) * 100)
        recon_improvement = ((results_std['reconstruction_error'] - results_nat['reconstruction_error']) /
                             results_std['reconstruction_error'] * 100)

        print(f"  Training Mode           Standard          Natural Gradient    Improvement")
        print(f"  -------------------     ----------------  ----------------    -----------")

        # Iterations
        print(f"  Iterations              {results_std['n_iterations']:<18}  "
              f"{results_nat['n_iterations']:<18}  "
              f"{results_nat['n_iterations']/results_std['n_iterations']:.2f}x")

        # Final ELBO
        print(f"  Final ELBO              {results_std['final_elbo']:<18.4f}  "
              f"{results_nat['final_elbo']:<18.4f}  "
              f"+{elbo_improvement:.2f}%")

        # Reconstruction error
        print(f"  Recon Error             {results_std['reconstruction_error']:<18.6f}  "
              f"{results_nat['reconstruction_error']:<18.6f}  "
              f"+{recon_improvement:.2f}%")
        print()

    # Overall winner
    print("=" * 100)
    print("Overall Analysis")
    print("=" * 100)

    # Find best combination
    all_results = list(results_dict.values())
    best_elbo_idx = np.argmax([r['final_elbo'] for r in all_results])
    best_elbo_combo = all_results[best_elbo_idx]
    print(f"Best ELBO: {best_elbo_combo['training_mode'].title()} + {best_elbo_combo['mode'].title()} "
          f"(ELBO = {best_elbo_combo['final_elbo']:.4f})")

    best_recon_idx = np.argmin([r['reconstruction_error'] for r in all_results])
    best_recon_combo = all_results[best_recon_idx]
    print(f"Best Reconstruction: {best_recon_combo['training_mode'].title()} + {best_recon_combo['mode'].title()} "
          f"(Error = {best_recon_combo['reconstruction_error']:.6f})")

    fastest_idx = np.argmin([r['n_iterations'] for r in all_results])
    fastest_combo = all_results[fastest_idx]
    print(f"Fastest Convergence: {fastest_combo['training_mode'].title()} + {fastest_combo['mode'].title()} "
          f"({fastest_combo['n_iterations']} iterations)")

    # Average improvement from natural gradients
    elbo_improvements = []
    for mode in modes:
        std = results_dict[f'{mode}_standard']['final_elbo']
        nat = results_dict[f'{mode}_natural']['final_elbo']
        elbo_improvements.append((nat - std) / abs(std) * 100)

    print(f"\nAverage ELBO improvement from natural gradients: +{np.mean(elbo_improvements):.2f}%")

    print("=" * 100)


def main():
    """Main benchmark function."""
    print("Running PNMF benchmark: Natural Gradient vs Standard Training Mode...")
    print()

    # Set random seed for reproducibility
    np.random.seed(42)

    results_dict = {}

    # Run benchmarks for all combinations
    for mode in ['simple', 'expanded', 'lower-bound']:
        for training_mode in ['standard', 'natural']:
            key = f'{mode}_{training_mode}'
            print(f"Running {mode} mode with {training_mode} training...")
            results = run_benchmark(
                mode=mode,
                training_mode=training_mode,
                n_components=5,
                max_iter=4000,
                random_state=42,
                verbose=False,
                learning_rate=0.005
            )
            results_dict[key] = results
            print(f"  Completed in {results['n_iterations']} iterations, "
                  f"ELBO = {results['final_elbo']:.4f}")
            print()

    # Print summary
    print_summary(results_dict)

    # Plot results
    print()
    print("Generating plots...")
    plot_results(results_dict)

    plot_elbo_comparison(
        results_dict['simple_standard'],
        results_dict['expanded_standard'],
        results_dict['lower-bound_standard'],
        results_dict['simple_natural'],
        results_dict['expanded_natural'],
        results_dict['lower-bound_natural']
    )

    print()
    print("Benchmark complete!")


if __name__ == '__main__':
    main()
