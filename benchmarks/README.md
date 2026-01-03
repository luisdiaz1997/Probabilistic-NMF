# PNMF Benchmarks

This folder contains benchmarks comparing different aspects of the PNMF implementation.

## Benchmark: Simple vs Expanded ELBO Modes

The main benchmark compares the two ELBO computation modes:

### `mode='simple'`
Uses `torch.distributions.Poisson.log_prob()` directly with Monte Carlo estimation.

### `mode='expanded'` (default)
Uses a hybrid approach:
- First term: Monte Carlo estimation for `Y * E[log(rate)]`
- Second term: Analytic computation for `E[exp(F)]` using Gaussian moment-generating function
- Third term: Poisson normalization constant `log(Y!)`

### Key Differences

| Aspect | Simple | Expanded |
|--------|--------|----------|
| Variance | Higher (full MC) | Lower (hybrid) |
| Computation | Direct PyTorch call | Custom implementation |
| Convergence | May be slower | Often faster |

## Running the Benchmarks

```bash
# From the project root
python benchmarks/simple_vs_expanded.py
```

Or use the Jupyter notebook:
```bash
jupyter notebook benchmarks/simple_vs_expanded.ipynb
```
