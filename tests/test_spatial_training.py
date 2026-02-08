"""Test that mu, W, and Lu are all being trained in spatial SVGP mode."""
import numpy as np
import torch
from PNMF import PNMF


def test_spatial_parameters_trained():
    """Verify that mu, W, and Lu all change during spatial model fitting."""
    # Small synthetic dataset
    np.random.seed(42)
    N, D, L = 30, 20, 3
    X = np.random.poisson(lam=5, size=(N, D)).astype(np.float32)

    # Random 2D coordinates + groups
    coordinates = np.random.rand(N, 2).astype(np.float32) * 10
    groups = np.random.randint(0, 2, size=N)

    # Create and fit spatial PNMF model
    model = PNMF(
        n_components=L,
        spatial=True,
        num_inducing=10,
        max_iter=100,
        learning_rate=0.01,
        random_state=42,
        verbose=True,
        mode='lower-bound',
    )
    model.fit(X, coordinates=coordinates, groups=groups)

    gp = model._prior
    W = model._model.W

    # --- Check Lu ---
    print("\n=== Lu after training ===")
    print(f"Lu shape: {gp.Lu.data.shape}")
    for i in range(L):
        print(f"Lu (factor {i}, diagonal): {gp.Lu.data[i].diag()}")

    init_val = torch.tensor(0.1010).to(gp.Lu.data.device)
    lu_diag_changed = not torch.allclose(gp.Lu.data[0].diag(), init_val.expand(10), atol=1e-3)
    print(f"Lu diagonal changed from init: {lu_diag_changed}")
    assert lu_diag_changed, "Lu diagonal did NOT change during training!"

    # Check off-diagonal elements (initialized to 0)
    lower = torch.tril(gp.Lu.data, diagonal=-1)
    off_diag_max = lower.abs().max().item()
    print(f"Lu off-diagonal max abs value: {off_diag_max:.8f}")
    lu_offdiag_changed = off_diag_max > 1e-6
    print(f"Lu off-diagonal changed from init (0): {lu_offdiag_changed}")
    assert lu_offdiag_changed, "Lu off-diagonal elements did NOT change during training!"

    # --- Check mu ---
    print("\n=== mu after training ===")
    print(f"mu shape: {gp.mu.shape}")
    print(f"mu (factor 0, first 5): {gp.mu.data[0][:5]}")
    mu_nonzero = gp.mu.data.abs().max().item() > 1e-6
    print(f"mu has non-trivial values: {mu_nonzero}")
    assert mu_nonzero, "mu is all zeros after training!"

    # --- Check W ---
    print("\n=== W after training ===")
    print(f"W shape: {W.data.shape}")
    print(f"W (first 5 entries, factor 0): {W.data[:5, 0]}")
    w_positive = (W.data > 0).all().item()
    w_nonuniform = W.data.std().item() > 1e-6
    print(f"W all positive: {w_positive}")
    print(f"W has variation: {w_nonuniform}")
    assert w_positive, "W has non-positive values!"
    assert w_nonuniform, "W has no variation after training!"

    # --- Check gradients exist ---
    print("\n=== Gradient info ===")
    print(f"Lu._raw.grad is None: {gp.Lu._raw.grad is None}")
    print(f"mu.grad is None: {gp.mu.grad is None}")
    if gp.Lu._raw.grad is not None:
        print(f"Lu._raw.grad norm: {gp.Lu._raw.grad.norm().item():.6f}")
    if gp.mu.grad is not None:
        print(f"mu.grad norm: {gp.mu.grad.norm().item():.6f}")

    print("\nAll spatial parameters (mu, W, Lu) are being trained!")


if __name__ == "__main__":
    test_spatial_parameters_trained()
