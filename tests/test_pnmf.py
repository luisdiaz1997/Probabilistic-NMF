"""
Tests for PNMF package.

Run with: python -m pytest tests/test_pnmf.py -v
Or simply: python tests/test_pnmf.py
"""

import numpy as np
import torch
import pytest

from PNMF import PNMF, PoissonFactorization, GaussianPrior
from PNMF import compute_elbo, expected_log_likelihood, kl_divergence


# =============================================================================
# Test fixtures
# =============================================================================


def generate_count_data(n_samples=50, n_features=30, seed=42):
    """Generate integer count data appropriate for Poisson model."""
    np.random.seed(seed)
    # Generate counts via Poisson sampling
    rate = np.random.rand(n_samples, n_features) * 10 + 1
    X = np.random.poisson(rate).astype(np.float32)
    return X


def generate_small_count_data(seed=42):
    """Generate small integer count data for quick tests."""
    return generate_count_data(n_samples=20, n_features=15, seed=seed)


# =============================================================================
# Test PNMF sklearn API
# =============================================================================


class TestPNMFBasic:
    """Basic tests for PNMF sklearn-like API."""

    def test_fit_transform_shapes(self):
        """Test that fit_transform returns correct shapes."""
        X = generate_small_count_data()
        n_samples, n_features = X.shape
        n_components = 5

        model = PNMF(n_components=n_components, max_iter=10)
        transformed = model.fit_transform(X)

        assert transformed.shape == (n_samples, n_components)
        assert model.components_.shape == (n_components, n_features)

    def test_fit_then_transform(self):
        """Test fit and transform separately."""
        X = generate_small_count_data()
        n_components = 5

        model = PNMF(n_components=n_components, max_iter=10)
        model.fit(X)

        transformed = model.transform(X)
        assert transformed.shape == (X.shape[0], n_components)

    def test_inverse_transform(self):
        """Test inverse_transform reconstructs data."""
        X = generate_small_count_data()
        n_components = 5

        model = PNMF(n_components=n_components, max_iter=10)
        transformed = model.fit_transform(X)
        reconstructed = model.inverse_transform(transformed)

        assert reconstructed.shape == X.shape

    def test_elbo_improves(self):
        """Test that ELBO improves during training."""
        X = generate_small_count_data()

        model = PNMF(n_components=5, max_iter=50)
        history, _ = model.fit(X, return_history=True)

        # ELBO should generally improve (allow some noise)
        assert history[-1] > history[0], "ELBO should improve during training"

    def test_reproducibility(self):
        """Test that random_state gives reproducible results."""
        X = generate_small_count_data()

        model1 = PNMF(n_components=5, max_iter=20, random_state=42)
        model1.fit(X)

        model2 = PNMF(n_components=5, max_iter=20, random_state=42)
        model2.fit(X)

        np.testing.assert_array_almost_equal(
            model1.components_, model2.components_, decimal=5
        )


# =============================================================================
# Test ELBO modes
# =============================================================================


class TestELBOModes:
    """Test different ELBO computation modes."""

    def test_mode_simple(self):
        """Test simple mode runs without error."""
        X = generate_small_count_data()
        model = PNMF(n_components=5, mode='simple', max_iter=10)
        model.fit(X)
        assert model.elbo_ is not None

    def test_mode_expanded(self):
        """Test expanded mode runs without error."""
        X = generate_small_count_data()
        model = PNMF(n_components=5, mode='expanded', max_iter=10)
        model.fit(X)
        assert model.elbo_ is not None

    def test_mode_lower_bound(self):
        """Test lower-bound mode runs without error."""
        X = generate_small_count_data()
        model = PNMF(n_components=5, mode='lower-bound', max_iter=10)
        model.fit(X)
        assert model.elbo_ is not None

    def test_all_modes_comparable(self):
        """Test that all modes produce comparable results."""
        X = generate_small_count_data()

        elbos = {}
        for mode in ['simple', 'expanded', 'lower-bound']:
            model = PNMF(n_components=5, mode=mode, max_iter=50, random_state=42)
            model.fit(X)
            elbos[mode] = model.elbo_

        # All modes should produce negative ELBO values (since they're log-likelihoods)
        for mode, elbo in elbos.items():
            assert elbo < 0, f"{mode} mode should produce negative ELBO"

        # Lower-bound should be <= expanded (it's a lower bound)
        # Allow some tolerance due to stochasticity
        assert elbos['lower-bound'] <= elbos['expanded'] + 100


# =============================================================================
# Test training modes
# =============================================================================


class TestTrainingModes:
    """Test different training modes."""

    def test_standard_training(self):
        """Test standard training mode."""
        X = generate_small_count_data()
        model = PNMF(n_components=5, training_mode='standard', max_iter=10)
        model.fit(X)
        assert model.elbo_ is not None

    def test_natural_training(self):
        """Test natural gradient training mode."""
        X = generate_small_count_data()
        model = PNMF(n_components=5, training_mode='natural', max_iter=10)
        model.fit(X)
        assert model.elbo_ is not None


# =============================================================================
# Test ELBO functions directly
# =============================================================================


class TestELBOFunctions:
    """Test ELBO computation functions directly."""

    def setup_method(self):
        """Set up test fixtures."""
        self.X = generate_small_count_data()
        self.X_torch = torch.from_numpy(self.X.T.astype(np.float32))
        self.L = 5
        self.E = 3

        self.prior = GaussianPrior(self.X_torch, L=self.L)
        self.model = PoissonFactorization(self.prior, self.X_torch, L=self.L)
        self.rate, self.qF, self.pF = self.model(E=self.E)
        self.W = self.model.W.data

    def test_expected_log_likelihood_simple(self):
        """Test simple expected log-likelihood computation."""
        from PNMF.elbo import expected_log_likelihood_simple

        result = expected_log_likelihood_simple(self.rate, self.X_torch)
        assert torch.isfinite(result), "Expected log-likelihood should be finite"
        assert result < 0, "Expected log-likelihood should be negative"

    def test_expected_log_likelihood_expanded(self):
        """Test expanded expected log-likelihood computation."""
        from PNMF.elbo import expected_log_likelihood_expanded

        result = expected_log_likelihood_expanded(
            self.rate, self.qF, self.X_torch, self.W
        )
        assert torch.isfinite(result), "Expected log-likelihood should be finite"
        assert result < 0, "Expected log-likelihood should be negative"

    def test_expected_log_likelihood_lower_bound(self):
        """Test lower-bound expected log-likelihood computation."""
        from PNMF.elbo import expected_log_likelihood_lower_bound

        result = expected_log_likelihood_lower_bound(self.qF, self.X_torch, self.W)
        assert torch.isfinite(result), "Expected log-likelihood should be finite"
        assert result < 0, "Expected log-likelihood should be negative"

    def test_kl_divergence(self):
        """Test KL divergence computation."""
        kl = kl_divergence(self.qF, self.pF)
        assert torch.isfinite(kl), "KL should be finite"
        assert kl >= 0, "KL divergence should be non-negative"

    def test_compute_elbo_dispatcher(self):
        """Test ELBO dispatcher with all modes."""
        for mode in ['simple', 'expanded', 'lower-bound']:
            loss = compute_elbo(
                mode, self.rate, self.qF, self.pF, self.X_torch, self.W
            )
            assert torch.isfinite(loss), f"ELBO should be finite for mode={mode}"

    def test_custom_kl_function(self):
        """Test ELBO with custom KL function."""

        def custom_kl(q, p):
            # Just return standard KL for testing
            return torch.distributions.kl_divergence(q, p).sum()

        loss_standard = compute_elbo(
            'expanded', self.rate, self.qF, self.pF, self.X_torch, self.W
        )
        loss_custom = compute_elbo(
            'expanded', self.rate, self.qF, self.pF, self.X_torch, self.W,
            kl_fn=custom_kl
        )

        torch.testing.assert_close(loss_standard, loss_custom)


# =============================================================================
# Test PyTorch API
# =============================================================================


class TestPyTorchAPI:
    """Test PyTorch-native API."""

    def test_poisson_factorization_forward(self):
        """Test PoissonFactorization forward pass."""
        X = generate_small_count_data()
        X_torch = torch.from_numpy(X.T.astype(np.float32))
        L = 5
        E = 3

        prior = GaussianPrior(X_torch, L=L)
        model = PoissonFactorization(prior, X_torch, L=L)
        rate, qF, pF = model(E=E)

        D, N = X_torch.shape
        assert rate.shape == (E, D, N)
        assert qF.mean.shape == (L, N)
        assert qF.scale.shape == (L, N)

    def test_gaussian_prior_natural_gradients(self):
        """Test GaussianPrior with natural gradient parameterization."""
        X = generate_small_count_data()
        X_torch = torch.from_numpy(X.T.astype(np.float32))
        L = 5

        prior = GaussianPrior(X_torch, L=L, use_natural_gradients=True)
        qF, pF = prior()

        assert qF.mean.shape == (L, X_torch.shape[1])
        assert qF.scale.shape == (L, X_torch.shape[1])


# =============================================================================
# Test batched training
# =============================================================================


class TestBatchedPNMF:
    """Test batched training functionality."""

    def test_batch_size_runs(self):
        """Test that batch_size parameter works."""
        X = generate_small_count_data()
        model = PNMF(n_components=5, batch_size=10, max_iter=20)
        model.fit(X)
        assert model.elbo_ is not None
        assert model.components_.shape == (5, X.shape[1])

    def test_y_batch_size_runs(self):
        """Test that y_batch_size parameter works."""
        X = generate_small_count_data()
        model = PNMF(n_components=5, y_batch_size=8, max_iter=20)
        model.fit(X)
        assert model.elbo_ is not None
        assert model.components_.shape == (5, X.shape[1])

    def test_dual_batching(self):
        """Test both batch_size and y_batch_size together."""
        X = generate_small_count_data()
        model = PNMF(n_components=5, batch_size=10, y_batch_size=8, max_iter=20)
        model.fit(X)
        assert model.elbo_ is not None
        assert model.components_.shape == (5, X.shape[1])

    def test_batched_elbo_improves(self):
        """Test that ELBO improves during batched training."""
        X = generate_small_count_data()

        model = PNMF(n_components=5, batch_size=10, max_iter=50)
        history, _ = model.fit(X, return_history=True)

        # ELBO should generally improve (allow some noise for batched)
        # Check if final ELBO is better than initial
        assert history[-1] > history[0], "ELBO should improve during batched training"

    def test_batched_vs_full_comparable(self):
        """Test that batched and full-batch produce similar results."""
        X = generate_count_data(n_samples=50, n_features=30, seed=42)

        # Full batch training
        model_full = PNMF(n_components=5, max_iter=100, random_state=42)
        model_full.fit(X)

        # Batched training (larger batch for more stability)
        model_batched = PNMF(n_components=5, batch_size=25, max_iter=100, random_state=42)
        model_batched.fit(X)

        # Both should produce valid ELBO values (negative)
        assert model_full.elbo_ < 0
        assert model_batched.elbo_ < 0

        # Reconstruction quality should be similar (within tolerance)
        transformed_full = model_full.transform(X)
        transformed_batched = model_batched.transform(X)
        recon_full = model_full.inverse_transform(transformed_full)
        recon_batched = model_batched.inverse_transform(transformed_batched)

        error_full = np.mean((X - recon_full) ** 2)
        error_batched = np.mean((X - recon_batched) ** 2)

        # Batched error should be within 2x of full batch error
        assert error_batched < error_full * 2, "Batched should have similar reconstruction"

    def test_shuffle_parameter(self):
        """Test that shuffle parameter works."""
        X = generate_small_count_data()
        model = PNMF(n_components=5, batch_size=10, shuffle=True, max_iter=10)
        model.fit(X)
        assert model.elbo_ is not None

        model_no_shuffle = PNMF(n_components=5, batch_size=10, shuffle=False, max_iter=10)
        model_no_shuffle.fit(X)
        assert model_no_shuffle.elbo_ is not None

    def test_batched_with_modes(self):
        """Test batched training with different ELBO modes."""
        X = generate_small_count_data()

        for mode in ['simple', 'expanded', 'lower-bound']:
            model = PNMF(n_components=5, mode=mode, batch_size=10, max_iter=20)
            model.fit(X)
            assert model.elbo_ is not None, f"Batched {mode} mode failed"

    def test_batched_with_natural_gradients(self):
        """Test batched training with natural gradient mode."""
        X = generate_small_count_data()
        model = PNMF(
            n_components=5,
            training_mode='natural',
            batch_size=10,
            max_iter=20
        )
        model.fit(X)
        assert model.elbo_ is not None

    def test_batch_size_larger_than_data(self):
        """Test that batch_size larger than data works (clamped to N)."""
        X = generate_small_count_data()  # 20 samples
        model = PNMF(n_components=5, batch_size=100, max_iter=10)  # batch > samples
        model.fit(X)
        assert model.elbo_ is not None


# =============================================================================
# Test parameter validation
# =============================================================================


class TestParameterValidation:
    """Test parameter validation."""

    def test_invalid_n_components(self):
        """Test that invalid n_components raises error."""
        with pytest.raises(ValueError):
            model = PNMF(n_components=0)
            model.fit(generate_small_count_data())

    def test_invalid_mode(self):
        """Test that invalid mode raises error."""
        with pytest.raises(ValueError):
            model = PNMF(mode='invalid')
            model.fit(generate_small_count_data())

    def test_invalid_training_mode(self):
        """Test that invalid training_mode raises error."""
        with pytest.raises(ValueError):
            model = PNMF(training_mode='invalid')
            model.fit(generate_small_count_data())

    def test_invalid_batch_size(self):
        """Test that invalid batch_size raises error."""
        with pytest.raises(ValueError):
            model = PNMF(batch_size=0)
            model.fit(generate_small_count_data())

    def test_invalid_y_batch_size(self):
        """Test that invalid y_batch_size raises error."""
        with pytest.raises(ValueError):
            model = PNMF(y_batch_size=-1)
            model.fit(generate_small_count_data())


# =============================================================================
# Run tests
# =============================================================================


def run_all_tests():
    """Run all tests without pytest."""
    test_classes = [
        TestPNMFBasic,
        TestELBOModes,
        TestTrainingModes,
        TestELBOFunctions,
        TestPyTorchAPI,
        TestBatchedPNMF,
        TestParameterValidation,
    ]

    passed = 0
    failed = 0
    errors = []

    for test_class in test_classes:
        print(f"\n{'='*60}")
        print(f"Running {test_class.__name__}")
        print('='*60)

        instance = test_class()

        # Run setup if exists
        if hasattr(instance, 'setup_method'):
            try:
                instance.setup_method()
            except Exception as e:
                print(f"  Setup failed: {e}")
                continue

        # Get all test methods
        test_methods = [m for m in dir(instance) if m.startswith('test_')]

        for method_name in test_methods:
            # Re-run setup for each test
            if hasattr(instance, 'setup_method'):
                try:
                    instance.setup_method()
                except Exception as e:
                    print(f"  {method_name}: SETUP ERROR - {e}")
                    failed += 1
                    errors.append((test_class.__name__, method_name, str(e)))
                    continue

            try:
                method = getattr(instance, method_name)
                method()
                print(f"  {method_name}: PASSED")
                passed += 1
            except AssertionError as e:
                print(f"  {method_name}: FAILED - {e}")
                failed += 1
                errors.append((test_class.__name__, method_name, str(e)))
            except Exception as e:
                print(f"  {method_name}: ERROR - {e}")
                failed += 1
                errors.append((test_class.__name__, method_name, str(e)))

    print(f"\n{'='*60}")
    print(f"SUMMARY: {passed} passed, {failed} failed")
    print('='*60)

    if errors:
        print("\nFailed tests:")
        for cls, method, error in errors:
            print(f"  {cls}.{method}: {error}")

    return failed == 0


if __name__ == '__main__':
    import sys

    # Try pytest first, fall back to manual runner
    try:
        import pytest
        sys.exit(pytest.main([__file__, '-v']))
    except ImportError:
        success = run_all_tests()
        sys.exit(0 if success else 1)
