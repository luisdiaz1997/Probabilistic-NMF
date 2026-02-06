"""
Tests for PNMF package.

Run with: python -m pytest tests/test_pnmf.py -v
Or simply: python tests/test_pnmf.py
"""

import numpy as np
import torch
import pytest

from PNMF import PNMF, PoissonFactorization, GaussianPrior
from PNMF import compute_elbo, compute_log_likelihood_terms, expected_log_likelihood, kl_divergence


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
        self.qF, self.pF = self.prior()
        self.W = self.model.W.data

        # Compute terms for different modes
        self.terms_expanded = compute_log_likelihood_terms(
            self.W, self.qF, self.X_torch, self.E, mode='expanded'        )
        self.terms_simple = compute_log_likelihood_terms(
            self.W, self.qF, self.X_torch, self.E, mode='simple'        )
        self.terms_lb = compute_log_likelihood_terms(
            self.W, self.qF, self.X_torch, self.E, mode='lower-bound'
        )

    def test_expected_log_likelihood_simple(self):
        """Test simple expected log-likelihood computation."""
        from PNMF.elbo import expected_log_likelihood_simple

        result = expected_log_likelihood_simple(self.terms_simple, self.X_torch)
        assert torch.isfinite(result), "Expected log-likelihood should be finite"
        assert result < 0, "Expected log-likelihood should be negative"

    def test_expected_log_likelihood_expanded(self):
        """Test expanded expected log-likelihood computation."""
        from PNMF.elbo import expected_log_likelihood_expanded

        result = expected_log_likelihood_expanded(self.terms_expanded, self.X_torch)
        assert torch.isfinite(result), "Expected log-likelihood should be finite"
        assert result < 0, "Expected log-likelihood should be negative"

    def test_expected_log_likelihood_lower_bound(self):
        """Test lower-bound expected log-likelihood computation."""
        from PNMF.elbo import expected_log_likelihood_lower_bound

        result = expected_log_likelihood_lower_bound(self.terms_lb, self.X_torch)
        assert torch.isfinite(result), "Expected log-likelihood should be finite"
        assert result < 0, "Expected log-likelihood should be negative"

    def test_kl_divergence(self):
        """Test KL divergence computation."""
        kl = kl_divergence(self.qF, self.pF)
        assert torch.isfinite(kl), "KL should be finite"
        assert kl >= 0, "KL divergence should be non-negative"

    def test_compute_elbo_dispatcher(self):
        """Test ELBO dispatcher with all modes."""
        mode_terms = {
            'simple': self.terms_simple,
            'expanded': self.terms_expanded,
            'lower-bound': self.terms_lb,
        }
        for mode, terms in mode_terms.items():
            exp_ll, kl = compute_elbo(
                mode, terms, self.qF, self.pF, self.X_torch
            )
            assert torch.isfinite(exp_ll), f"exp_ll should be finite for mode={mode}"
            assert torch.isfinite(kl), f"kl should be finite for mode={mode}"

    def test_custom_kl_function(self):
        """Test ELBO with custom KL function."""

        def custom_kl(q, p):
            # Just return standard KL for testing
            return torch.distributions.kl_divergence(q, p).sum()

        exp_ll_standard, kl_standard = compute_elbo(
            'expanded', self.terms_expanded, self.qF, self.pF, self.X_torch
        )
        exp_ll_custom, kl_custom = compute_elbo(
            'expanded', self.terms_expanded, self.qF, self.pF, self.X_torch,
            kl_fn=custom_kl
        )

        torch.testing.assert_close(exp_ll_standard, exp_ll_custom)
        torch.testing.assert_close(kl_standard, kl_custom)


# =============================================================================
# Test compute_log_likelihood_terms
# =============================================================================


class TestLogLikelihoodTerms:
    """Test compute_log_likelihood_terms function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.X = generate_small_count_data()
        self.X_torch = torch.from_numpy(self.X.T.astype(np.float32))
        self.L = 5
        self.E = 3
        self.D, self.N = self.X_torch.shape

        self.prior = GaussianPrior(self.X_torch, L=self.L)
        self.model = PoissonFactorization(self.prior, self.X_torch, L=self.L)
        self.qF, _ = self.prior()
        self.W = self.model.W.data

    def test_lower_bound_keys(self):
        """Assert lower-bound terms have only analytic keys, no MC keys."""
        terms = compute_log_likelihood_terms(
            self.W, self.qF, self.X_torch, self.E, mode='lower-bound'
        )
        # Should have: exp_mu, exp_mu_sigma, rate_mean, rate_mgf
        assert 'exp_mu' in terms
        assert 'exp_mu_sigma' in terms
        assert 'rate_mean' in terms
        assert 'rate_mgf' in terms
        # Should NOT have MC keys
        assert 'exp_F_samples' not in terms
        assert 'term1_mc' not in terms
        assert 'term2_mc' not in terms

    def test_expanded_keys(self):
        """Assert expanded mode has exp_mu, exp_mu_sigma, rate_mgf, term1_mc."""
        terms = compute_log_likelihood_terms(
            self.W, self.qF, self.X_torch, self.E, mode='expanded',
        )
        # Should have
        assert 'exp_mu' in terms
        assert 'exp_mu_sigma' in terms
        assert 'rate_mgf' in terms
        assert 'term1_mc' in terms
        # Should NOT have
        assert 'rate_mean' not in terms
        assert 'term2_mc' not in terms
        assert 'exp_F_samples' not in terms

    def test_expanded_with_samples(self):
        """Assert expanded mode with return_samples=True also has exp_F_samples."""
        terms = compute_log_likelihood_terms(
            self.W, self.qF, self.X_torch, self.E, mode='expanded', return_samples=True,
        )
        assert 'exp_F_samples' in terms
        assert 'term1_mc' in terms
        assert 'rate_mc' not in terms

    def test_simple_keys(self):
        """Assert simple mode has exp_mu, term1_mc, term2_mc only."""
        terms = compute_log_likelihood_terms(
            self.W, self.qF, self.X_torch, self.E, mode='simple',
        )
        # Should have
        assert 'exp_mu' in terms
        assert 'term1_mc' in terms
        assert 'term2_mc' in terms
        # Should NOT have (not needed by simple mode)
        assert 'exp_mu_sigma' not in terms
        assert 'rate_mean' not in terms
        assert 'rate_mgf' not in terms

    def test_shapes_lower_bound(self):
        """Verify tensor shapes for lower-bound mode."""
        terms = compute_log_likelihood_terms(
            self.W, self.qF, self.X_torch, self.E, mode='lower-bound'
        )
        assert terms['exp_mu'].shape == (self.L, self.N)
        assert terms['exp_mu_sigma'].shape == (self.L, self.N)
        assert terms['rate_mean'].shape == (self.D, self.N)
        assert terms['rate_mgf'].shape == (self.D, self.N)

    def test_shapes_expanded(self):
        """Verify tensor shapes for expanded mode with samples."""
        terms = compute_log_likelihood_terms(
            self.W, self.qF, self.X_torch, self.E, mode='expanded', return_samples=True
        )
        assert terms['exp_mu'].shape == (self.L, self.N)
        assert terms['exp_mu_sigma'].shape == (self.L, self.N)
        assert terms['rate_mgf'].shape == (self.D, self.N)
        assert terms['exp_F_samples'].shape == (self.E, self.L, self.N)

    def test_memory_reduction(self):
        """Verify stored tensor sizes in terms dict.

        Memory used by terms (stored tensors):
        - return_samples=False: only scalars + (L, N) and (D, N) analytic tensors
        - return_samples=True:  above + exp_F_samples (E, L, N)
        - Old approach stored:  above + rate_mc (E, D, N), which is D/L times larger

        The batch matmul W @ exp_F_samples -> (E, D, N) is no longer in the
        autograd graph. The multiplicative update computes it under no_grad.
        """
        terms_default = compute_log_likelihood_terms(
            self.W, self.qF, self.X_torch, self.E, mode='expanded',
        )
        terms_with_samples = compute_log_likelihood_terms(
            self.W, self.qF, self.X_torch, self.E, mode='expanded', return_samples=True,
        )

        # Default should NOT have exp_F_samples
        assert 'exp_F_samples' not in terms_default

        # return_samples=True should have exp_F_samples but NOT rate_mc
        assert 'exp_F_samples' in terms_with_samples
        assert 'rate_mc' not in terms_with_samples

        # exp_F_samples uses E * L * N elements
        exp_F_size = terms_with_samples['exp_F_samples'].numel()
        assert exp_F_size == self.E * self.L * self.N

        # rate_mc would use E * D * N — strictly larger since D > L
        rate_mc_would_be = self.E * self.D * self.N
        assert rate_mc_would_be > exp_F_size


# =============================================================================
# Test separate scaling
# =============================================================================


class TestSeparateScaling:
    """Test that exp_ll and kl are returned as separate tensors."""

    def setup_method(self):
        """Set up test fixtures."""
        self.X = generate_small_count_data()
        self.X_torch = torch.from_numpy(self.X.T.astype(np.float32))
        self.L = 5
        self.E = 3

        self.prior = GaussianPrior(self.X_torch, L=self.L)
        self.qF, self.pF = self.prior()
        self.model = PoissonFactorization(self.prior, self.X_torch, L=self.L)
        self.W = self.model.W.data

    def test_exp_ll_and_kl_are_separate_tensors(self):
        """Verify compute_elbo returns two separate scalars."""
        terms = compute_log_likelihood_terms(
            self.W, self.qF, self.X_torch, self.E, mode='expanded'        )
        exp_ll, kl = compute_elbo('expanded', terms, self.qF, self.pF, self.X_torch)
        assert isinstance(exp_ll, torch.Tensor)
        assert isinstance(kl, torch.Tensor)
        assert exp_ll.dim() == 0  # scalar
        assert kl.dim() == 0      # scalar

    def test_kl_independent_of_mode(self):
        """KL should be the same regardless of mode."""
        terms_exp = compute_log_likelihood_terms(
            self.W, self.qF, self.X_torch, self.E, mode='expanded'        )
        terms_lb = compute_log_likelihood_terms(
            self.W, self.qF, self.X_torch, self.E, mode='lower-bound'
        )
        terms_simple = compute_log_likelihood_terms(
            self.W, self.qF, self.X_torch, self.E, mode='simple'        )

        _, kl_exp = compute_elbo('expanded', terms_exp, self.qF, self.pF, self.X_torch)
        _, kl_lb = compute_elbo('lower-bound', terms_lb, self.qF, self.pF, self.X_torch)
        _, kl_simple = compute_elbo('simple', terms_simple, self.qF, self.pF, self.X_torch)

        torch.testing.assert_close(kl_exp, kl_lb)
        torch.testing.assert_close(kl_exp, kl_simple)

    def test_custom_kl_with_new_signature(self):
        """Verify kl_fn still works with the new signature."""
        terms = compute_log_likelihood_terms(
            self.W, self.qF, self.X_torch, self.E, mode='expanded'        )

        def custom_kl(q, p):
            return torch.distributions.kl_divergence(q, p).sum() * 2.0

        _, kl_standard = compute_elbo('expanded', terms, self.qF, self.pF, self.X_torch)
        _, kl_custom = compute_elbo(
            'expanded', terms, self.qF, self.pF, self.X_torch, kl_fn=custom_kl
        )

        torch.testing.assert_close(kl_custom, kl_standard * 2.0)


# =============================================================================
# Test PyTorch API
# =============================================================================


class TestPyTorchAPI:
    """Test PyTorch-native API."""

    def test_poisson_factorization_forward(self):
        """Test PoissonFactorization forward pass (default: expanded mode)."""
        X = generate_small_count_data()
        X_torch = torch.from_numpy(X.T.astype(np.float32))
        L = 5
        E = 3

        prior = GaussianPrior(X_torch, L=L)
        model = PoissonFactorization(prior, X_torch, L=L)
        terms, qF, pF = model(E=E, X=X_torch)

        D, N = X_torch.shape
        # terms should be a dict with expanded-mode keys
        assert isinstance(terms, dict)
        assert 'exp_mu' in terms
        assert 'exp_mu_sigma' in terms
        assert 'rate_mgf' in terms
        assert 'term1_mc' in terms
        assert terms['exp_mu'].shape == (L, N)
        assert terms['rate_mgf'].shape == (D, N)
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
        TestLogLikelihoodTerms,
        TestSeparateScaling,
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
