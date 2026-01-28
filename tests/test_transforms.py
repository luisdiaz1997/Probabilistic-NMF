"""
Tests for PNMF transforms module.

Run with: python -m pytest tests/test_transforms.py -v
"""

import numpy as np
import torch
import pytest

from PNMF import PNMF
from PNMF import (
    log_factors,
    get_factors,
    factor_uncertainty,
    factor_samples,
    get_loadings,
    get_prior,
    transform_F,
    transform_W,
    log_factors_from_prior,
    factors_from_prior,
    uncertainty_from_prior,
)


# =============================================================================
# Test fixtures
# =============================================================================


def generate_count_data(n_samples=50, n_features=30, seed=42):
    """Generate integer count data appropriate for Poisson model."""
    np.random.seed(seed)
    rate = np.random.rand(n_samples, n_features) * 10 + 1
    X = np.random.poisson(rate).astype(np.float32)
    return X


def generate_small_count_data(seed=42):
    """Generate small integer count data for quick tests."""
    return generate_count_data(n_samples=20, n_features=15, seed=seed)


@pytest.fixture
def fitted_model():
    """Create a fitted PNMF model for testing."""
    X = generate_small_count_data()
    model = PNMF(n_components=5, max_iter=20, random_state=42)
    model.fit(X)
    return model, X


@pytest.fixture
def fitted_model_natural():
    """Create a fitted PNMF model with natural gradients for testing."""
    X = generate_small_count_data()
    model = PNMF(n_components=5, max_iter=20, random_state=42, training_mode='natural')
    model.fit(X)
    return model, X


# =============================================================================
# Test factor extraction functions
# =============================================================================


class TestLogFactors:
    """Test log_factors function."""

    def test_log_factors_shape(self, fitted_model):
        """Test that log_factors returns correct shape."""
        model, X = fitted_model
        F_log = log_factors(model)
        assert F_log.shape == (X.shape[0], model.n_components_)

    def test_log_factors_returns_numpy(self, fitted_model):
        """Test that log_factors returns numpy array by default."""
        model, _ = fitted_model
        F_log = log_factors(model)
        assert isinstance(F_log, np.ndarray)

    def test_log_factors_returns_tensor(self, fitted_model):
        """Test that log_factors returns tensor when requested."""
        model, _ = fitted_model
        F_log = log_factors(model, return_tensor=True)
        assert isinstance(F_log, torch.Tensor)

    def test_log_factors_natural_mode(self, fitted_model_natural):
        """Test log_factors with natural gradient parameterization."""
        model, X = fitted_model_natural
        F_log = log_factors(model)
        assert F_log.shape == (X.shape[0], model.n_components_)

    def test_log_factors_unfitted_raises(self):
        """Test that log_factors raises error for unfitted model."""
        model = PNMF(n_components=5)
        with pytest.raises(ValueError, match="not been fitted"):
            log_factors(model)


class TestGetFactors:
    """Test get_factors function."""

    def test_get_factors_shape(self, fitted_model):
        """Test that get_factors returns correct shape."""
        model, X = fitted_model
        F_exp = get_factors(model)
        assert F_exp.shape == (X.shape[0], model.n_components_)

    def test_get_factors_positive(self, fitted_model):
        """Test that get_factors returns positive values (exp-space)."""
        model, _ = fitted_model
        F_exp = get_factors(model)
        assert np.all(F_exp > 0)

    def test_get_factors_mgf_vs_direct(self, fitted_model):
        """Test that MGF gives different results than direct exp."""
        model, _ = fitted_model
        F_mgf = get_factors(model, use_mgf=True)
        F_direct = get_factors(model, use_mgf=False)

        # MGF should give larger values (exp(μ + σ²/2) > exp(μ) when σ > 0)
        assert np.mean(F_mgf) >= np.mean(F_direct) - 1e-6

    def test_get_factors_returns_tensor(self, fitted_model):
        """Test that get_factors returns tensor when requested."""
        model, _ = fitted_model
        F_exp = get_factors(model, return_tensor=True)
        assert isinstance(F_exp, torch.Tensor)


class TestFactorUncertainty:
    """Test factor_uncertainty function."""

    def test_uncertainty_shape(self, fitted_model):
        """Test that factor_uncertainty returns correct shape."""
        model, X = fitted_model
        F_std = factor_uncertainty(model)
        assert F_std.shape == (X.shape[0], model.n_components_)

    def test_uncertainty_positive(self, fitted_model):
        """Test that uncertainty is positive."""
        model, _ = fitted_model
        F_std = factor_uncertainty(model)
        assert np.all(F_std > 0)

    def test_variance_vs_std(self, fitted_model):
        """Test that variance = std^2."""
        model, _ = fitted_model
        F_std = factor_uncertainty(model, return_variance=False)
        F_var = factor_uncertainty(model, return_variance=True)
        np.testing.assert_array_almost_equal(F_var, F_std ** 2)

    def test_uncertainty_natural_mode(self, fitted_model_natural):
        """Test factor_uncertainty with natural gradient parameterization."""
        model, X = fitted_model_natural
        F_std = factor_uncertainty(model)
        assert F_std.shape == (X.shape[0], model.n_components_)
        assert np.all(F_std > 0)


class TestFactorSamples:
    """Test factor_samples function."""

    def test_samples_shape(self, fitted_model):
        """Test that factor_samples returns correct shape."""
        model, X = fitted_model
        n_samples = 50
        samples = factor_samples(model, n_samples=n_samples)
        assert samples.shape == (n_samples, X.shape[0], model.n_components_)

    def test_samples_exp_positive(self, fitted_model):
        """Test that exp samples are positive."""
        model, _ = fitted_model
        samples = factor_samples(model, n_samples=20, return_exp=True)
        assert np.all(samples > 0)

    def test_samples_statistics(self, fitted_model):
        """Test that sample statistics match variational parameters."""
        model, _ = fitted_model
        n_samples = 1000

        samples = factor_samples(model, n_samples=n_samples)
        F_log = log_factors(model)
        F_std = factor_uncertainty(model)

        # Sample mean should be close to variational mean
        sample_mean = samples.mean(axis=0)
        np.testing.assert_array_almost_equal(sample_mean, F_log, decimal=1)

        # Sample std should be close to variational std
        sample_std = samples.std(axis=0)
        np.testing.assert_array_almost_equal(sample_std, F_std, decimal=1)


# =============================================================================
# Test model accessor functions
# =============================================================================


class TestGetLoadings:
    """Test get_loadings function."""

    def test_loadings_shape(self, fitted_model):
        """Test that get_loadings returns correct shape."""
        model, X = fitted_model
        W = get_loadings(model)
        assert W.shape == (X.shape[1], model.n_components_)

    def test_loadings_consistent(self, fitted_model):
        """Test that get_loadings is consistent with components_."""
        model, _ = fitted_model
        W = get_loadings(model)
        np.testing.assert_array_equal(W, model.components_.T)

    def test_loadings_returns_tensor(self, fitted_model):
        """Test that get_loadings returns tensor when requested."""
        model, _ = fitted_model
        W = get_loadings(model, return_tensor=True)
        assert isinstance(W, torch.Tensor)

    def test_loadings_unfitted_raises(self):
        """Test that get_loadings raises error for unfitted model."""
        model = PNMF(n_components=5)
        with pytest.raises(ValueError, match="not been fitted"):
            get_loadings(model)


class TestGetPrior:
    """Test get_prior function."""

    def test_get_prior_returns_prior(self, fitted_model):
        """Test that get_prior returns a GaussianPrior."""
        from PNMF import GaussianPrior

        model, _ = fitted_model
        prior = get_prior(model)
        assert isinstance(prior, GaussianPrior)

    def test_get_prior_is_same_object(self, fitted_model):
        """Test that get_prior returns the same object."""
        model, _ = fitted_model
        prior1 = get_prior(model)
        prior2 = get_prior(model)
        assert prior1 is prior2

    def test_get_prior_unfitted_raises(self):
        """Test that get_prior raises error for unfitted model."""
        model = PNMF(n_components=5)
        with pytest.raises(ValueError, match="not been fitted"):
            get_prior(model)


# =============================================================================
# Test conditional inference functions
# =============================================================================


class TestTransformF:
    """Test transform_F function."""

    def test_transform_F_shape(self, fitted_model):
        """Test that transform_F returns correct shape."""
        model, X = fitted_model
        W = get_loadings(model)

        # Transform new data
        X_new = generate_small_count_data(seed=123)[:10]  # 10 new samples
        F_new = transform_F(X_new, W, max_iter=20)

        assert F_new.shape == (10, model.n_components_)

    def test_transform_F_positive(self, fitted_model):
        """Test that transform_F returns positive values."""
        model, X = fitted_model
        W = get_loadings(model)

        X_new = generate_small_count_data(seed=123)[:10]
        F_new = transform_F(X_new, W, max_iter=20)

        assert np.all(F_new > 0)

    def test_transform_F_return_prior(self, fitted_model):
        """Test that transform_F can return prior object."""
        from PNMF import GaussianPrior

        model, X = fitted_model
        W = get_loadings(model)

        X_new = generate_small_count_data(seed=123)[:10]
        prior = transform_F(X_new, W, max_iter=20, return_prior=True)

        assert isinstance(prior, GaussianPrior)

    def test_transform_F_dimension_mismatch(self, fitted_model):
        """Test that transform_F raises error on dimension mismatch."""
        model, X = fitted_model
        W = get_loadings(model)

        # Wrong number of features
        X_bad = np.random.rand(10, 100).astype(np.float32)
        with pytest.raises(ValueError, match="mismatch"):
            transform_F(X_bad, W)

    def test_transform_F_modes(self, fitted_model):
        """Test transform_F with different ELBO modes."""
        model, X = fitted_model
        W = get_loadings(model)
        X_new = generate_small_count_data(seed=123)[:10]

        for mode in ['simple', 'expanded', 'lower-bound']:
            F_new = transform_F(X_new, W, mode=mode, max_iter=20)
            assert F_new.shape == (10, model.n_components_)


class TestTransformW:
    """Test transform_W function."""

    def test_transform_W_shape(self, fitted_model):
        """Test that transform_W returns correct shape."""
        model, X = fitted_model
        F_exp = get_factors(model)  # Uses exp-space factors

        # Learn new W for same data
        W_new = transform_W(X, F_exp, max_iter=20)

        assert W_new.shape == (X.shape[1], model.n_components_)

    def test_transform_W_positive(self, fitted_model):
        """Test that transform_W returns positive values."""
        model, X = fitted_model
        F_exp = get_factors(model)

        W_new = transform_W(X, F_exp, max_iter=20)

        assert np.all(W_new >= 0)

    def test_transform_W_dimension_mismatch(self, fitted_model):
        """Test that transform_W raises error on dimension mismatch."""
        model, X = fitted_model
        F_exp = get_factors(model)

        # Wrong number of samples
        X_bad = np.random.rand(100, X.shape[1]).astype(np.float32)
        with pytest.raises(ValueError, match="mismatch"):
            transform_W(X_bad, F_exp)

    def test_transform_W_with_init(self, fitted_model):
        """Test transform_W with initial W provided."""
        model, X = fitted_model
        F_exp = get_factors(model)

        # Provide initial W
        W_init = np.random.rand(X.shape[1], model.n_components_).astype(np.float32) * 0.1
        W_new = transform_W(X, F_exp, W_init=W_init, max_iter=20)

        assert W_new.shape == (X.shape[1], model.n_components_)
        assert np.all(W_new >= 0)


# =============================================================================
# Test prior utility functions
# =============================================================================


class TestPriorUtilities:
    """Test utility functions for working with priors."""

    def test_log_factors_from_prior(self, fitted_model):
        """Test log_factors_from_prior matches log_factors."""
        model, X = fitted_model
        prior = get_prior(model)

        F_log_model = log_factors(model)
        F_log_prior = log_factors_from_prior(prior)

        np.testing.assert_array_equal(F_log_model, F_log_prior)

    def test_factors_from_prior(self, fitted_model):
        """Test factors_from_prior matches get_factors."""
        model, X = fitted_model
        prior = get_prior(model)

        # Test with same use_mgf setting
        F_exp_model = get_factors(model, use_mgf=True)
        F_exp_prior = factors_from_prior(prior, use_mgf=True)

        np.testing.assert_array_equal(F_exp_model, F_exp_prior)

    def test_uncertainty_from_prior(self, fitted_model):
        """Test uncertainty_from_prior matches factor_uncertainty."""
        model, X = fitted_model
        prior = get_prior(model)

        F_std_model = factor_uncertainty(model)
        F_std_prior = uncertainty_from_prior(prior)

        np.testing.assert_array_equal(F_std_model, F_std_prior)

    def test_prior_utilities_with_transform_F(self, fitted_model):
        """Test prior utilities work with transform_F result."""
        model, X = fitted_model
        W = get_loadings(model)

        X_new = generate_small_count_data(seed=123)[:10]
        prior_new = transform_F(X_new, W, max_iter=20, return_prior=True)

        # All utilities should work
        F_log = log_factors_from_prior(prior_new)
        F_exp = factors_from_prior(prior_new)
        F_std = uncertainty_from_prior(prior_new)

        assert F_log.shape == (10, model.n_components_)
        assert F_exp.shape == (10, model.n_components_)
        assert F_std.shape == (10, model.n_components_)


# =============================================================================
# Integration tests
# =============================================================================


class TestIntegration:
    """Integration tests for transforms module."""

    def test_reconstruction_with_get_factors(self, fitted_model):
        """Test that factors and loadings can reconstruct data."""
        model, X = fitted_model

        F_exp = get_factors(model)
        W = get_loadings(model)

        # Reconstruct: X ≈ exp(F) @ W.T
        X_recon = F_exp @ W.T

        # Should have reasonable reconstruction error
        error = np.mean((X - X_recon) ** 2)
        assert error < np.var(X) * 2, "Reconstruction error too high"

    def test_transform_F_improves_reconstruction(self, fitted_model):
        """Test that transform_F gives good reconstruction."""
        model, X = fitted_model
        W = get_loadings(model)

        # Use full VI transform with more iterations for better fit
        F_vi = transform_F(X, W, max_iter=100)

        # Reconstruct
        X_recon = F_vi @ W.T

        # Should have reasonable reconstruction error (relaxed threshold for stochastic model)
        error = np.mean((X - X_recon) ** 2)
        assert error < np.var(X) * 5, "VI transform reconstruction error too high"

    def test_full_pipeline(self):
        """Test full pipeline: fit -> extract -> transform new data."""
        # Generate training and test data
        X_train = generate_count_data(n_samples=50, n_features=30, seed=42)
        X_test = generate_count_data(n_samples=20, n_features=30, seed=123)

        # Fit model
        model = PNMF(n_components=5, max_iter=50, random_state=42)
        model.fit(X_train)

        # Extract factors and loadings from training
        F_train = get_factors(model)
        W = get_loadings(model)

        # Transform test data with full VI
        F_test = transform_F(X_test, W, max_iter=50)

        # Check shapes
        assert F_train.shape == (50, 5)
        assert F_test.shape == (20, 5)
        assert W.shape == (30, 5)

        # Reconstruct test data
        X_test_recon = F_test @ W.T
        assert X_test_recon.shape == X_test.shape

    def test_uncertainty_quantification_pipeline(self):
        """Test uncertainty quantification workflow."""
        X = generate_small_count_data()

        model = PNMF(n_components=5, max_iter=50, random_state=42)
        model.fit(X)

        # Get point estimates
        F_log = log_factors(model)
        F_exp = get_factors(model)
        F_std = factor_uncertainty(model)

        # Get samples for uncertainty propagation
        samples = factor_samples(model, n_samples=1000, return_exp=True)

        # Verify shapes
        assert F_log.shape == (X.shape[0], 5)
        assert F_exp.shape == (X.shape[0], 5)
        assert F_std.shape == (X.shape[0], 5)
        assert samples.shape == (1000, X.shape[0], 5)

        # Verify MGF relationship approximately holds for samples
        sample_mean = samples.mean(axis=0)
        # E[exp(F)] ≈ exp(μ + σ²/2) for large number of samples
        expected_mean = get_factors(model, use_mgf=True)
        # Relaxed tolerance for stochastic comparison
        np.testing.assert_array_almost_equal(sample_mean, expected_mean, decimal=0)


# =============================================================================
# Run tests
# =============================================================================


def run_all_tests():
    """Run all tests without pytest."""
    test_classes = [
        TestLogFactors,
        TestGetFactors,
        TestFactorUncertainty,
        TestFactorSamples,
        TestGetLoadings,
        TestGetPrior,
        TestTransformF,
        TestTransformW,
        TestPriorUtilities,
        TestIntegration,
    ]

    passed = 0
    failed = 0
    errors = []

    # Create fixtures
    X = generate_small_count_data()
    model_standard = PNMF(n_components=5, max_iter=20, random_state=42)
    model_standard.fit(X)
    fitted_model_fixture = (model_standard, X)

    model_natural = PNMF(n_components=5, max_iter=20, random_state=42, training_mode='natural')
    model_natural.fit(X)
    fitted_model_natural_fixture = (model_natural, X)

    for test_class in test_classes:
        print(f"\n{'='*60}")
        print(f"Running {test_class.__name__}")
        print('='*60)

        instance = test_class()

        # Get all test methods
        test_methods = [m for m in dir(instance) if m.startswith('test_')]

        for method_name in test_methods:
            try:
                method = getattr(instance, method_name)

                # Check method signature for fixtures
                import inspect
                sig = inspect.signature(method)
                params = list(sig.parameters.keys())

                if 'fitted_model' in params:
                    method(fitted_model_fixture)
                elif 'fitted_model_natural' in params:
                    method(fitted_model_natural_fixture)
                else:
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
