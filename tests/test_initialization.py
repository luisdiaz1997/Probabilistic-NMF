"""Tests for initialization module."""

import numpy as np
import pytest
import torch

from PNMF.initialization import (
    initialize_factors,
    _initialize_random,
    _initialize_nndsvd,
    _initialize_kmeans,
)
from PNMF import PNMF


class TestInitializeFactors:
    """Test initialize_factors function."""

    @pytest.fixture
    def X(self):
        """Simple test data."""
        np.random.seed(42)
        return np.random.poisson(5, size=(50, 30)).astype(np.float32)

    def test_random_init(self, X):
        """Test random initialization."""
        W, exp_F = initialize_factors(X, n_components=5, init='random', random_state=42)

        assert W.shape == (30, 5)
        assert exp_F.shape == (50, 5)
        assert np.all(W >= 0)
        assert np.all(exp_F >= 0)

    def test_nndsvd_init(self, X):
        """Test NNDSVD initialization."""
        W, exp_F = initialize_factors(X, n_components=5, init='nndsvd', random_state=42)

        assert W.shape == (30, 5)
        assert exp_F.shape == (50, 5)
        assert np.all(W >= 0)
        assert np.all(exp_F >= 0)
        # NNDSVD should produce sparse results
        assert np.mean(W == 0) > 0  # Some zeros

    def test_nndsvda_init(self, X):
        """Test NNDSVD with average fill."""
        W, exp_F = initialize_factors(X, n_components=5, init='nndsvda', random_state=42)

        assert W.shape == (30, 5)
        assert exp_F.shape == (50, 5)
        assert np.all(W >= 0)
        assert np.all(exp_F >= 0)
        # nndsvda should have no zeros
        assert np.all(W > 0)
        assert np.all(exp_F > 0)

    def test_nndsvdar_init(self, X):
        """Test NNDSVD with random fill."""
        W, exp_F = initialize_factors(X, n_components=5, init='nndsvdar', random_state=42)

        assert W.shape == (30, 5)
        assert exp_F.shape == (50, 5)
        assert np.all(W >= 0)
        assert np.all(exp_F >= 0)

    def test_kmeans_init(self, X):
        """Test k-means initialization."""
        W, exp_F = initialize_factors(X, n_components=5, init='k-means', random_state=42)

        assert W.shape == (30, 5)
        assert exp_F.shape == (50, 5)
        assert np.all(W >= 0)
        assert np.all(exp_F >= 0)

    def test_auto_init(self, X):
        """Test auto init selection (None)."""
        # n_components=5 <= min(50, 30), so should use nndsvd
        W, exp_F = initialize_factors(X, n_components=5, init=None, random_state=42)
        # Should be same as nndsvd
        W2, exp_F2 = initialize_factors(X, n_components=5, init='nndsvd', random_state=42)
        np.testing.assert_array_almost_equal(W, W2)
        np.testing.assert_array_almost_equal(exp_F, exp_F2)

    def test_auto_init_fallback_to_random(self):
        """Test auto init falls back to random when too many components."""
        X = np.random.poisson(5, size=(10, 5)).astype(np.float32)
        # n_components=20 > min(10, 5), so should use random
        W, exp_F = initialize_factors(X, n_components=20, init=None, random_state=42)
        assert W.shape == (5, 20)
        assert exp_F.shape == (10, 20)

    def test_invalid_init(self, X):
        """Test invalid init parameter."""
        with pytest.raises(ValueError, match="Unknown init parameter"):
            initialize_factors(X, n_components=5, init='invalid_method')

    def test_nndsvd_too_many_components(self, X):
        """Test NNDSVD with too many components."""
        with pytest.raises(ValueError, match="must be <= min"):
            initialize_factors(X, n_components=100, init='nndsvd')

    def test_negative_input(self):
        """Test with negative input."""
        X = np.random.randn(10, 5)
        with pytest.raises(ValueError, match="must be non-negative"):
            initialize_factors(X, n_components=3)

    def test_torch_input(self, X):
        """Test with torch tensor input."""
        X_torch = torch.from_numpy(X)
        W, exp_F = initialize_factors(X_torch, n_components=5, init='random', random_state=42)

        assert W.shape == (30, 5)
        assert exp_F.shape == (50, 5)


class TestPNMFInitParameter:
    """Test PNMF with init parameter."""

    @pytest.fixture
    def X(self):
        """Simple test data."""
        np.random.seed(42)
        return np.random.poisson(5, size=(50, 30)).astype(np.float32)

    @pytest.mark.parametrize("init", [None, 'random', 'nndsvda', 'k-means'])
    def test_pnmf_with_init(self, X, init):
        """Test PNMF with different init options."""
        model = PNMF(n_components=5, init=init, random_state=42, max_iter=20, verbose=False)
        model.fit(X)

        assert model.components_.shape == (5, 30)
        assert model.n_features_in_ == 30
        assert model.elbo_ is not None

    def test_nndsvd_init_with_valid_components(self, X):
        """Test NNDSVD init with valid number of components."""
        model = PNMF(n_components=5, init='nndsvd', random_state=42, max_iter=20, verbose=False)
        model.fit(X)
        assert model.components_.shape == (5, 30)

    def test_nndsvdar_init(self, X):
        """Test NNDSVD with random fill init."""
        model = PNMF(n_components=5, init='nndsvdar', random_state=42, max_iter=20, verbose=False)
        model.fit(X)
        assert model.components_.shape == (5, 30)

    def test_invalid_init_raises(self, X):
        """Test that invalid init raises error."""
        with pytest.raises(ValueError, match="init must be one of"):
            PNMF(n_components=5, init='invalid').fit(X)

    def test_init_with_natural_training_mode(self, X):
        """Test init parameter works with natural gradient training."""
        model = PNMF(
            n_components=5,
            init='nndsvda',
            training_mode='natural',
            random_state=42,
            max_iter=20,
            verbose=False
        )
        model.fit(X)
        assert model.components_.shape == (5, 30)

    def test_init_with_different_modes(self, X):
        """Test init parameter works with different ELBO modes."""
        for mode in ['simple', 'expanded', 'lower-bound']:
            model = PNMF(
                n_components=5,
                init='random',
                mode=mode,
                random_state=42,
                max_iter=10,
                verbose=False
            )
            model.fit(X)
            assert model.components_.shape == (5, 30)


class TestReproducibility:
    """Test reproducibility of initialization methods."""

    def test_random_init_reproducibility(self):
        """Test that random init is reproducible with same random_state."""
        X = np.random.poisson(5, size=(30, 20)).astype(np.float32)

        W1, exp_F1 = initialize_factors(X, n_components=5, init='random', random_state=42)
        W2, exp_F2 = initialize_factors(X, n_components=5, init='random', random_state=42)

        np.testing.assert_array_equal(W1, W2)
        np.testing.assert_array_equal(exp_F1, exp_F2)

    def test_random_init_different_state(self):
        """Test that different random states give different results."""
        X = np.random.poisson(5, size=(30, 20)).astype(np.float32)

        W1, exp_F1 = initialize_factors(X, n_components=5, init='random', random_state=42)
        W2, exp_F2 = initialize_factors(X, n_components=5, init='random', random_state=123)

        assert not np.array_equal(W1, W2)
        assert not np.array_equal(exp_F1, exp_F2)

    def test_nndsvd_reproducibility(self):
        """Test that NNDSVD is deterministic."""
        X = np.random.poisson(5, size=(30, 20)).astype(np.float32)

        W1, exp_F1 = initialize_factors(X, n_components=5, init='nndsvda', random_state=42)
        W2, exp_F2 = initialize_factors(X, n_components=5, init='nndsvda', random_state=42)

        np.testing.assert_array_almost_equal(W1, W2)
        np.testing.assert_array_almost_equal(exp_F1, exp_F2)
