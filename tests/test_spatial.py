"""Tests for PNMF spatial mode with SVGP prior."""

import numpy as np
import pytest
import torch

# Check if gpzoo is available
try:
    import gpzoo
    HAS_GPZOO = True
except ImportError:
    HAS_GPZOO = False

pytestmark = pytest.mark.skipif(not HAS_GPZOO, reason="gpzoo not installed")

from PNMF import PNMF
from PNMF.transforms import (
    log_factors,
    get_factors,
    factor_uncertainty,
    factor_samples,
    get_loadings,
)


# Shared test data fixtures
@pytest.fixture
def spatial_data():
    """Generate synthetic spatial count data with 3 groups."""
    np.random.seed(42)
    N, D = 100, 50
    X = np.random.poisson(5, (N, D)).astype(np.float32)
    coords = np.random.randn(N, 2).astype(np.float32)
    groups = np.random.randint(0, 3, N)
    return X, coords, groups


@pytest.fixture
def small_spatial_data():
    """Smaller data for fast tests."""
    np.random.seed(123)
    N, D = 30, 20
    X = np.random.poisson(5, (N, D)).astype(np.float32)
    coords = np.random.randn(N, 2).astype(np.float32)
    groups = np.random.randint(0, 2, N)
    return X, coords, groups


class TestSpatialValidation:
    """Test parameter validation for spatial mode."""

    def test_spatial_requires_coordinates(self, spatial_data):
        X, coords, groups = spatial_data
        model = PNMF(spatial=True, n_components=3, max_iter=1, num_inducing=20)
        with pytest.raises(ValueError, match="coordinates"):
            model.fit(X)

    def test_multigroup_requires_groups(self, spatial_data):
        X, coords, groups = spatial_data
        model = PNMF(spatial=True, multigroup=True, n_components=3, max_iter=1, num_inducing=20)
        with pytest.raises(ValueError, match="groups"):
            model.fit(X, coordinates=coords)

    def test_natural_gradient_not_supported(self):
        with pytest.raises(ValueError, match="Natural gradient"):
            model = PNMF(spatial=True, training_mode='natural')
            model._validate_params()

    def test_invalid_inducing_allocation(self):
        with pytest.raises(ValueError, match="inducing_allocation"):
            model = PNMF(spatial=True, inducing_allocation='invalid')
            model._validate_params()

    def test_invalid_kernel(self):
        with pytest.raises(ValueError, match="kernel"):
            model = PNMF(spatial=True, kernel='RBF')
            model._validate_params()

    def test_auto_prior_type(self):
        model = PNMF(spatial=True)
        assert model.prior_type == 'SVGP'


class TestSpatialFit:
    """Test basic spatial fitting."""

    def test_fit_spatial_multigroup(self, spatial_data):
        X, coords, groups = spatial_data
        model = PNMF(
            n_components=3, spatial=True, max_iter=5,
            num_inducing=20, random_state=42,
        )
        model.fit(X, coordinates=coords, groups=groups)

        assert model.components_.shape == (3, 50)
        assert model.elbo_ is not None
        assert model.n_iter_ > 0

    def test_fit_spatial_no_multigroup(self, spatial_data):
        X, coords, groups = spatial_data
        model = PNMF(
            n_components=3, spatial=True, multigroup=False,
            max_iter=5, num_inducing=20, random_state=42,
        )
        model.fit(X, coordinates=coords)

        assert model.components_.shape == (3, 50)
        assert model.elbo_ is not None

    def test_fit_spatial_with_history(self, small_spatial_data):
        X, coords, groups = small_spatial_data
        model = PNMF(
            n_components=2, spatial=True, max_iter=5,
            num_inducing=15, random_state=42,
        )
        history, model = model.fit(
            X, coordinates=coords, groups=groups, return_history=True
        )

        assert isinstance(history, list)
        assert len(history) == 5
        assert model.components_ is not None

    def test_fit_spatial_elbo_modes(self, small_spatial_data):
        X, coords, groups = small_spatial_data
        for mode in ['simple', 'expanded', 'lower-bound']:
            model = PNMF(
                n_components=2, spatial=True, mode=mode,
                max_iter=3, num_inducing=15, random_state=42,
            )
            model.fit(X, coordinates=coords, groups=groups)
            assert model.components_.shape == (2, 20)


class TestSpatialBatching:
    """Test spatial mode with mini-batching."""

    def test_spatial_with_sample_batching(self, spatial_data):
        X, coords, groups = spatial_data
        model = PNMF(
            n_components=3, spatial=True, max_iter=5,
            num_inducing=20, batch_size=30, random_state=42,
        )
        model.fit(X, coordinates=coords, groups=groups)
        assert model.components_.shape == (3, 50)

    def test_spatial_with_feature_batching(self, spatial_data):
        X, coords, groups = spatial_data
        model = PNMF(
            n_components=3, spatial=True, max_iter=5,
            num_inducing=20, y_batch_size=20, random_state=42,
        )
        model.fit(X, coordinates=coords, groups=groups)
        assert model.components_.shape == (3, 50)

    def test_spatial_with_both_batching(self, spatial_data):
        X, coords, groups = spatial_data
        model = PNMF(
            n_components=3, spatial=True, max_iter=5,
            num_inducing=20, batch_size=30, y_batch_size=20,
            random_state=42,
        )
        model.fit(X, coordinates=coords, groups=groups)
        assert model.components_.shape == (3, 50)


class TestSpatialTransform:
    """Test transform and factor extraction for spatial models."""

    def test_transform_spatial(self, spatial_data):
        X, coords, groups = spatial_data
        model = PNMF(
            n_components=3, spatial=True, max_iter=5,
            num_inducing=20, random_state=42,
        )
        model.fit(X, coordinates=coords, groups=groups)

        # Transform with same coordinates
        transformed = model.transform(X, coordinates=coords, groups=groups)
        assert transformed.shape == (100, 3)
        assert np.all(transformed > 0)  # exp(F) should be positive

    def test_transform_new_coordinates(self, spatial_data):
        X, coords, groups = spatial_data
        model = PNMF(
            n_components=3, spatial=True, max_iter=5,
            num_inducing=20, random_state=42,
        )
        model.fit(X, coordinates=coords, groups=groups)

        # Transform with new coordinates
        np.random.seed(99)
        new_coords = np.random.randn(20, 2).astype(np.float32)
        new_groups = np.random.randint(0, 3, 20)
        new_X = np.random.poisson(5, (20, 50)).astype(np.float32)

        transformed = model.transform(new_X, coordinates=new_coords, groups=new_groups)
        assert transformed.shape == (20, 3)

    def test_transform_requires_coordinates(self, spatial_data):
        X, coords, groups = spatial_data
        model = PNMF(
            n_components=3, spatial=True, max_iter=5,
            num_inducing=20, random_state=42,
        )
        model.fit(X, coordinates=coords, groups=groups)

        with pytest.raises(ValueError, match="coordinates"):
            model.transform(X)

    def test_fit_transform_spatial(self, spatial_data):
        X, coords, groups = spatial_data
        model = PNMF(
            n_components=3, spatial=True, max_iter=5,
            num_inducing=20, random_state=42,
        )
        transformed = model.fit_transform(X, coordinates=coords, groups=groups)
        assert transformed.shape == (100, 3)


class TestSpatialFactorExtraction:
    """Test factor extraction functions with spatial models."""

    @pytest.fixture(autouse=True)
    def setup_model(self, small_spatial_data):
        X, coords, groups = small_spatial_data
        self.model = PNMF(
            n_components=2, spatial=True, max_iter=5,
            num_inducing=15, random_state=42,
        )
        self.model.fit(X, coordinates=coords, groups=groups)
        self.X = X
        self.coords = coords
        self.groups = groups

    def test_log_factors(self):
        F_log = log_factors(self.model, coordinates=self.coords, groups=self.groups)
        assert F_log.shape == (30, 2)

    def test_get_factors(self):
        F_exp = get_factors(self.model, coordinates=self.coords, groups=self.groups)
        assert F_exp.shape == (30, 2)
        assert np.all(F_exp > 0)

    def test_get_factors_mgf(self):
        F_exp = get_factors(
            self.model, use_mgf=True,
            coordinates=self.coords, groups=self.groups
        )
        assert F_exp.shape == (30, 2)
        assert np.all(F_exp > 0)

    def test_factor_uncertainty(self):
        F_std = factor_uncertainty(
            self.model, coordinates=self.coords, groups=self.groups
        )
        assert F_std.shape == (30, 2)
        assert np.all(F_std > 0)

    def test_factor_uncertainty_variance(self):
        F_var = factor_uncertainty(
            self.model, return_variance=True,
            coordinates=self.coords, groups=self.groups
        )
        assert F_var.shape == (30, 2)
        assert np.all(F_var > 0)

    def test_factor_samples(self):
        samples = factor_samples(
            self.model, n_samples=10,
            coordinates=self.coords, groups=self.groups
        )
        assert samples.shape == (10, 30, 2)

    def test_factor_samples_exp(self):
        samples = factor_samples(
            self.model, n_samples=10, return_exp=True,
            coordinates=self.coords, groups=self.groups
        )
        assert samples.shape == (10, 30, 2)
        assert np.all(samples > 0)

    def test_get_loadings(self):
        W = get_loadings(self.model)
        assert W.shape == (20, 2)

    def test_log_factors_uses_stored_coords(self):
        """log_factors should work without explicit coordinates (uses stored ones)."""
        F_log = log_factors(self.model)
        assert F_log.shape == (30, 2)
