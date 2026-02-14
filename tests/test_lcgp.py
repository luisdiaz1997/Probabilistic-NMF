"""Tests for PNMF spatial mode with LCGP prior."""

import numpy as np
import pytest
import torch
import warnings

# Check if gpzoo is available
try:
    import gpzoo
    HAS_GPZOO = True
except ImportError:
    HAS_GPZOO = False

pytestmark = pytest.mark.skipif(not HAS_GPZOO, reason="gpzoo not installed")


# Shared test data fixtures
@pytest.fixture
def lcgp_data():
    """Generate synthetic spatial count data for LCGP tests (smaller N for faster tests)."""
    np.random.seed(42)
    N, D = 50, 20
    X = np.random.poisson(5, (N, D)).astype(np.float32)
    coords = np.random.rand(N, 2).astype(np.float32) * 50
    groups = np.random.randint(0, 3, N)
    return X, coords, groups


@pytest.fixture
def small_lcgp_data():
    """Even smaller data for fast tests. Uses K=3 (not 5) to work around GPzoo LCGP rank constraint."""
    np.random.seed(123)
    N, D = 30, 15
    X = np.random.poisson(5, (N, D)).astype(np.float32)
    coords = np.random.rand(N, 2).astype(np.float32) * 50
    groups = np.random.randint(0, 2, N)
    return X, coords, groups


from PNMF import PNMF
from PNMF.transforms import (
    log_factors,
    get_factors,
    factor_uncertainty,
    factor_samples,
    get_loadings,
)


class TestLCGPValidation:
    """Test parameter validation for LCGP mode."""

    def test_lcgp_prior_accepted(self):
        """local=True should be accepted with spatial=True."""
        model = PNMF(
            spatial=True, local=True, n_components=3, K=10, max_iter=1,
        )
        assert model.prior_type == 'LCGP'
        # Should not raise
        model._validate_params()

    def test_lcgp_requires_spatial(self):
        """local=True without spatial=True should raise error."""
        model = PNMF(spatial=False, local=True)
        with pytest.raises(ValueError, match="local=True requires spatial=True"):
            model._validate_params()

    def test_invalid_K_raises_error(self):
        """K < 1 should raise error."""
        model = PNMF(spatial=True, local=True, K=0)
        with pytest.raises(ValueError, match="K must be >= 1"):
            model._validate_params()

    def test_invalid_rank_raises_error(self):
        """rank < 1 should raise error."""
        model = PNMF(spatial=True, local=True, rank=0)
        with pytest.raises(ValueError, match="rank must be >= 1"):
            model._validate_params()

    def test_invalid_low_rank_mode_raises_error(self):
        """Invalid low_rank_mode should raise error."""
        model = PNMF(spatial=True, local=True, low_rank_mode='invalid')
        with pytest.raises(ValueError, match="low_rank_mode must be"):
            model._validate_params()

    def test_natural_gradient_not_supported_lcgp(self):
        """Natural gradient training not supported with LCGP."""
        model = PNMF(spatial=True, local=True, training_mode='natural')
        with pytest.raises(ValueError, match="Natural gradient"):
            model._validate_params()

    def test_num_inducing_warning(self, lcgp_data):
        """Setting num_inducing with LCGP should warn (it's ignored)."""
        X, coords, groups = lcgp_data
        model = PNMF(
            spatial=True, local=True, n_components=3,
            num_inducing=100,  # Different from default (3000)
            K=10, max_iter=1,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model._validate_params()
            assert len(w) == 1
            assert "num_inducing is ignored" in str(w[0].message)


class TestLCGPFit:
    """Test basic LCGP fitting."""

    def test_fit_lcgp_no_groups(self, lcgp_data):
        """Test LCGP fit without groups (multigroup=False)."""
        X, coords, _ = lcgp_data
        model = PNMF(
            n_components=3,
            spatial=True,
            local=True,
            multigroup=False,
            K=3,
            max_iter=5,
            random_state=42,
        )
        model.fit(X, coordinates=coords)

        assert model.components_.shape == (3, 20)
        assert model.elbo_ is not None
        assert not np.isnan(model.elbo_)
        assert model.n_iter_ > 0

    def test_fit_lcgp_multigroup(self, lcgp_data):
        """Test LCGP fit with groups (multigroup=True)."""
        X, coords, groups = lcgp_data
        model = PNMF(
            n_components=3,
            spatial=True,
            local=True,
            multigroup=True,
            K=3,
            max_iter=5,
            random_state=42,
        )
        model.fit(X, coordinates=coords, groups=groups)

        assert model.components_.shape == (3, 20)
        assert model.elbo_ is not None
        assert not np.isnan(model.elbo_)
        assert model.n_iter_ > 0

    def test_fit_lcgp_with_history(self, small_lcgp_data):
        """Test LCGP fit with history tracking."""
        X, coords, groups = small_lcgp_data
        model = PNMF(
            n_components=2,
            spatial=True,
            local=True,
            multigroup=True,
            K=3,
            max_iter=5,
            random_state=42,
        )
        history, model = model.fit(
            X, coordinates=coords, groups=groups, return_history=True
        )

        assert isinstance(history, list)
        assert len(history) == 5
        assert model.components_ is not None
        # ELBO should increase (become less negative)
        assert history[-1] >= history[0] - 100  # Allow some noise

    def test_fit_lcgp_elbo_modes(self, small_lcgp_data):
        """Test all three ELBO modes with LCGP."""
        X, coords, groups = small_lcgp_data
        for mode in ['simple', 'expanded', 'lower-bound']:
            model = PNMF(
                n_components=2,
                spatial=True,
                local=True,
                multigroup=True,
                K=3,
                mode=mode,
                max_iter=3,
                random_state=42,
            )
            model.fit(X, coordinates=coords, groups=groups)
            assert model.components_.shape == (2, 15)
            assert not np.isnan(model.elbo_)

    def test_fit_lcgp_custom_rank(self, lcgp_data):
        """Test LCGP with custom rank parameter."""
        X, coords, _ = lcgp_data
        model = PNMF(
            n_components=3,
            spatial=True,
            local=True,
            multigroup=False,
            K=3,
            rank=8,
            max_iter=5,
            random_state=42,
        )
        model.fit(X, coordinates=coords)
        assert model.components_.shape == (3, 20)

    def test_fit_lcgp_exp_low_rank_mode(self, lcgp_data):
        """Test LCGP with exp low_rank_mode."""
        X, coords, _ = lcgp_data
        model = PNMF(
            n_components=3,
            spatial=True,
            local=True,
            multigroup=False,
            K=3,
            low_rank_mode='exp',
            max_iter=5,
            random_state=42,
        )
        model.fit(X, coordinates=coords)
        assert model.components_.shape == (3, 20)

    def test_fit_lcgp_no_precompute_knn(self, lcgp_data):
        """Test LCGP with precompute_knn=False."""
        X, coords, _ = lcgp_data
        model = PNMF(
            n_components=3,
            spatial=True,
            local=True,
            multigroup=False,
            K=3,
            precompute_knn=False,
            max_iter=5,
            random_state=42,
        )
        model.fit(X, coordinates=coords)
        assert model.components_.shape == (3, 20)


class TestLCGPBatching:
    """Test LCGP with mini-batching."""

    def test_lcgp_with_sample_batching(self, lcgp_data):
        """Test LCGP with sample mini-batching."""
        X, coords, groups = lcgp_data
        model = PNMF(
            n_components=3,
            spatial=True,
            local=True,
            multigroup=True,
            K=3,
            max_iter=5,
            batch_size=15,
            random_state=42,
        )
        model.fit(X, coordinates=coords, groups=groups)
        assert model.components_.shape == (3, 20)

    def test_lcgp_with_feature_batching(self, lcgp_data):
        """Test LCGP with feature mini-batching."""
        X, coords, groups = lcgp_data
        model = PNMF(
            n_components=3,
            spatial=True,
            local=True,
            multigroup=True,
            K=3,
            max_iter=5,
            y_batch_size=10,
            random_state=42,
        )
        model.fit(X, coordinates=coords, groups=groups)
        assert model.components_.shape == (3, 20)

    def test_lcgp_with_both_batching(self, lcgp_data):
        """Test LCGP with both sample and feature batching."""
        X, coords, groups = lcgp_data
        model = PNMF(
            n_components=3,
            spatial=True,
            local=True,
            multigroup=True,
            K=3,
            max_iter=5,
            batch_size=15,
            y_batch_size=10,
            random_state=42,
        )
        model.fit(X, coordinates=coords, groups=groups)
        assert model.components_.shape == (3, 20)


class TestLCGPTransform:
    """Test transform and fit_transform for LCGP."""

    def test_transform_lcgp_no_groups(self, lcgp_data):
        """Test transform with LCGP without groups."""
        X, coords, _ = lcgp_data
        model = PNMF(
            n_components=3,
            spatial=True,
            local=True,
            multigroup=False,
            K=3,
            max_iter=5,
            random_state=42,
        )
        model.fit(X, coordinates=coords)

        # Transform with same coordinates
        transformed = model.transform(X, coordinates=coords)
        assert transformed.shape == (50, 3)
        assert np.all(transformed > 0)  # exp(F) should be positive

    def test_transform_lcgp_multigroup(self, lcgp_data):
        """Test transform with LCGP with groups."""
        X, coords, groups = lcgp_data
        model = PNMF(
            n_components=3,
            spatial=True,
            local=True,
            multigroup=True,
            K=3,
            max_iter=5,
            random_state=42,
        )
        model.fit(X, coordinates=coords, groups=groups)

        # Transform with same coordinates
        transformed = model.transform(X, coordinates=coords, groups=groups)
        assert transformed.shape == (50, 3)
        assert np.all(transformed > 0)

    def test_transform_new_coordinates_lcgp(self, lcgp_data):
        """Test transform at new coordinates with LCGP."""
        X, coords, groups = lcgp_data
        model = PNMF(
            n_components=3,
            spatial=True,
            local=True,
            multigroup=True,
            K=3,
            max_iter=5,
            random_state=42,
        )
        model.fit(X, coordinates=coords, groups=groups)

        # Transform with new coordinates
        np.random.seed(99)
        new_coords = np.random.rand(15, 2).astype(np.float32) * 50
        new_groups = np.random.randint(0, 3, 15)
        new_X = np.random.poisson(5, (15, 20)).astype(np.float32)

        transformed = model.transform(new_X, coordinates=new_coords, groups=new_groups)
        assert transformed.shape == (15, 3)
        assert np.all(transformed > 0)

    def test_fit_transform_lcgp(self, lcgp_data):
        """Test fit_transform with LCGP."""
        X, coords, groups = lcgp_data
        model = PNMF(
            n_components=3,
            spatial=True,
            local=True,
            multigroup=True,
            K=3,
            max_iter=5,
            random_state=42,
        )
        transformed = model.fit_transform(X, coordinates=coords, groups=groups)
        assert transformed.shape == (50, 3)
        assert np.all(transformed > 0)

    def test_transform_requires_coordinates_lcgp(self, lcgp_data):
        """Transform should require coordinates when using LCGP."""
        X, coords, groups = lcgp_data
        model = PNMF(
            n_components=3,
            spatial=True,
            local=True,
            multigroup=True,
            K=3,
            max_iter=5,
            random_state=42,
        )
        model.fit(X, coordinates=coords, groups=groups)

        with pytest.raises(ValueError, match="coordinates"):
            model.transform(X)


class TestLCGPFactorExtraction:
    """Test factor extraction functions with LCGP."""

    @pytest.fixture(autouse=True)
    def setup_model(self, small_lcgp_data):
        X, coords, groups = small_lcgp_data
        self.model = PNMF(
            n_components=2,
            spatial=True,
            local=True,
            multigroup=True,
            K=3,
            max_iter=5,
            random_state=42,
        )
        self.model.fit(X, coordinates=coords, groups=groups)
        self.X = X
        self.coords = coords
        self.groups = groups

    def test_log_factors_lcgp(self):
        F_log = log_factors(self.model, coordinates=self.coords, groups=self.groups)
        assert F_log.shape == (30, 2)

    def test_get_factors_lcgp(self):
        F_exp = get_factors(self.model, coordinates=self.coords, groups=self.groups)
        assert F_exp.shape == (30, 2)
        assert np.all(F_exp > 0)

    def test_get_factors_mgf_lcgp(self):
        F_exp = get_factors(
            self.model, use_mgf=True,
            coordinates=self.coords, groups=self.groups
        )
        assert F_exp.shape == (30, 2)
        assert np.all(F_exp > 0)

    def test_factor_uncertainty_lcgp(self):
        F_std = factor_uncertainty(
            self.model, coordinates=self.coords, groups=self.groups
        )
        assert F_std.shape == (30, 2)
        assert np.all(F_std >= 0)

    def test_factor_uncertainty_variance_lcgp(self):
        F_var = factor_uncertainty(
            self.model, return_variance=True,
            coordinates=self.coords, groups=self.groups
        )
        assert F_var.shape == (30, 2)
        assert np.all(F_var >= 0)

    def test_factor_samples_lcgp(self):
        samples = factor_samples(
            self.model, n_samples=10,
            coordinates=self.coords, groups=self.groups
        )
        assert samples.shape == (10, 30, 2)

    def test_factor_samples_exp_lcgp(self):
        samples = factor_samples(
            self.model, n_samples=10, return_exp=True,
            coordinates=self.coords, groups=self.groups
        )
        assert samples.shape == (10, 30, 2)
        assert np.all(samples > 0)

    def test_get_loadings_lcgp(self):
        W = get_loadings(self.model)
        assert W.shape == (15, 2)

    def test_log_factors_uses_stored_coords_lcgp(self):
        """log_factors should work without explicit coordinates."""
        F_log = log_factors(self.model)
        assert F_log.shape == (30, 2)


class TestLCGPNoGroups:
    """Test LCGP without groups (single-group)."""

    @pytest.fixture
    def no_group_data(self):
        """Spatial data without group labels."""
        np.random.seed(42)
        N, D = 40, 15
        X = np.random.poisson(5, (N, D)).astype(np.float32)
        coords = np.random.rand(N, 2).astype(np.float32) * 50
        return X, coords

    def test_fit_lcgp_no_groups(self, no_group_data):
        """Fit LCGP model without groups."""
        X, coords = no_group_data
        model = PNMF(
            n_components=2,
            spatial=True,
            local=True,
            multigroup=False,
            K=3,
            max_iter=5,
            random_state=42,
        )
        model.fit(X, coordinates=coords)

        assert model.components_.shape == (2, 15)
        assert model.elbo_ is not None
        assert not np.isnan(model.elbo_)

    def test_transform_lcgp_no_groups(self, no_group_data):
        """Transform with LCGP, no groups."""
        X, coords = no_group_data
        model = PNMF(
            n_components=2,
            spatial=True,
            local=True,
            multigroup=False,
            K=3,
            max_iter=5,
            random_state=42,
        )
        model.fit(X, coordinates=coords)

        # Transform at same coordinates
        transformed = model.transform(X, coordinates=coords)
        assert transformed.shape == (40, 2)
        assert np.all(transformed > 0)

    def test_transform_new_coords_lcgp_no_groups(self, no_group_data):
        """Transform at new coordinates, no groups."""
        X, coords = no_group_data
        model = PNMF(
            n_components=2,
            spatial=True,
            local=True,
            multigroup=False,
            K=3,
            max_iter=5,
            random_state=42,
        )
        model.fit(X, coordinates=coords)

        new_coords = np.random.randn(15, 2).astype(np.float32)
        new_X = np.random.poisson(5, (15, 15)).astype(np.float32)

        transformed = model.transform(new_X, coordinates=new_coords)
        assert transformed.shape == (15, 2)
        assert np.all(transformed > 0)

    def test_fit_transform_lcgp_no_groups(self, no_group_data):
        """fit_transform without groups."""
        X, coords = no_group_data
        model = PNMF(
            n_components=2,
            spatial=True,
            local=True,
            multigroup=False,
            K=3,
            max_iter=5,
            random_state=42,
        )
        transformed = model.fit_transform(X, coordinates=coords)
        assert transformed.shape == (40, 2)

    def test_elbo_modes_lcgp_no_groups(self, no_group_data):
        """All ELBO modes work without groups."""
        X, coords = no_group_data
        for mode in ['simple', 'expanded', 'lower-bound']:
            model = PNMF(
                n_components=2,
                spatial=True,
                local=True,
                multigroup=False,
                K=3,
                mode=mode,
                max_iter=3,
                random_state=42,
            )
            model.fit(X, coordinates=coords)
            assert model.components_.shape == (2, 15)
            assert not np.isnan(model.elbo_)

    def test_all_points_as_inducing_lcgp(self, no_group_data):
        """Verify LCGP uses all points as inducing points."""
        X, coords = no_group_data
        N = X.shape[0]
        model = PNMF(
            n_components=2,
            spatial=True,
            local=True,
            multigroup=False,
            K=3,
            max_iter=1,
            random_state=42,
        )
        model.fit(X, coordinates=coords)

        # LCGP should use all N points as inducing points
        Z = model._prior.Z.detach().cpu().numpy()
        assert Z.shape == (N, 2)
        # All inducing points should be original coordinates
        np.testing.assert_array_equal(Z, coords)

    def test_factor_extraction_lcgp_no_groups(self, no_group_data):
        """Factor extraction functions work without groups."""
        X, coords = no_group_data
        model = PNMF(
            n_components=2,
            spatial=True,
            local=True,
            multigroup=False,
            K=3,
            max_iter=5,
            random_state=42,
        )
        model.fit(X, coordinates=coords)

        F_log = log_factors(model, coordinates=coords)
        assert F_log.shape == (40, 2)

        F_exp = get_factors(model, coordinates=coords)
        assert F_exp.shape == (40, 2)
        assert np.all(F_exp > 0)

        F_std = factor_uncertainty(model, coordinates=coords)
        assert F_std.shape == (40, 2)
        assert np.all(F_std >= 0)
