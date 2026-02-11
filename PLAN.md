# Plan: No-Groups SVGP (Branch: no-groups)

## Goal

Implement SVGP spatial prior **without groups** using `batched_Matern32` kernel and `SVGP` class from GPzoo. This removes the requirement for group labels, making the spatial mode usable for single-tissue/single-condition spatial data.

## Current State

`_create_spatial_prior()` in `PNMF/models.py:573-668` already has a non-MGGP path, but:
- **Line 622-626**: Uses random permutation for inducing points (poor quality)
- **Line 383**: `multigroup: bool = True` default forces users to opt out
- The non-MGGP path (`lines 608-612, 622-626, 637-643`) uses `batched_Matern32` and `SVGP` correctly

## Changes

### 1. K-means inducing points for non-MGGP

**File**: `PNMF/models.py:622-626`

Replace random permutation with K-means clustering:
```python
# Current:
perm = torch.randperm(N)[:M]
Z = coordinates[perm].clone()
groupsZ = None

# New:
from sklearn.cluster import MiniBatchKMeans
kmeans = MiniBatchKMeans(n_clusters=M, random_state=self.random_state or 123, n_init=3)
kmeans.fit(coordinates.cpu().numpy())
Z = torch.from_numpy(kmeans.cluster_centers_.astype(np.float32)).to(coordinates.device)
groupsZ = None
```

### 2. Change `multigroup` default to `False`

**File**: `PNMF/models.py:383`

```python
multigroup: bool = False  # was True
```

### 3. Update docstring

**File**: `PNMF/models.py:284-285`

Update `multigroup` docstring to reflect new default.

### 4. Add tests

**File**: `tests/test_spatial.py`

Add test class `TestSpatialNoGroups` covering:
- `test_fit_no_groups` — spatial fit with `multigroup=False`, no groups arg
- `test_transform_no_groups` — transform at new coordinates without groups
- `test_fit_transform_no_groups` — combined fit_transform
- `test_kmeans_inducing_points` — verify inducing points are K-means centroids (not random)

## Files Modified

| File | Lines | Change |
|------|-------|--------|
| `PNMF/models.py` | 383 | Default `multigroup=False` |
| `PNMF/models.py` | 284-285 | Update docstring |
| `PNMF/models.py` | 622-626 | K-means inducing points |
| `tests/test_spatial.py` | new class | No-groups test cases |
