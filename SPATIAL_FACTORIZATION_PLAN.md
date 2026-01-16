# Spatial Factorization Repo Plan

## Overview

Create a new repository `spatial-factorization` (or `genomic-factorization`) that:
1. Houses all dataset-specific analysis code
2. Uses config-driven training (no registry)
3. Imports models from `gpzoo` and `pnmf` as dependencies

This allows GPzoo to be a clean backend library and PNMF to be a clean sklearn interface.

---

## Phase 1: Create `spatial-factorization` Repo Structure

### Repository Layout

```
spatial-factorization/
├── pyproject.toml              # Dependencies: gpzoo, pnmf, scanpy, squidpy
├── README.md
├── LICENSE
│
├── configs/                    # YAML configs for all experiments
│   ├── slideseq/
│   │   ├── svgp.yaml
│   │   ├── svgp_mggp.yaml
│   │   ├── vnngp.yaml
│   │   ├── vnngp_mggp.yaml
│   │   ├── lcgp.yaml
│   │   └── lcgp_mggp.yaml
│   ├── tenxvisium/
│   │   ├── vnngp.yaml
│   │   └── vnngp_mggp.yaml
│   └── liver/
│       ├── vnngp.yaml
│       └── vnngp_mggp.yaml
│
├── src/
│   └── spatial_factorization/
│       ├── __init__.py
│       ├── config.py           # Config loading/validation (pydantic or dataclasses)
│       ├── datasets/           # Data loaders (one per dataset)
│       │   ├── __init__.py
│       │   ├── base.py         # Abstract base class for datasets
│       │   ├── slideseq.py     # SlideseqV2 loader
│       │   ├── tenxvisium.py   # 10x Visium loader
│       │   └── liver.py        # Liver dataset loader
│       ├── models/             # Model factory (thin wrapper, no registry)
│       │   ├── __init__.py
│       │   └── factory.py      # build_model(config) -> nn.Module
│       ├── training/           # Generic training utilities
│       │   ├── __init__.py
│       │   ├── trainer.py      # Main training loop
│       │   ├── callbacks.py    # Checkpointing, logging, early stopping
│       │   └── schedulers.py   # LR scheduler utilities
│       └── cli.py              # Command-line interface
│
├── notebooks/                  # Analysis notebooks
│   ├── slideseq_analysis.ipynb
│   ├── tenxvisium_analysis.ipynb
│   └── comparison_plots.ipynb
│
├── scripts/                    # Convenience scripts
│   ├── train.py                # Entry point: python scripts/train.py --config ...
│   ├── run_slideseq_all.sh     # Run all slideseq models
│   └── run_benchmarks.sh
│
└── outputs/                    # Git-ignored, stores checkpoints/logs
    └── .gitkeep
```

### Config Schema (Example)

```yaml
# configs/slideseq/lcgp.yaml

# Metadata
name: slideseq_lcgp
seed: 67

# Dataset
dataset:
  name: slideseq
  spatial_scale: 50.0
  filter_mt: true
  min_counts: 100
  min_cells: 10

# Model
model:
  type: lcgp_nsf          # Maps to gpzoo.models.LCGP_NSF
  n_components: 10
  lengthscale: 4.0
  sigma: 1.0
  jitter: 1e-5
  loadings_mode: projected
  # LCGP-specific
  K: 50
  rank: 55
  diag_mode: softplus
  precompute_knn: true
  scale_multiplier: 1.0

# Training
training:
  steps: 10000
  batch_size_x: 34000
  batch_size_y: 1000
  optimizer: adam
  learning_rates:
    default: 0.01
    mean: 0.01
    loading: 0.001
    scale: 0.01
    lengthscale: 0.0001
  freeze:
    kernel: true
    lengthscale_until: 10000   # Never unfreeze (steps == total)
    scale_until: 1
  scheduler:
    enabled: false
    warmup_fraction: 0.2
    min_lr: 3e-5

# Logging
logging:
  tensorboard: true
  image_log_every: 100
  checkpoint_every: 1000

# Output
output:
  dir: outputs/slideseq/lcgp
```

---

## Phase 2: Implement Core Components

### 2.1 Config System (`src/spatial_factorization/config.py`)

```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any
import yaml

@dataclass
class DatasetConfig:
    name: str
    spatial_scale: float = 50.0
    filter_mt: bool = True
    min_counts: int = 100
    min_cells: int = 10

@dataclass
class ModelConfig:
    type: str                           # e.g., "lcgp_nsf", "svgp_nsf"
    n_components: int = 10
    lengthscale: float = 4.0
    sigma: float = 1.0
    jitter: float = 1e-5
    loadings_mode: str = "projected"
    # Optional model-specific
    K: Optional[int] = None             # For VNNGP/LCGP
    rank: Optional[int] = None          # For LCGP
    num_inducing: Optional[int] = None  # For SVGP
    # ... other optional params

@dataclass
class TrainingConfig:
    steps: int = 10000
    batch_size_x: int = 34000
    batch_size_y: int = 1000
    optimizer: str = "adam"
    learning_rates: Dict[str, float] = field(default_factory=lambda: {"default": 0.01})
    # ...

@dataclass
class Config:
    name: str
    seed: int
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
    # ...

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(...)  # Parse nested dataclasses
```

### 2.2 Dataset Loaders (`src/spatial_factorization/datasets/`)

```python
# datasets/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch

@dataclass
class SpatialData:
    X: torch.Tensor          # Spatial coordinates (N, 2)
    Y: torch.Tensor          # Count matrix (D, N) - genes x spots
    V: torch.Tensor          # Size factors (N,)
    groups: Optional[torch.Tensor] = None  # For MGGP
    n_groups: int = 0
    gene_names: Optional[list] = None

class DatasetLoader(ABC):
    @abstractmethod
    def load(self, config: DatasetConfig) -> SpatialData:
        pass

# datasets/slideseq.py
class SlideseqLoader(DatasetLoader):
    def load(self, config: DatasetConfig) -> SpatialData:
        import squidpy as sq
        import scanpy as sc

        adata = sq.datasets.slideseqv2()
        adata = adata.raw.to_adata()
        # ... filtering logic from current common.py ...
        return SpatialData(X=X_t, Y=Y_t, V=V_t, groups=groups_t, n_groups=n_groups)

# datasets/__init__.py
LOADERS = {
    "slideseq": SlideseqLoader,
    "tenxvisium": TenxVisiumLoader,
    "liver": LiverLoader,
}

def load_dataset(config: DatasetConfig) -> SpatialData:
    loader_cls = LOADERS[config.name]
    return loader_cls().load(config)
```

### 2.3 Model Factory (`src/spatial_factorization/models/factory.py`)

No registry - just a simple factory function:

```python
from gpzoo.models import SVGP_NSF, VNNGP_NSF, LCGP_NSF
from gpzoo.models import SVGP_MGGP_NSF, VNNGP_MGGP_NSF, LCGP_MGGP_NSF

MODEL_CLASSES = {
    "svgp_nsf": SVGP_NSF,
    "vnngp_nsf": VNNGP_NSF,
    "lcgp_nsf": LCGP_NSF,
    "svgp_mggp_nsf": SVGP_MGGP_NSF,
    "vnngp_mggp_nsf": VNNGP_MGGP_NSF,
    "lcgp_mggp_nsf": LCGP_MGGP_NSF,
}

def build_model(config: ModelConfig, data: SpatialData, device: torch.device):
    """Build model from config - no registry, no create_model functions."""
    model_cls = MODEL_CLASSES[config.type]

    # Build kwargs from config
    kwargs = {
        "X": data.X,
        "Y": data.Y,
        "V": data.V,
        "L": config.n_components,
        "lengthscale": config.lengthscale,
        "sigma": config.sigma,
        "jitter": config.jitter,
        "loadings_mode": config.loadings_mode,
        "device": device,
    }

    # Add model-specific kwargs
    if config.K is not None:
        kwargs["K"] = config.K
    if config.rank is not None:
        kwargs["rank"] = config.rank
    if config.num_inducing is not None:
        kwargs["num_inducing"] = config.num_inducing
    # ...

    return model_cls(**kwargs)
```

### 2.4 Generic Trainer (`src/spatial_factorization/training/trainer.py`)

Single training function that works for all models:

```python
def train(
    model: nn.Module,
    data: SpatialData,
    config: TrainingConfig,
    output_dir: Path,
    writer: Optional[SummaryWriter] = None,
):
    """Generic training loop for all spatial factorization models."""

    # Setup optimizer with param groups
    param_groups = build_param_groups(model, config.learning_rates)
    optimizer = build_optimizer(config.optimizer, param_groups)
    scheduler = build_scheduler(optimizer, config) if config.scheduler.enabled else None

    # Determine training function based on model type
    train_fn = get_training_fn(model)  # e.g., train_svgp_batched, train_lcgp_batched

    # Run training
    losses = train_fn(
        model=model,
        optimizer=optimizer,
        X=data.X,
        y=data.Y,
        steps=config.steps,
        x_batch_size=config.batch_size_x,
        y_batch_size=config.batch_size_y,
        writer=writer,
        scheduler=scheduler,
        # ... unfreeze schedules from config
    )

    # Save checkpoint
    torch.save(model.state_dict(), output_dir / "checkpoint.pth")
    save_losses(losses, output_dir / "losses.csv")

    return losses
```

### 2.5 CLI Entry Point (`scripts/train.py`)

```python
#!/usr/bin/env python
"""Train a spatial factorization model from a config file."""

import argparse
from pathlib import Path
import torch

from spatial_factorization.config import Config
from spatial_factorization.datasets import load_dataset
from spatial_factorization.models import build_model
from spatial_factorization.training import train

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--device", default=None, help="Override device")
    parser.add_argument("--checkpoint", default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    # Load config
    config = Config.from_yaml(args.config)

    # Setup
    torch.manual_seed(config.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(config.output.dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    data = load_dataset(config.dataset)

    # Build model
    model = build_model(config.model, data, device)

    # Load checkpoint if resuming
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    # Train
    train(model, data, config.training, output_dir)

    print(f"Training complete. Outputs saved to {output_dir}")

if __name__ == "__main__":
    main()
```

---

## Phase 3: Clean Up GPzoo

### Files to Remove from GPzoo

```
gpzoo/
├── datasets/                   # REMOVE ENTIRE FOLDER
│   ├── slideseq/              # Move loaders to spatial-factorization
│   ├── tenxvisium/
│   └── liver/
├── models/
│   └── registry.py            # REMOVE (no longer needed)
```

### Files to Keep in GPzoo (Backend Only)

```
gpzoo/
├── gp.py                      # GaussianPrior, SVGP, VNNGP, LCGP
├── modules.py                 # PositiveParameter, CholeskyParameter, etc.
├── kernels.py                 # RBF, Matern, etc.
├── likelihoods.py             # PoissonFactorization base class
├── models/
│   ├── __init__.py            # Export model classes
│   └── nsf.py                 # SVGP_NSF, VNNGP_NSF, LCGP_NSF, *_MGGP_NSF
├── training_utilities.py      # Keep training functions (used by spatial-factorization)
└── utilities.py               # General utilities
```

### Update GPzoo's `__init__.py`

Remove dataset imports, keep only backend exports:
```python
from .gp import GaussianPrior, SVGP, VNNGP, LCGP
from .modules import PositiveParameter, CholeskyParameter
from .likelihoods import PoissonFactorization
from .models import SVGP_NSF, VNNGP_NSF, LCGP_NSF  # etc.
```

---

## Phase 4: Update PNMF

Per the existing PLANNING.md, PNMF should:
1. Import from GPzoo: `from gpzoo.gp import GaussianPrior`
2. Keep its own ELBO computation (`PNMF/elbo.py`)
3. Provide sklearn-compatible API

No changes needed for spatial-factorization - PNMF is already well-scoped.

---

## Migration Checklist

### Step 1: Create spatial-factorization repo
- [ ] Initialize repo with pyproject.toml
- [ ] Add dependencies: gpzoo, pnmf, scanpy, squidpy, torch, pyyaml
- [ ] Create folder structure

### Step 2: Port dataset loaders
- [ ] Create `datasets/base.py` with SpatialData class
- [ ] Port `slideseq/common.py` → `datasets/slideseq.py`
- [ ] Port `tenxvisium/common.py` → `datasets/tenxvisium.py`
- [ ] Port `liver/common.py` → `datasets/liver.py`

### Step 3: Implement config system
- [ ] Create dataclass schema in `config.py`
- [ ] Create YAML configs for each dataset/model combo
- [ ] Add config validation

### Step 4: Implement model factory
- [ ] Create `models/factory.py` with build_model()
- [ ] Test instantiation for all model types

### Step 5: Implement generic trainer
- [ ] Create `training/trainer.py`
- [ ] Port param group building from GPzoo
- [ ] Test training loop

### Step 6: Create CLI
- [ ] Implement `scripts/train.py`
- [ ] Test end-to-end: config → data → model → train

### Step 7: Clean up GPzoo
- [ ] Remove `gpzoo/datasets/` folder
- [ ] Remove `gpzoo/models/registry.py`
- [ ] Update `gpzoo/__init__.py`
- [ ] Bump version, release

### Step 8: Documentation
- [ ] README for spatial-factorization
- [ ] Example notebooks
- [ ] Update GPzoo docs to remove dataset references

---

## Questions to Resolve

1. **Repo name**: `spatial-factorization` vs `genomic-factorization` vs something else?
   - `spatial-factorization` is more specific to the spatial transcriptomics use case
   - `genomic-factorization` is broader but might be misleading

2. **Training functions location**: Keep in GPzoo or move to spatial-factorization?
   - Option A: Keep `train_svgp_batched_with_tracking` etc. in GPzoo (current)
   - Option B: Move to spatial-factorization (cleaner separation)
   - Recommendation: Keep in GPzoo since they're model-specific, not dataset-specific

3. **Config format**: YAML vs TOML vs Python dataclasses?
   - YAML is standard for ML configs (Hydra, PyTorch Lightning)
   - Could add Hydra support later for advanced config composition

4. **MGGP handling**: How to configure group information?
   - Groups come from dataset (cluster labels)
   - Config just enables/disables MGGP mode
   - Model factory handles group tensor if MGGP model type

---

## Timeline Estimate

| Phase | Description | Complexity |
|-------|-------------|------------|
| 1 | Create repo structure | Low |
| 2 | Implement core components | Medium |
| 3 | Clean up GPzoo | Low |
| 4 | Testing & documentation | Medium |

---

## Benefits Summary

1. **GPzoo becomes a clean backend** (~1500 lines removed)
2. **No more registry** - configs are the source of truth
3. **Easy to add datasets** - just create loader + config
4. **Reproducible experiments** - configs capture everything
5. **Better for papers** - can share configs alongside results
6. **Notebook-friendly** - analysis separate from training code
