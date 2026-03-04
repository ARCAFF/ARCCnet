# Timeseries Flare Forecasting

A complete pipeline for solar flare forecasting from SDO timeseries imagery using a CNN-Transformer hybrid architecture.

## Overview

This module implements a spatiotemporal deep learning approach for predicting solar flares from sequences of SDO/AIA and HMI observations.

**Architecture:**
1. **Spatial Encoder** (ResNet34-based CNN) - Extracts features from each timestep independently
2. **Temporal Transformer** - Processes the sequence of spatial features with multi-head self-attention
3. **Prediction Head** - Supports multiclass max-flare prediction (default: No-flare/C/M+) or regression of log flare counts

**Dataset:**
- Input: 6 timesteps (hourly cadence) × 10 channels (9 AIA wavelengths + 1 HMI magnetogram)
- Spatial resolution: 400×800 pixels (resized to 256×512 for training)
- Labels:
  - Multiclass (default): highest flare class in next 24h (`0=No-flare`, `1=C`, `2=M+` where `M+` includes X)
  - Regression: `[log10(Ca+1), log10(Ma+1), log10(Xa+1)]`

## Directory Structure

```
arccnet/models/timeseries/
├── __init__.py              # Module exports
├── config.py                # Hyperparameters and configuration
├── manifest.py              # Dataset manifest builder
├── splitters.py             # Data splitting strategies
├── dataset.py               # PyTorch Dataset class
├── spatial_encoder.py       # CNN spatial feature extractor
├── temporal_transformer.py  # Transformer temporal encoder
├── flare_forecaster.py      # Complete forecasting model
├── train.py                 # Training script
├── pit_train.py             # PIT wrapper (single-timestep training)
├── evaluate.py              # Evaluation script
├── README.md                # This file
└── tests/                   # Test suite
    ├── __init__.py
    ├── test_dataset_build.py
    ├── test_dataset_shapes.py
    └── test_model_forward.py
```

## Quick Start

### 1. Train Model

Train the forecasting model:

```bash
python -m arccnet.models.timeseries.train \
    --data_root /ARCAFF/data/timeseries/04_final/data \
    --manifest_path /ARCAFF/ARCCnet/outputs/timeseries/manifest.parq \
    --output_dir /ARCAFF/ARCCnet/outputs/timeseries/run_001
```

`train.py` rebuilds the manifest from `--data_root` on every run and writes it to `--manifest_path`.

### 1b. Train Point-in-Time (PIT) Model

Run PIT training (single timestep from the end of each sample, temporal transformer disabled):

```bash
python -m arccnet.models.timeseries.pit_train \
    --data_root /ARCAFF/data/timeseries/04_final/data \
    --manifest_path /ARCAFF/ARCCnet/outputs/timeseries/manifest_pit.parq \
    --output_dir /ARCAFF/ARCCnet/outputs/timeseries/pit_run_001
```

`pit_train.py` is a thin wrapper around `train.py` that forwards:
- `--num_timesteps 1`
- `--timestep_selection last`
- `--use_temporal_transformer false`

You can still pass any regular `train.py` arguments to the PIT wrapper.

Training outputs:
- `.../best-epoch-metric.ckpt` - Best Lightning checkpoint (highest validation primary metric)
- `.../last.ckpt` - Latest checkpoint
- `manifest.parq` (or your `--manifest_path`) - Manifest generated for this run
- `norm_stats.json` - Normalization statistics
- `tensorboard/` - TensorBoard logs
- `training_summary.json` - Final training summary

### 2. Evaluate Model

Evaluate on test set:

```bash
python -m arccnet.models.timeseries.evaluate \
    --checkpoint_path /ARCAFF/data/checkpoints/timeseries/multiclass/<run>/best-*.ckpt \
    --manifest_path /ARCAFF/ARCCnet/outputs/timeseries/manifest.parq \
    --split_assignments_path /ARCAFF/ARCCnet/outputs/timeseries/run_001/split_assignments.parquet \
    --split test \
    --output_dir /ARCAFF/ARCCnet/outputs/timeseries/eval \
    --save_predictions
```

Evaluation outputs:
- `metrics_test.json` - Comprehensive metrics (TSS, ROC-AUC, PR-AUC per class)
- `predictions_test.csv` - Per-sample predictions and labels

### 3. Run Tests

Verify the pipeline works correctly:

```bash
# Test manifest building
python arccnet/models/timeseries/tests/test_dataset_build.py

# Test dataset shapes
python arccnet/models/timeseries/tests/test_dataset_shapes.py

# Test model forward pass
python arccnet/models/timeseries/tests/test_model_forward.py
```

## Data Format

### Directory Structure

Each sample is stored in a directory with the following naming convention:
```
{YYYY-MM-DD}_{NOAA}_{MagClass}_{McIntosh}_Xb{#}_Mb{#}_Cb{#}_Xa{#}_Ma{#}_Ca{#}/
```

Example: `2011-01-03_11142_Beta_Dso_Xb0_Mb0_Cb0_Xa0_Ma0_Ca1/`

Where:
- `YYYY-MM-DD`: Date of observation
- `NOAA`: NOAA Active Region number
- `MagClass`: Hale magnetic classification (e.g., Beta, Beta-Gamma)
- `McIntosh`: McIntosh classification (e.g., Dso)
- `Xb#, Mb#, Cb#`: X, M, C class flares in 6h **before** observation
- `Xa#, Ma#, Ca#`: X, M, C class flares in 24h **after** observation (targets)

### CSV Format

Each sample directory contains a CSV file with 60 rows (10 wavelengths × 6 timesteps):

| Column | Description |
|--------|-------------|
| `AIA wavelength` | Wavelength identifier (94, 131, 171, 193, 211, 304, 335, 1600, 1700 for AIA; 6173 for HMI) |
| `AIA files` | Path to AIA FITS file (or HMI path if wavelength=6173) |
| `AIA quality` | Data quality flag |
| `HMI files` | Path to HMI FITS file |
| `HMI quality` | HMI quality flag |

The CSV is organized with all wavelengths for timestep 0, then all wavelengths for timestep 1, etc.

### FITS Files

- **AIA**: 9 EUV wavelength channels (94, 131, 171, 193, 211, 304, 335, 1600, 1700 Å)
- **HMI**: Line-of-sight magnetogram (6173 Å)
- **Spatial**: 400×800 pixels (resized to 256×512 for training)
- **Temporal**: 6 hourly observations

## Model Architecture

### Spatial Encoder (`spatial_encoder.py`)

- **Backbone**: ResNet34 (ImageNet pretrained)
- **Input Adaptation**: First conv layer adapted for 10 channels via weight replication
- **Output**: 512-dimensional feature vector per timestep

### Temporal Transformer (`temporal_transformer.py`)

- **Type**: Transformer Encoder with positional encoding
- **Layers**: 4 transformer blocks
- **Heads**: 8 attention heads
- **Feedforward**: 2048 dimensions
- **Pooling**: Mean pooling over temporal dimension (alternatives: CLS token, last timestep)

### Classification Head (`flare_forecaster.py`)

- **Input**: 512-dimensional temporal feature
- **Hidden**: [256] with ReLU and 0.3 dropout
- **Output**:
  - Multiclass: `NUM_CLASSES` logits (default 3 for No-flare/C/M+ classes, CrossEntropyLoss)
  - Regression: 3 values for log flare-count targets (MSELoss)

## Configuration

Key hyperparameters in `config.py`:

```python
# Data
NUM_CHANNELS = 10           # 9 AIA + 1 HMI
RESIZE = (256, 512)         # Spatial resize
SPLIT_STRATEGY = 'noaa'     # or 'time'

# Model
SPATIAL_FEATURE_DIM = 512
TEMPORAL_NUM_LAYERS = 4
TEMPORAL_NUM_HEADS = 8
TEMPORAL_POOLING = 'mean'
PRETRAINED_SPATIAL = True

# Training
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 50

# Augmentation
HFLIP_PROB = 0.5
VFLIP_PROB = 0.5
ROTATION_DEGREES = 10
```

## Data Splitting

Two splitting strategies are available to prevent data leakage:

### 1. NOAA-based (Recommended)
Groups samples by NOAA Active Region number to ensure the same AR never appears in multiple splits:
```python
split_data = get_split(manifest, strategy='noaa', train_frac=0.7, val_frac=0.15, seed=42)
train_df = split_data['train_df']
```

### 2. Time-based
Splits by year for temporal validation (simulates operational deployment):
```python
split_data = get_split(
    manifest,
    strategy='time',
    train_years=[2011, 2017, 2018, 2019, 2020],
    val_years=[2021],
    test_years=[2022]
)
```

## Normalization

Per-channel normalization with percentile clipping:
1. **Training**: Compute mean/std from subset of training data (50 samples)
2. **Per-channel**: Each wavelength normalized independently
3. **Clipping**: 1st-99th percentile clipping before standardization
4. **Inference**: Use training statistics for val/test splits

Statistics are saved in `norm_stats.json` for reproducibility.

## Metrics

The pipeline computes comprehensive evaluation metrics:

- **TSS (True Skill Statistic)**: Primary metric for solar flare forecasting
  - TSS = TPR - FPR
  - Range: [-1, 1], higher is better

- **ROC-AUC**: Area under ROC curve (threshold-independent)

- **PR-AUC**: Area under precision-recall curve (better for imbalanced data)

- **TPR/FPR**: True/False Positive Rate at threshold=0.5

Metrics are computed per class and include operational `M+` scores (and `X+` when an explicit X class exists).

## Training Tips

1. **Class Imbalance**: Consider class weighting in loss function for minority classes
2. **Overfitting**: Monitor train/val gap; consider stronger regularization
3. **Convergence**: TSS may fluctuate; save best validation checkpoint
4. **GPU Memory**: Reduce batch size or spatial resolution if OOM
5. **Data Variance**: Ensure normalization stats are computed on diverse samples

## API Usage

```python
from arccnet.models.timeseries import (
    build_dataset,
    get_split,
    SDOTimeseriesDataset,
    FlareForecaster,
)

# Build manifest
manifest = build_dataset('/path/to/data')

# Split data
split_data = get_split(manifest, strategy='noaa')

# Create dataset
dataset = SDOTimeseriesDataset(split_data['train_df'], split='train', augment=True)

# Create model
model = FlareForecaster(
    num_channels=10,
    temporal_num_layers=4,
    pretrained_spatial=True,
)

# Forward pass
import torch
x = torch.randn(2, 6, 10, 256, 512)  # (B, T, C, H, W)
logits = model(x)  # (B, NUM_CLASSES) for multiclass
probs = model.predict_proba(x)  # (B, NUM_CLASSES) in [0, 1]
```

## Troubleshooting

**Q: CUDA out of memory**
- Reduce `BATCH_SIZE` in `config.py`
- Reduce `RESIZE` dimensions
- Set `FREEZE_SPATIAL=True` to reduce memory

**Q: Poor performance on minority classes**
- Use time-based split for better temporal coverage
- Apply class weights in loss function
- Adjust classification threshold per class

**Q: Normalization errors**
- Ensure FITS files are valid and readable
- Check for corrupted files (NaN/Inf values)
- Verify CSV paths match actual file locations

**Q: Slow data loading**
- Increase `NUM_WORKERS` (but not > CPU cores)
- Place `--data_root` on fast local storage to speed up manifest rebuilds
- Consider caching normalized data to disk

## Citation

If you use this code, please cite:

```
TBD - Add citation when published
```

## License

See LICENSE.rst in repository root.
