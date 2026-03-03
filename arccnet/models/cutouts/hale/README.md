# Hale Classification (Cutouts)

This directory contains the Hale cutout classification pipeline (data prep, training, evaluation, and inference) built on PyTorch Lightning.

## Core Files

- `config.py`: training/data configuration
- `data_preparation.py`: preprocessing pipeline (cleanup, label mapping, NaN filtering, CV fold creation)
- `dataset.py`: `HaleDataset` and transforms
- `lightning_data.py`: `HaleDataModule`
- `lightning_model.py`: Lightning model (ResNet backbone)
- `trainer.py`: single-fold training/evaluation orchestration
- `cross_validation.py`: multi-fold orchestration and aggregation
- `train.py`: CLI entrypoint for training
- `evaluation.py`: confusion matrix, ROC, and misclassification logging
- `logging_utils.py`: logger setup and experiment logging helpers
- `inference.py`: model download + single FITS inference
- `test_setup.py`: consolidated environment/data setup verification

## Quick Start

1. Install dependencies:
   ```bash
   pip install -e .[models]
   ```
2. Verify setup:
   ```bash
   python arccnet/models/cutouts/hale/test_setup.py
   ```
3. Run training:
   ```bash
   python arccnet/models/cutouts/hale/train.py
   ```

## Dataset Output

Processed parquet includes:
- `grouped_labels`: mapped Hale classes used for training
- `model_labels`: contiguous integer labels (`0..N-1`)
- `Fold 1..N`: train/val/test assignments with AR-number group separation

## Notes

- Redundant helper scripts were consolidated into `test_setup.py`.
- Exploratory EDA script files were removed from this runtime directory to keep the training path focused and maintainable.
