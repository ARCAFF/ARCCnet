#!/usr/bin/env python3
"""Quick test script to verify Lightning setup."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

try:
    import importlib.util

    if importlib.util.find_spec("pytorch_lightning") is None:
        raise ImportError("pytorch_lightning not available")
    if importlib.util.find_spec("torch") is None:
        raise ImportError("torch not available")
    if importlib.util.find_spec("torchmetrics") is None:
        raise ImportError("torchmetrics not available")

    logging.info("✓ PyTorch and Lightning imports successful")
except ImportError as e:
    logging.error(f"✗ Import error: {e}")
    exit(1)

try:
    from sklearn.utils.class_weight import compute_class_weight

    import arccnet.models.cutouts.hale.config as config
    from arccnet.models.cutouts.hale.dataset import get_fold_data
    from arccnet.models.cutouts.hale.lightning_data import HaleDataModule
    from arccnet.models.cutouts.hale.lightning_model import HaleLightningModel

    logging.info("✓ Local imports successful")
except ImportError as e:
    logging.error(f"✗ Local import error: {e}")
    exit(1)


def test_model_creation():
    """Test model creation."""
    try:
        HaleLightningModel(
            num_classes=config.NUM_CLASSES,
            learning_rate=config.LEARNING_RATE,
            model_name=config.MODEL_NAME,
        )
        logging.info(f"✓ Model created successfully: {config.MODEL_NAME}")
        logging.info(f"  - Number of classes: {config.NUM_CLASSES}")
        logging.info(f"  - Learning rate: {config.LEARNING_RATE}")
        return True
    except Exception as e:
        logging.error(f"✗ Model creation failed: {e}")
        return False


def test_data_loading():
    """Test data loading."""
    try:
        data_path = Path(config.DATA_FOLDER) / config.PROCESSED_DATASET_FILENAME

        if not data_path.exists():
            logging.warning(f"✗ Processed data not found at: {data_path}")
            return False

        df = pd.read_parquet(data_path)
        logging.info(f"✓ Dataset loaded: {df.shape}")

        train_df, val_df, test_df = get_fold_data(df, fold_num=1)
        logging.info(f"✓ Fold 1 split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

        data_module = HaleDataModule(df=df, fold_num=1)
        data_module.setup("fit")
        train_labels = data_module.get_train_labels()
        unique_labels = np.unique(train_labels)
        class_weights = compute_class_weight("balanced", classes=unique_labels, y=train_labels)

        logging.info("✓ DataModule created successfully")
        logging.info(f"✓ Original labels in training: {np.unique(train_labels)}")
        logging.info(f"✓ Class weights computed: {class_weights}")

        return True
    except Exception as e:
        logging.error(f"✗ Data loading failed: {e}")
        return False


def test_raw_dataset_access():
    """Test access to raw parquet source data."""
    try:
        raw_data_path = Path(config.DATA_FOLDER) / config.DATASET_FOLDER / config.DF_FILE_NAME
        if not raw_data_path.exists():
            logging.warning(f"✗ Raw dataset file not found at: {raw_data_path}")
            return False

        try:
            df_raw = pd.read_parquet(raw_data_path, columns=["region_type", "magnetic_class"])
        except Exception:
            df_raw = pd.read_parquet(raw_data_path)

        logging.info(f"✓ Raw dataset loaded: {df_raw.shape}")
        return True
    except Exception as e:
        logging.error(f"✗ Raw dataset access failed: {e}")
        return False


def check_dependencies():
    """Check if required dependencies are available."""
    missing_deps = []

    try:
        import importlib.util

        deps = ["pytorch_lightning", "torchmetrics", "torchvision"]
        for dep in deps:
            if importlib.util.find_spec(dep) is None:
                missing_deps.append(f"No module named '{dep}'")

    except ImportError as e:
        missing_deps.append(str(e))

    return missing_deps


def main():
    """Run all tests."""
    logging.info("Starting Lightning setup verification...")

    missing_deps = check_dependencies()

    tests = [
        ("Raw Dataset Access", test_raw_dataset_access),
        ("Model Creation", test_model_creation),
        ("Data Loading", test_data_loading),
    ]

    results = []
    for test_name, test_func in tests:
        logging.info(f"\n--- Testing {test_name} ---")
        result = test_func()
        results.append(result)

    passed = sum(results)
    total = len(results)

    logging.info(f"\n{'=' * 50}")
    logging.info(f"TEST SUMMARY: {passed}/{total} tests passed")

    if passed == total:
        logging.info("✓ All tests passed! Lightning setup is ready.")
        logging.info("\nNext steps:")

        if missing_deps:
            logging.info(f"Missing dependencies: {', '.join(missing_deps)}")
        else:
            logging.info("✓ All dependencies are already installed!")
    else:
        logging.error("✗ Some tests failed. Please check the setup.")
        exit(1)


if __name__ == "__main__":
    main()
