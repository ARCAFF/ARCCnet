"""Shared dataset/datamodule components for flare cutout classifiers."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from arccnet.models import preprocessing_common as pp_common
from arccnet.models.flares import split_cache_utils as cache_utils
from arccnet.visualisation import utils as ut_v

logger = logging.getLogger(__name__)


class FlareDataset(Dataset):
    """PyTorch Dataset for flare cutout FITS images."""

    def __init__(
        self,
        data_folder,
        dataset_folder,
        df,
        target_column,
        target_height=224,
        target_width=224,
        divisor=800.0,
        min_val=-1.0,
        max_val=1.0,
        transform=None,
    ):
        self.data_folder = data_folder
        self.dataset_folder = dataset_folder
        self.df = df
        self.target_column = target_column
        self.transform = transform
        self.target_height = target_height
        self.target_width = target_width
        self.divisor = divisor
        self.min_val = min_val
        self.max_val = max_val

        required_cols = {"path_image_cutout_hmi", "path_image_cutout_mdi", self.target_column}
        missing_cols = required_cols.difference(df.columns)
        if missing_cols:
            raise ValueError(f"DataFrame missing required columns: {sorted(missing_cols)}")

    def _load_image(self, row):
        fits_path = pp_common.resolve_preferred_cutout_fits_path(
            row,
            data_folder=self.data_folder,
            dataset_folder=self.dataset_folder,
        )
        if fits_path is None:
            raise FileNotFoundError(f"No valid cutout FITS path found for row index {row.name}")

        try:
            image_data = pp_common.load_fits_hdu_data(fits_path, hdu_index=1, dtype=np.float32)
        except Exception as exc:
            raise OSError(f"Error reading FITS file {fits_path} (index: {row.name}): {exc}") from exc

        image_data = np.nan_to_num(image_data, nan=0.0)
        image_data = ut_v.hardtanh_transform_npy(
            image_data,
            divisor=self.divisor,
            min_val=self.min_val,
            max_val=self.max_val,
        )
        image_data = ut_v.pad_resize_normalize(
            image_data,
            target_height=self.target_height,
            target_width=self.target_width,
        )

        image_tensor = torch.from_numpy(image_data).unsqueeze(0)
        label = int(row[self.target_column])

        return image_tensor, label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if idx >= len(self.df):
            raise IndexError("Index out of range")
        row = self.df.iloc[idx]
        image, label = self._load_image(row)

        if self.transform:
            image = self.transform(image)

        return image, label


class FlareDataModule(pl.LightningDataModule):
    """Shared Lightning DataModule for flare cutout training."""

    def __init__(
        self,
        data_folder,
        dataset_folder,
        train_df=None,
        val_df=None,
        test_df=None,
        target_column=None,
        df_flares_name=None,
        test_size=0.1,
        val_size=0.2,
        random_state=None,
        apply_quality_filter=True,
        apply_path_filter=True,
        apply_longitude_filter=True,
        apply_nan_filter=False,
        max_longitude=65.0,
        nan_threshold=0.05,
        split_cache_dir=None,
        split_cache_name=None,
        batch_size=32,
        num_workers=4,
        img_target_height=224,
        img_target_width=224,
        img_divisor=800.0,
        img_min_val=-1.0,
        img_max_val=1.0,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=1,
        multiprocessing_context=None,
        train_transform=None,
        val_test_transform=None,
    ):
        super().__init__()
        self.data_folder = data_folder
        self.dataset_folder = dataset_folder
        if target_column is None:
            raise ValueError("target_column must be provided.")
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.target_column = target_column
        self.df_flares_name = df_flares_name
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.apply_quality_filter = apply_quality_filter
        self.apply_path_filter = apply_path_filter
        self.apply_longitude_filter = apply_longitude_filter
        self.apply_nan_filter = apply_nan_filter
        self.max_longitude = max_longitude
        self.nan_threshold = nan_threshold
        self.split_cache_dir = split_cache_dir
        self.split_cache_name = split_cache_name or str(target_column)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_target_height = img_target_height
        self.img_target_width = img_target_width
        self.img_divisor = img_divisor
        self.img_min_val = img_min_val
        self.img_max_val = img_max_val
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.multiprocessing_context = multiprocessing_context
        self.train_transform = train_transform
        self.val_test_transform = val_test_transform

        self.save_hyperparameters(ignore=["train_df", "val_df", "test_df"])
        self.prepare_data_per_node = False

    def _has_in_memory_splits(self):
        return self.train_df is not None and self.val_df is not None and self.test_df is not None

    def _split_paths(self):
        split_dir = (
            Path(self.split_cache_dir)
            if self.split_cache_dir
            else (Path(self.data_folder) / "cache" / "flares" / "binary_classification")
        )
        return cache_utils.build_split_cache_paths(cache_root=split_dir, cache_name=str(self.split_cache_name))

    def _split_files_exist(self):
        return cache_utils.split_cache_exists(self._split_paths())

    def prepare_data(self):
        """Build and persist train/val/test splits once on rank zero."""
        if self._has_in_memory_splits():
            return

        if self._split_files_exist():
            return

        if self.df_flares_name is None:
            raise ValueError("df_flares_name must be set when train/val/test dataframes are not provided.")

        from arccnet.models.flares import preprocessing as flare_preprocessing
        from arccnet.models.flares.binary_classification import dataset as binary_dataset

        logger.info("Preparing binary flare data splits (rank-zero only).")
        df = binary_dataset.load_prepared_dataframe(
            data_folder=self.data_folder,
            df_flares_name=self.df_flares_name,
            dataset_folder=self.dataset_folder,
        )
        df_preprocessed = flare_preprocessing.preprocess_flare_data(
            df,
            apply_quality_filter=self.apply_quality_filter,
            apply_path_filter=self.apply_path_filter,
            apply_longitude_filter=self.apply_longitude_filter,
            apply_nan_filter=self.apply_nan_filter,
            max_longitude=self.max_longitude,
            nan_threshold=self.nan_threshold,
            data_folder=self.data_folder,
            dataset_folder=self.dataset_folder,
        )
        train_df, val_df, test_df = binary_dataset.split_preprocessed_data(
            df_preprocessed,
            target_column=self.target_column,
            test_size=self.test_size,
            val_size=self.val_size,
            random_state=self.random_state,
        )

        paths = self._split_paths()
        paths["train"].parent.mkdir(parents=True, exist_ok=True)
        cache_utils.write_split_parquets(train_df, val_df, test_df, paths)
        logger.info("Saved prepared splits to %s", paths["train"].parent)

    def _load_split_dataframes(self):
        if self._has_in_memory_splits():
            return

        paths = self._split_paths()
        missing = [name for name, path in paths.items() if not path.exists()]
        if missing:
            missing_paths = [str(paths[name]) for name in missing]
            raise FileNotFoundError(
                f"Missing prepared split files. Expected prepare_data() to create them first. Missing: {missing_paths}"
            )

        self.train_df, self.val_df, self.test_df = cache_utils.read_split_parquets(paths)
        logger.info(
            "Loaded prepared splits from disk. Shapes: Train=%s, Val=%s, Test=%s",
            self.train_df.shape,
            self.val_df.shape,
            self.test_df.shape,
        )

    def _dataloader_kwargs(self, shuffle):
        kwargs = {
            "batch_size": self.batch_size,
            "shuffle": shuffle,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers if self.num_workers > 0 else False,
        }
        if self.num_workers > 0:
            kwargs["prefetch_factor"] = self.prefetch_factor
            kwargs["multiprocessing_context"] = self.multiprocessing_context
        return kwargs

    def setup(self, stage=None):
        self._load_split_dataframes()

        dataset_args = {
            "data_folder": self.data_folder,
            "dataset_folder": self.dataset_folder,
            "target_column": self.target_column,
            "target_height": self.img_target_height,
            "target_width": self.img_target_width,
            "divisor": self.img_divisor,
            "min_val": self.img_min_val,
            "max_val": self.img_max_val,
        }

        if stage == "fit" or stage is None:
            self.train_dataset = FlareDataset(df=self.train_df, transform=self.train_transform, **dataset_args)
            self.val_dataset = FlareDataset(df=self.val_df, transform=self.val_test_transform, **dataset_args)
        if stage == "validate" or stage == "fit" or stage is None:
            if not hasattr(self, "val_dataset"):
                self.val_dataset = FlareDataset(df=self.val_df, transform=self.val_test_transform, **dataset_args)
        if stage == "test" or stage is None:
            self.test_dataset = FlareDataset(df=self.test_df, transform=self.val_test_transform, **dataset_args)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self._dataloader_kwargs(shuffle=True))

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self._dataloader_kwargs(shuffle=False))

    def test_dataloader(self):
        return DataLoader(self.test_dataset, **self._dataloader_kwargs(shuffle=False))

    def teardown(self, stage=None):
        """Clean up dataset references after fit/test."""
        if stage == "fit" or stage is None:
            if hasattr(self, "train_dataset"):
                del self.train_dataset
            if hasattr(self, "val_dataset"):
                del self.val_dataset
        if stage == "test" or stage is None:
            if hasattr(self, "test_dataset"):
                del self.test_dataset
