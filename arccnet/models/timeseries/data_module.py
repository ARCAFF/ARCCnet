"""PyTorch Lightning data module for timeseries flare forecasting."""

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .config import (
    HFLIP_PROB,
    NUM_WORKERS,
    PERSISTENT_WORKERS,
    PIN_MEMORY,
    RESIZE,
    ROTATION_DEGREES,
    USE_AUGMENTATION,
    VFLIP_PROB,
)
from .dataset import SDOTimeseriesDataset


class FlareDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for flare forecasting."""

    def __init__(
        self,
        manifest_df,
        train_mask,
        val_mask,
        test_mask,
        data_dir,
        norm_stats=None,
        task_type="multiclass",
        batch_size=32,
        num_workers=NUM_WORKERS,
        resize=RESIZE,
        use_augmentation=USE_AUGMENTATION,
        hflip_prob=HFLIP_PROB,
        vflip_prob=VFLIP_PROB,
        rotation_degrees=ROTATION_DEGREES,
    ):
        """
        Initialize the data module.

        Parameters
        ----------
        manifest_df : pd.DataFrame
            Full dataset manifest
        train_mask : pd.Series
            Boolean mask for training samples
        val_mask : pd.Series
            Boolean mask for validation samples
        test_mask : pd.Series
            Boolean mask for test samples
        data_dir : Path
            Directory containing the data
        norm_stats : dict or None
            Normalization statistics (computed from train if None)
        task_type : str
            'multiclass' or 'regression'
        batch_size : int
            Batch size for dataloaders
        num_workers : int
            Number of workers for dataloaders
        """
        super().__init__()
        self.manifest_df = manifest_df
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        self.data_dir = data_dir
        self.norm_stats = norm_stats
        self.task_type = task_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.resize = resize
        self.use_augmentation = use_augmentation
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.rotation_degrees = rotation_degrees

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _make_dataloader(self, dataset, shuffle):
        """Create a DataLoader with the module's common runtime options."""
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=PIN_MEMORY,
            persistent_workers=PERSISTENT_WORKERS if self.num_workers > 0 else False,
        )

    def setup(self, stage=None):
        """Setup datasets for each stage."""
        train_df = self.manifest_df[self.train_mask].reset_index(drop=True)
        val_df = self.manifest_df[self.val_mask].reset_index(drop=True)
        test_df = self.manifest_df[self.test_mask].reset_index(drop=True)

        if self.norm_stats is None:
            temp_train = SDOTimeseriesDataset(
                train_df,
                split="train",
                task_type=self.task_type,
                resize=self.resize,
                augment=False,
                norm_stats=None,
            )
            self.norm_stats = temp_train.get_norm_stats()

        self.train_dataset = SDOTimeseriesDataset(
            train_df,
            split="train",
            task_type=self.task_type,
            resize=self.resize,
            augment=self.use_augmentation,
            norm_stats=self.norm_stats,
            hflip_prob=self.hflip_prob,
            vflip_prob=self.vflip_prob,
            rotation_degrees=self.rotation_degrees,
        )

        self.val_dataset = SDOTimeseriesDataset(
            val_df,
            split="val",
            task_type=self.task_type,
            resize=self.resize,
            augment=False,
            norm_stats=self.norm_stats,
        )

        self.test_dataset = SDOTimeseriesDataset(
            test_df,
            split="test",
            task_type=self.task_type,
            resize=self.resize,
            augment=False,
            norm_stats=self.norm_stats,
        )

    def train_dataloader(self):
        """Create training dataloader."""
        return self._make_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        """Create validation dataloader."""
        return self._make_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        """Create test dataloader."""
        return self._make_dataloader(self.test_dataset, shuffle=False)
