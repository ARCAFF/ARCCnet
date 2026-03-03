"""Multiclass flare data module aliases built on shared flare datamodule components."""

from arccnet.models.flares.common_datamodule import FlareDataModule, FlareDataset

__all__ = ["FlareDataset", "FlareDataModule"]
