"""Shared split-cache helpers for flare model training pipelines."""

from __future__ import annotations

import os
import json
from typing import Any
from pathlib import Path
from collections.abc import Mapping

import pandas as pd

SPLIT_NAMES = ("train", "val", "test")


def sanitize_cache_token(value: Any) -> str:
    """Make config values filesystem-safe for deterministic cache keys."""
    return str(value).replace("/", "_").replace(".", "_").replace(" ", "_")


def build_split_cache_paths(cache_root: str | Path, cache_name: str, include_meta: bool = False) -> dict[str, Path]:
    """Build cache artifact paths for train/val/test (and optional metadata)."""
    root = Path(cache_root)
    paths = {name: root / f"{cache_name}_{name}.parquet" for name in SPLIT_NAMES}
    if include_meta:
        paths["meta"] = root / f"{cache_name}_meta.json"
    return paths


def split_cache_exists(paths: Mapping[str, Path]) -> bool:
    """Return True when all expected cache artifacts exist."""
    return all(path.exists() for path in paths.values())


def atomic_write_parquet(df: pd.DataFrame, final_path: str | Path) -> None:
    """Write parquet atomically to avoid partial files on interruption."""
    path = Path(final_path)
    tmp_path = Path(f"{path}.{os.getpid()}.tmp")
    df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, path)


def atomic_write_json(payload: Any, final_path: str | Path, *, indent: int = 2) -> None:
    """Write JSON atomically to avoid partial files on interruption."""
    path = Path(final_path)
    tmp_path = Path(f"{path}.{os.getpid()}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=indent)
    os.replace(tmp_path, path)


def write_split_parquets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    paths: Mapping[str, Path],
) -> None:
    """Persist train/val/test split dataframes atomically."""
    for name, df in zip(SPLIT_NAMES, (train_df, val_df, test_df), strict=True):
        atomic_write_parquet(df, paths[name])


def read_split_parquets(paths: Mapping[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/val/test split dataframes from cache files."""
    train_df = pd.read_parquet(paths["train"])
    val_df = pd.read_parquet(paths["val"])
    test_df = pd.read_parquet(paths["test"])
    return train_df, val_df, test_df


def read_json(path: str | Path) -> Any:
    """Read and deserialize JSON payload."""
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)
