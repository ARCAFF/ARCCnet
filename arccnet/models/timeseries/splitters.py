import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def _build_split_dict(df, train_mask, val_mask, test_mask, strategy, metadata=None):
    """Build a standard split dictionary shared by train/eval."""
    train_mask = pd.Series(train_mask, index=df.index).astype(bool)
    val_mask = pd.Series(val_mask, index=df.index).astype(bool)
    test_mask = pd.Series(test_mask, index=df.index).astype(bool)

    out = {
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
        "strategy": strategy,
        "train_df": df[train_mask].copy(),
        "val_df": df[val_mask].copy(),
        "test_df": df[test_mask].copy(),
    }
    # Backward-compatible aliases.
    out["train"] = out["train_df"]
    out["val"] = out["val_df"]
    out["test"] = out["test_df"]
    if metadata:
        out.update(metadata)
    return out


def _can_stratify(labels):
    """Return True when class counts support sklearn stratified splitting."""
    if labels is None:
        return False
    labels = np.asarray(labels)
    if labels.size == 0:
        return False
    unique, counts = np.unique(labels, return_counts=True)
    return len(unique) > 1 and bool(np.all(counts >= 2))


def _group_stratified_split(groups, strata, test_size, seed):
    """
    Split grouped identifiers with stratification fallback.

    Parameters
    ----------
    groups : np.ndarray
        Group identifiers to split.
    strata : np.ndarray
        Per-group stratification labels.
    test_size : int | float
        train_test_split test_size argument.
    seed : int
        Random seed.
    """
    stratify = strata if _can_stratify(strata) else None
    try:
        train_groups, test_groups, train_strata, test_strata = train_test_split(
            groups,
            strata,
            test_size=test_size,
            random_state=seed,
            stratify=stratify,
        )
    except ValueError:
        train_groups, test_groups, train_strata, test_strata = train_test_split(
            groups,
            strata,
            test_size=test_size,
            random_state=seed,
            stratify=None,
        )
    return np.asarray(train_groups), np.asarray(test_groups), np.asarray(train_strata), np.asarray(test_strata)


def split_by_noaa_group(df, train_frac=0.7, val_frac=0.15, seed=42):
    """
    Split data by NOAA AR with group-level class stratification.

    This preserves active-region separation while approximately matching the
    flare-class severity distribution across train/val/test by stratifying on
    each NOAA AR's maximum `flare_class` value.
    """
    if "noaa_ar" not in df.columns:
        raise ValueError("Expected 'noaa_ar' column in manifest for NOAA split")
    if "flare_class" not in df.columns:
        raise ValueError("Expected 'flare_class' column in manifest for stratified NOAA split")
    if train_frac <= 0 or val_frac <= 0 or train_frac + val_frac >= 1:
        raise ValueError("Expected fractions with train_frac > 0, val_frac > 0, and train_frac + val_frac < 1")

    group_severity = df.groupby("noaa_ar")["flare_class"].max().sort_index()
    unique_noaa = group_severity.index.to_numpy()
    group_strata = group_severity.to_numpy(dtype=np.int64)

    n_noaa = len(unique_noaa)
    if n_noaa == 0:
        raise ValueError("No NOAA AR values found in manifest")
    if n_noaa < 3:
        raise ValueError("Need at least 3 NOAA AR groups to create train/val/test splits")

    holdout_frac = 1.0 - float(train_frac)
    holdout_count = int(round(n_noaa * holdout_frac))
    if n_noaa >= 3:
        holdout_count = min(max(holdout_count, 2), n_noaa - 1)
    else:
        holdout_count = max(1, n_noaa - 1)

    train_noaa, holdout_noaa, _, holdout_strata = _group_stratified_split(
        unique_noaa,
        group_strata,
        test_size=holdout_count,
        seed=seed,
    )

    val_ratio_within_holdout = float(val_frac) / holdout_frac
    val_count = int(round(len(holdout_noaa) * val_ratio_within_holdout))
    if len(holdout_noaa) >= 2:
        val_count = min(max(val_count, 1), len(holdout_noaa) - 1)
    else:
        val_count = len(holdout_noaa)
    test_count = len(holdout_noaa) - val_count

    if test_count > 0:
        val_noaa, test_noaa, _, _ = _group_stratified_split(
            holdout_noaa,
            holdout_strata,
            test_size=test_count,
            seed=seed,
        )
    else:
        val_noaa = holdout_noaa
        test_noaa = np.asarray([], dtype=holdout_noaa.dtype)

    train_noaa = set(np.asarray(train_noaa).tolist())
    val_noaa = set(np.asarray(val_noaa).tolist())
    test_noaa = set(np.asarray(test_noaa).tolist())

    train_mask = df["noaa_ar"].isin(train_noaa)
    val_mask = df["noaa_ar"].isin(val_noaa)
    test_mask = df["noaa_ar"].isin(test_noaa)

    metadata = {
        "train_noaa": sorted(train_noaa),
        "val_noaa": sorted(val_noaa),
        "test_noaa": sorted(test_noaa),
        "stratified_by": "noaa_ar_max_flare_class",
    }
    return _build_split_dict(df, train_mask, val_mask, test_mask, strategy="noaa", metadata=metadata)


def split_by_time(df, train_years=None, val_years=None, test_years=None):
    """
    Split data by year to simulate operational forecasting.
    """
    if "date" not in df.columns:
        raise ValueError("Expected 'date' column in manifest for time split")

    if train_years is None:
        train_years = [2011, 2017, 2018, 2019, 2020]
    if val_years is None:
        val_years = [2021]
    if test_years is None:
        test_years = [2022]

    years = pd.to_datetime(df["date"], errors="coerce").dt.year
    train_mask = years.isin(train_years)
    val_mask = years.isin(val_years)
    test_mask = years.isin(test_years)

    metadata = {
        "train_years": list(train_years),
        "val_years": list(val_years),
        "test_years": list(test_years),
    }
    return _build_split_dict(df, train_mask, val_mask, test_mask, strategy="time", metadata=metadata)


def get_split(df, strategy="noaa", **kwargs):
    """
    Main entry point for split generation.

    Returns
    -------
    dict
        Keys: train_mask, val_mask, test_mask, train_df, val_df, test_df, strategy
    """
    if strategy == "noaa":
        return split_by_noaa_group(df, **kwargs)
    if strategy == "time":
        return split_by_time(df, **kwargs)
    raise ValueError(f"Unknown split strategy: {strategy}")
