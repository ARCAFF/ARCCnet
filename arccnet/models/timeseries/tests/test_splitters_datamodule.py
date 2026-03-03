"""Integration tests for split contract and Lightning DataModule wiring."""

import sys
from pathlib import Path

import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from arccnet.models.timeseries.data_module import FlareDataModule
from arccnet.models.timeseries.splitters import get_split


def _mock_manifest(n_samples=6):
    rows = []
    for i in range(n_samples):
        rows.append(
            {
                "sample_id": f"sample_{i}",
                "sample_path": f"/fake/path/{i}",
                "noaa_ar": 1000 + i,
                "date": f"202{i % 3 + 1}-01-01",
                "hale_class": "Beta",
                "mcintosh": "Dso",
                "num_timesteps": 6,
                "timestamps": ["t0", "t1", "t2", "t3", "t4", "t5"],
                "paths": [[None] * 10] * 6,
                "xb": 0,
                "mb": 0,
                "cb": 0,
                "xa": 1 if i % 5 == 0 else 0,
                "ma": 1 if i % 3 == 0 else 0,
                "ca": 1,
                "c_plus": 1,
                "m_plus": 1 if i % 3 == 0 else 0,
                "x_plus": 1 if i % 5 == 0 else 0,
                "flare_class": 2 if (i % 5 == 0 or i % 3 == 0) else 1,
                "log_ca": 0.301,
                "log_ma": 0.301 if i % 3 == 0 else 0.0,
                "log_xa": 0.301 if i % 5 == 0 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def test_splitter_contract():
    """Split helpers should return explicit masks/dataframes without mutating source df."""
    df = _mock_manifest()
    original_cols = list(df.columns)

    split_data = get_split(df, strategy="noaa", train_frac=0.5, val_frac=0.25, seed=42)
    required = {"train_mask", "val_mask", "test_mask", "train_df", "val_df", "test_df", "strategy"}
    assert required.issubset(split_data.keys()), f"Missing keys in split data: {required - set(split_data.keys())}"

    train_noaa = set(split_data["train_df"]["noaa_ar"].unique().tolist())
    val_noaa = set(split_data["val_df"]["noaa_ar"].unique().tolist())
    test_noaa = set(split_data["test_df"]["noaa_ar"].unique().tolist())

    assert train_noaa.isdisjoint(val_noaa)
    assert train_noaa.isdisjoint(test_noaa)
    assert val_noaa.isdisjoint(test_noaa)

    assert len(split_data["train_df"]) + len(split_data["val_df"]) + len(split_data["test_df"]) == len(df)

    _ = get_split(df, strategy="time", train_years=[2021], val_years=[2022], test_years=[2023])
    assert list(df.columns) == original_cols, "split_by_time should not mutate source dataframe columns"


def test_noaa_split_stratifies_group_severity():
    """NOAA split should preserve per-group flare-class strata across splits when feasible."""
    rows = []
    sample_idx = 0
    for noaa_group in range(30):
        group_class = noaa_group % 3
        for replicate in range(2):
            rows.append(
                {
                    "sample_id": f"sample_{sample_idx}",
                    "sample_path": f"/fake/path/{sample_idx}",
                    "noaa_ar": 2000 + noaa_group,
                    "date": f"202{replicate + 1}-01-01",
                    "hale_class": "Beta",
                    "mcintosh": "Dso",
                    "num_timesteps": 6,
                    "timestamps": ["t0", "t1", "t2", "t3", "t4", "t5"],
                    "paths": [[None] * 10] * 6,
                    "xb": 0,
                    "mb": 0,
                    "cb": 0,
                    "xa": int(group_class == 2),
                    "ma": int(group_class == 2),
                    "ca": int(group_class >= 1),
                    "c_plus": int(group_class >= 1),
                    "m_plus": int(group_class == 2),
                    "x_plus": int(group_class == 2),
                    "flare_class": group_class,
                    "log_ca": 0.301 if group_class >= 1 else 0.0,
                    "log_ma": 0.301 if group_class == 2 else 0.0,
                    "log_xa": 0.301 if group_class == 2 else 0.0,
                }
            )
            sample_idx += 1

    df = pd.DataFrame(rows)
    split_data = get_split(df, strategy="noaa", train_frac=0.6, val_frac=0.2, seed=42)

    train_classes = set(split_data["train_df"].groupby("noaa_ar")["flare_class"].max().unique().tolist())
    val_classes = set(split_data["val_df"].groupby("noaa_ar")["flare_class"].max().unique().tolist())
    test_classes = set(split_data["test_df"].groupby("noaa_ar")["flare_class"].max().unique().tolist())

    assert train_classes == {0, 1, 2}
    assert val_classes == {0, 1, 2}
    assert test_classes == {0, 1, 2}


def test_datamodule_dataset_wiring():
    """DataModule should pass explicit split names and return mask tensors in batches."""
    df = _mock_manifest(n_samples=9)
    train_mask = pd.Series([True, True, True, False, False, False, False, False, False])
    val_mask = pd.Series([False, False, False, True, True, False, False, False, False])
    test_mask = pd.Series([False, False, False, False, False, True, True, True, True])

    dm = FlareDataModule(
        manifest_df=df,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        data_dir=Path("/tmp"),
        task_type="multiclass",
        batch_size=2,
        num_workers=0,
    )
    dm.setup("fit")

    assert dm.train_dataset.split == "train"
    assert dm.val_dataset.split == "val"
    assert dm.test_dataset.split == "test"

    batch = next(iter(dm.train_dataloader()))
    assert "mask" in batch, "Expected timestep mask in dataloader batch"
    assert batch["mask"].shape[1] == 6, f"Expected mask with 6 timesteps, got {batch['mask'].shape}"


if __name__ == "__main__":
    test_splitter_contract()
    test_noaa_split_stratifies_group_severity()
    test_datamodule_dataset_wiring()
    print("\n✅ Splitter/DataModule tests passed!")
