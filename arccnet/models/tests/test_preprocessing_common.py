from pathlib import Path

import numpy as np
import pandas as pd

from astropy.io import fits

from arccnet.models import preprocessing_common as pp_common


def test_normalize_quality_flag():
    assert pp_common.normalize_quality_flag(None) == ""
    assert pp_common.normalize_quality_flag("") == ""
    assert pp_common.normalize_quality_flag("None") == ""
    assert pp_common.normalize_quality_flag("00000200") == "0x00000200"
    assert pp_common.normalize_quality_flag("0X00000400") == "0x00000400"
    assert pp_common.normalize_quality_flag("0x20") == "0x00000020"


def test_apply_quality_filter_is_instrument_aware():
    df = pd.DataFrame(
        {
            "path_image_cutout_hmi": ["hmi_1.fits", "", "", "hmi_2.fits", ""],
            "path_image_cutout_mdi": ["", "mdi_1.fits", "mdi_2.fits", "mdi_3.fits", ""],
            "QUALITY_hmi": ["0x00000000", None, None, "0x00000000", "0x00000000"],
            "QUALITY_mdi": [None, "00000200", "0x00000201", "0x00000201", None],
        }
    )

    filtered = pp_common.apply_quality_filter(df)
    assert filtered.index.tolist() == [0, 1, 4]


def test_apply_path_filter_removes_rows_with_both_paths_missing():
    df = pd.DataFrame(
        {
            "path_image_cutout_hmi": ["hmi.fits", "", "None", np.nan],
            "path_image_cutout_mdi": ["", "mdi.fits", "None", np.nan],
        }
    )

    filtered = pp_common.apply_path_filter(df)
    assert filtered.index.tolist() == [0, 1]


def test_apply_longitude_filter_prefers_hmi_then_mdi():
    df = pd.DataFrame(
        {
            "path_image_cutout_hmi": ["h1.fits", "", "h2.fits", "h3.fits", ""],
            "path_image_cutout_mdi": ["m1.fits", "m2.fits", "m3.fits", "m4.fits", "m5.fits"],
            "longitude_hmi": [30.0, np.nan, 80.0, np.nan, np.nan],
            "longitude_mdi": [80.0, 40.0, 30.0, 50.0, np.nan],
        }
    )

    filtered = pp_common.apply_longitude_filter(df, max_longitude=65.0)
    assert filtered.index.tolist() == [0, 1, 3]


def test_availability_mask_supports_multiple_columns():
    df = pd.DataFrame(
        {
            "path_image_cutout_hmi": ["", "", np.nan, "None"],
            "processed_path_image_hmi": ["a.fits", "", "b.fits", ""],
        }
    )
    mask = pp_common.availability_mask(df, ("path_image_cutout_hmi", "processed_path_image_hmi"))
    assert mask.tolist() == [True, False, True, False]


def test_resolve_cutout_fits_path_supports_legacy_remap(tmp_path: Path):
    data_folder = tmp_path / "data"
    dataset_folder = "dataset_x"
    fits_dir = data_folder / dataset_folder / "data" / "cutout_classification" / "fits"
    fits_dir.mkdir(parents=True)

    fits_file = fits_dir / "sample.fits"
    fits_file.write_bytes(b"test")

    # Basename lookup
    resolved = pp_common.resolve_cutout_fits_path(
        "sample.fits",
        data_folder=data_folder,
        dataset_folder=dataset_folder,
    )
    assert resolved == fits_file

    # Legacy mount-style remap
    legacy_path = "/mnt/ARCAFF/v0.3.0/04_final/data/cutout_classification/fits/sample.fits"
    resolved_legacy = pp_common.resolve_cutout_fits_path(
        legacy_path,
        data_folder=data_folder,
        dataset_folder=dataset_folder,
    )
    assert resolved_legacy == fits_file


def test_resolve_project_path_maps_legacy_mount_prefix(tmp_path: Path):
    local_root = tmp_path / "arccnet-v20251017"
    target = local_root / "04_final" / "data" / "region_detection" / "fits" / "sample.fits"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"test")

    legacy = "/mnt/ARCAFF/v0.3.0/04_final/data/region_detection/fits/sample.fits"
    resolved = pp_common.resolve_project_path(legacy, local_root=local_root)
    assert resolved == target


def test_resolve_project_path_maps_arccnet_data_prefix(tmp_path: Path):
    local_root = tmp_path / "arccnet-v20251017"
    target = local_root / "04_final" / "data" / "region_detection" / "fits" / "sample2.fits"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"test")

    legacy = "arccnet_data/04_final/data/region_detection/fits/sample2.fits"
    resolved = pp_common.resolve_project_path(legacy, local_root=local_root)
    assert resolved == target


def test_compute_cutout_nan_fraction_returns_one_for_missing_path():
    row = pd.Series({"path_image_cutout_hmi": "", "path_image_cutout_mdi": ""})
    nan_fraction = pp_common.compute_cutout_nan_fraction(
        row,
        data_folder="/tmp",
        dataset_folder="missing-dataset",
    )
    assert nan_fraction == 1.0


def test_filter_cutouts_by_nan_threshold_filters_expected_rows(tmp_path: Path):
    data_folder = tmp_path / "data"
    dataset_folder = "dataset_x"
    fits_dir = data_folder / dataset_folder / "data" / "cutout_classification" / "fits"
    fits_dir.mkdir(parents=True)

    low_nan = fits_dir / "low_nan.fits"
    high_nan = fits_dir / "high_nan.fits"

    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)),
        ]
    ).writeto(low_nan)

    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(np.array([[np.nan, np.nan], [np.nan, 1.0]], dtype=np.float32)),
        ]
    ).writeto(high_nan)

    df = pd.DataFrame(
        {
            "path_image_cutout_hmi": ["low_nan.fits", "high_nan.fits", ""],
            "path_image_cutout_mdi": ["", "", ""],
        }
    )

    filtered, nan_stats = pp_common.filter_cutouts_by_nan_threshold(
        df,
        nan_threshold=0.5,
        data_folder=data_folder,
        dataset_folder=dataset_folder,
    )

    assert filtered.index.tolist() == [0]
    assert np.allclose(nan_stats, np.array([0.0, 0.75, 1.0]))
