import shutil
import tempfile
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from arccnet.data_generation.data_manager import DataManager


# Define a fixture for creating a DataManager instance with default arguments
@pytest.fixture
def data_manager_default():
    return DataManager(
        datetime(2010, 6, 1),
        datetime(2010, 6, 7),
        merge_tolerance=pd.Timedelta("30m"),
        download_fits=False,
        overwrite_fits=False,
        save_to_csv=False,
    )


@pytest.fixture(scope="session")
def temp_path_fixture(request):
    temp_dir = tempfile.mkdtemp()  # Create temporary directory

    def cleanup():
        shutil.rmtree(temp_dir)  # Clean up the temporary directory

    request.addfinalizer(cleanup)  # noqa PT021
    # return the temp_dir, temp_dir/raw, temp_dir/processed
    return Path(temp_dir)


@pytest.fixture(params=[False, True, False])
def overwrite_fixture(request):
    # testing:
    # 1. overwrite = False # data doesn't exist
    # 1. overwrite = True # overwrite existing data
    # 1. overwrite = False # don't overwrite existing data
    return request.param


# test_fetch_fits_valid with different overwrite settings
def test_fetch_fits_overwrite(overwrite_fixture, data_manager_default, temp_path_fixture):
    # Create an instance of DataManager
    instance = data_manager_default

    test_urls = instance.merged_df[["url_hmi"]][0:3]

    # fetch_fits with the overwrite parameter from the fixture
    result = instance.fetch_fits(
        urls_df=test_urls,
        column_name="url_hmi",
        base_directory_path=temp_path_fixture,
        suffix="_hmi",
        overwrite=overwrite_fixture,
    )

    assert all(result["downloaded_successfully_hmi"])


# test_fetch_fits_valid
def test_fetch_fits_valid_one_invalid(overwrite_fixture, data_manager_default, temp_path_fixture):
    instance = data_manager_default

    test_urls = instance.merged_df[["url_hmi"]][0:3]
    test_urls["url_hmi"][1] = "http://url"  # will fail download

    result = instance.fetch_fits(
        urls_df=test_urls,
        column_name="url_hmi",
        base_directory_path=temp_path_fixture,
        suffix="_hmi",
        overwrite=overwrite_fixture,
    )

    expected_values = [True, False, True]

    # Ensure the downloaded_successfully column is True False True
    assert all(result["downloaded_successfully_hmi"] == expected_values)


def test_fetch_fits_valid_no_data(data_manager_default, temp_path_fixture):
    instance = data_manager_default

    # Test data that isn't valid
    urls = ["http://url1", "http://url2", "http://url3"]
    test_urls = pd.DataFrame({"url": urls})

    # Call the method being tested
    result = instance.fetch_fits(
        urls_df=test_urls,
        column_name="url",
        base_directory_path=temp_path_fixture,
        suffix="",
        overwrite=False,
    )

    # Check if the downloaded_successfully column is populated with False values
    assert not all(result["downloaded_successfully"])


def test_fetch_fits_invalid_df(data_manager_default, temp_path_fixture):
    instance = data_manager_default

    # provide None as the dataframe
    result = instance.fetch_fits(urls_df=None, column_name="url", base_directory_path=temp_path_fixture)

    assert result is None


# Test the merge_activeregionpatches method
def test_merge_activeregionpatches_basic(data_manager_default):
    # Test Case: Basic merge
    full_disk_data = pd.DataFrame(
        {
            "datetime": [datetime(2023, 1, 1, 0, 0), datetime(2023, 1, 2, 0, 0), datetime(2023, 1, 3, 0, 0)],
            "url": ["url1", "url2", "url3"],
        }
    )
    cutout_data = pd.DataFrame(
        {
            "datetime": [datetime(2023, 1, 1, 0, 0), datetime(2023, 1, 3, 0, 0)],
            "url": ["cutout_url1", "cutout_url3"],
        }
    )
    merged_df = data_manager_default.merge_activeregionpatches(full_disk_data, cutout_data)
    expected_merged_data = pd.DataFrame(
        {
            "datetime": [datetime(2023, 1, 1, 0, 0), datetime(2023, 1, 3, 0, 0)],
            "url": ["url1", "url3"],
            "datetime_arc": [datetime(2023, 1, 1, 0, 0), datetime(2023, 1, 3, 0, 0)],
            "url_arc": ["cutout_url1", "cutout_url3"],
        }
    ).dropna()
    pd.testing.assert_frame_equal(merged_df, expected_merged_data)


def test_merge_activeregionpatches_datetime_no_matching(data_manager_default):
    # Test Case: One cutout datetime doesn't match exactly to cutout data
    full_disk_data = pd.DataFrame(
        {
            "datetime": [datetime(2023, 1, 1, 0, 0), datetime(2023, 1, 2, 0, 0), datetime(2023, 1, 3, 0, 1)],
            "url": ["url1", "url2", "url3"],
        }
    )
    cutout_data = pd.DataFrame(
        {
            "datetime": [datetime(2023, 1, 1, 0, 0), datetime(2023, 1, 3, 0, 0)],
            "url": ["cutout_url1", "cutout_url3"],
        }
    )
    merged_df = data_manager_default.merge_activeregionpatches(full_disk_data, cutout_data)
    expected_merged_data = pd.DataFrame(
        {
            "datetime": [datetime(2023, 1, 1, 0, 0)],
            "url": ["url1"],
            "datetime_arc": [datetime(2023, 1, 1, 0, 0)],
            "url_arc": ["cutout_url1"],
        }
    ).dropna()
    pd.testing.assert_frame_equal(merged_df, expected_merged_data)


def test_merge_activeregionpatches_no_matching(data_manager_default):
    # Test Case: No matching cutout data
    full_disk_data = pd.DataFrame(
        {
            "datetime": [datetime(2023, 1, 1, 0, 0), datetime(2023, 1, 2, 0, 0)],
            "url": ["url1", "url2"],
        }
    )
    cutout_data = pd.DataFrame(
        {
            "datetime": [datetime(2023, 1, 3, 0, 0)],
            "url": ["cutout_url3"],
        }
    )
    merged_df = data_manager_default.merge_activeregionpatches(full_disk_data, cutout_data)
    expected_merged_data = pd.DataFrame(
        {
            "datetime": [datetime(2023, 1, 1, 0, 0), datetime(2023, 1, 2, 0, 0)],
            "url": ["url1", "url2"],
            "datetime_arc": [None, None],
            "url_arc": [None, None],
        }
    ).dropna()
    # the mergeactiveregion_patches drops any NaN as there are no matches to the fulldisk data

    # Check if both data frames are empty
    if merged_df.empty and expected_merged_data.empty:
        assert True  # Empty data frames are considered equivalent
    else:
        pd.testing.assert_frame_equal(merged_df, expected_merged_data)


def test_merge_activeregionpatches_multiple_cutouts(data_manager_default):
    # Test Case: Multiple cutout data for the same full_disk_data
    full_disk_data = pd.DataFrame(
        {
            "datetime": [datetime(2023, 1, 1, 0, 0), datetime(2023, 1, 2, 0, 0), datetime(2023, 1, 3, 0, 0)],
            "url": ["url1", "url2", "url3"],
        }
    )
    cutout_data = pd.DataFrame(
        {
            "datetime": [datetime(2023, 1, 1, 0, 0), datetime(2023, 1, 3, 0, 0), datetime(2023, 1, 3, 0, 0)],
            "url": ["cutout_url1", "cutout_url2", "cutout_url3"],
        }
    )
    merged_df = data_manager_default.merge_activeregionpatches(full_disk_data, cutout_data)
    expected_merged_data = pd.DataFrame(
        {
            "datetime": [datetime(2023, 1, 1, 0, 0), datetime(2023, 1, 3, 0, 0), datetime(2023, 1, 3, 0, 0)],
            "url": ["url1", "url3", "url3"],
            "datetime_arc": [datetime(2023, 1, 1, 0, 0), datetime(2023, 1, 3, 0, 0), datetime(2023, 1, 3, 0, 0)],
            "url_arc": ["cutout_url1", "cutout_url2", "cutout_url3"],
        }
    ).dropna()
    pd.testing.assert_frame_equal(merged_df, expected_merged_data)


@pytest.fixture
def sample_merged_data():
    # Create sample dataframes for testing
    # srs_keys, hmi_keys, mdi_keys
    srs_keys = pd.DataFrame(
        {
            "datetime": [
                datetime(2023, 1, 1, 0, 0),
                datetime(2023, 1, 2, 0, 0),
                datetime(2023, 1, 3, 0, 0),
            ],
        }
    )

    # HMI data within tolerance of SRS data
    hmi_keys = pd.DataFrame(
        {
            "magnetogram_fits": [
                "hmi_1.fits",
                "hmi_2.fits",
                "hmi_3.fits",
            ],
            "datetime": [
                datetime(2023, 1, 1, 0, 20),
                datetime(2023, 1, 1, 23, 30),
                datetime(2023, 1, 2, 23, 45),
            ],
            "url": [
                "https://example.com/hmi_1.fits",
                "https://example.com/hmi_2.fits",
                "https://example.com/hmi_3.fits",
            ],
        }
    )

    # MDI data outside tolerance of SRS data
    mdi_keys = pd.DataFrame(
        {
            "magnetogram_fits": [
                "mdi_1.fits",
                "mdi_2.fits",
                "mdi_3.fits",
            ],
            "datetime": [
                datetime(2023, 1, 1, 0, 29),
                datetime(2023, 1, 2, 0, 31),
                datetime(2023, 1, 3, 0, 30),
            ],
            "url": [
                "https://example.com/mdi_1.fits",
                "https://example.com/mdi_2.fits",
                "https://example.com/mdi_3.fits",
            ],
        }
    )

    # expected_output as a function of Timedelta
    # (just datetime keys)
    expected_output = {
        pd.Timedelta("20m"): pd.DataFrame(
            {
                "datetime_srs": [
                    datetime(2023, 1, 1, 0, 0),
                    # datetime(2023, 1, 2, 0, 0), # dropped as hmi/mdi are nan
                    datetime(2023, 1, 3, 0, 0),
                ],
                "datetime_hmi": [  #
                    datetime(2023, 1, 1, 0, 20),  # 20m <= 20m
                    # np.nan,  # datetime(2023, 1, 1, 23, 30),  # 30m > 20m
                    datetime(2023, 1, 2, 23, 45),  # 15m <= 20m
                ],
                "datetime_mdi": [
                    np.nan,  # datetime(2023, 1, 1, 0, 29),  # 29m > 20m
                    # np.nan,  # datetime(2023, 1, 2, 0, 31),  # 31m > 20m
                    np.nan,  # datetime(2023, 1, 3, 0, 30),  # 30m > 20m
                ],
            }
        ),
        pd.Timedelta("30m"): pd.DataFrame(
            {
                "datetime_srs": [
                    datetime(2023, 1, 1, 0, 0),
                    datetime(2023, 1, 2, 0, 0),
                    datetime(2023, 1, 3, 0, 0),
                ],
                "datetime_hmi": [  #
                    datetime(2023, 1, 1, 0, 20),  # 30m <= 30m
                    datetime(2023, 1, 1, 23, 30),  # 30m <= 30m
                    datetime(2023, 1, 2, 23, 45),  # 15m <= 30m
                ],
                "datetime_mdi": [
                    datetime(2023, 1, 1, 0, 29),  # 29m <= 30m
                    np.nan,  # datetime(2023, 1, 2, 0, 31), # 31m > 30m
                    datetime(2023, 1, 3, 0, 30),  # 30m <= 30m
                ],
            }
        ),
        pd.Timedelta("31m"): pd.DataFrame(
            {
                "datetime_srs": [
                    datetime(2023, 1, 1, 0, 0),
                    datetime(2023, 1, 2, 0, 0),
                    datetime(2023, 1, 3, 0, 0),
                ],
                "datetime_hmi": [  #
                    datetime(2023, 1, 1, 0, 20),  # 31m <= 31m
                    datetime(2023, 1, 1, 23, 30),  # 31m <= 31m
                    datetime(2023, 1, 2, 23, 45),  # 15m <= 31m
                ],
                "datetime_mdi": [
                    datetime(2023, 1, 1, 0, 29),  # 29m <= 31m
                    datetime(2023, 1, 2, 0, 31),  # 31m <= 31m
                    datetime(2023, 1, 3, 0, 30),  # 30m <= 31m
                ],
            }
        ),
    }

    return srs_keys, hmi_keys, mdi_keys, expected_output


def test_merge_hmimdi_metadata(sample_merged_data, data_manager_default):
    srs_keys, hmi_keys, mdi_keys, expected_output = sample_merged_data

    data_manager = data_manager_default

    # Iterate through the expected_output dictionary to get tolerance and expected DataFrame
    for tolerance, expected_df in expected_output.items():
        merged_df, _ = data_manager.merge_hmimdi_metadata(
            srs_keys=srs_keys,
            hmi_keys=hmi_keys,
            mdi_keys=mdi_keys,
            tolerance=tolerance,
        )

        # Iterate through all columns and convert to datetime objects
        # This is necessary as a column with just np.nan will be NaN,
        # but need NaT to match.
        # For columns with datetime objects, this is automatic
        for column in ["datetime_srs", "datetime_hmi", "datetime_mdi"]:
            expected_df[column] = pd.to_datetime(expected_df[column], errors="coerce")

        assert "datetime_srs" in merged_df.columns
        assert "datetime_hmi" in merged_df.columns
        assert "datetime_mdi" in merged_df.columns

        assert not any(merged_df["datetime_srs"].isna())  # No NaN in srs datetime

        assert (
            merged_df[["datetime_hmi", "datetime_mdi"]].notna().any(axis=1).all()
        )  # At least one non-NaN datetime in HMI or MDI

        merged_df_subset = merged_df[["datetime_srs", "datetime_hmi", "datetime_mdi"]]
        # Comparing the merged_df with the expected DataFrame for the current tolerance
        pd.testing.assert_frame_equal(merged_df_subset, expected_df)
