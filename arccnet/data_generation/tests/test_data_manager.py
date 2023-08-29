import tempfile
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import sunpy.map

from arccnet.data_generation.data_manager import DataManager


# Define a fixture for creating a DataManager instance with default arguments
@pytest.fixture
def data_manager_default():
    return DataManager(
        datetime(2010, 6, 1),
        datetime(2010, 7, 1),
        merge_tolerance=pd.Timedelta("30m"),
        download_fits=False,
        save_to_csv=False,
        save_to_html=False,
    )


def test_fetch_urls(data_manager_default):
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create temporary files inside the temporary directory
        temp_dir_path = Path(temp_dir)
        first_url = data_manager_default.urls_to_download[0:1]

        # Test the fetch_urls function with the temporary file paths
        results = data_manager_default.fetch_urls(first_url, base_directory_path=temp_dir_path)

        # Check if results are not None
        assert results is not None

        # Check if the number of completed results matches the number of temporary files
        assert len(results.errors) == 0

        # Check that the file can be loaded
        try:
            _ = sunpy.map.Map(results[0])
        except Exception as e:
            pytest.fail(f"An unexpected error occurred: {e}")


# Test the merge_activeregionpatchs method
def test_merge_activeregionpatchs_basic(data_manager_default):
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
    merged_df = data_manager_default.merge_activeregionpatchs(full_disk_data, cutout_data)
    expected_merged_data = pd.DataFrame(
        {
            "datetime": [datetime(2023, 1, 1, 0, 0), datetime(2023, 1, 3, 0, 0)],
            "url": ["url1", "url3"],
            "datetime_arc": [datetime(2023, 1, 1, 0, 0), datetime(2023, 1, 3, 0, 0)],
            "url_arc": ["cutout_url1", "cutout_url3"],
        }
    ).dropna()
    assert merged_df.equals(expected_merged_data)


def test_merge_activeregionpatchs_datetime_no_matching(data_manager_default):
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
    merged_df = data_manager_default.merge_activeregionpatchs(full_disk_data, cutout_data)
    expected_merged_data = pd.DataFrame(
        {
            "datetime": [datetime(2023, 1, 1, 0, 0)],
            "url": ["url1"],
            "datetime_arc": [datetime(2023, 1, 1, 0, 0)],
            "url_arc": ["cutout_url1"],
        }
    ).dropna()
    assert merged_df.equals(expected_merged_data)


def test_merge_activeregionpatchs_no_matching(data_manager_default):
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
    merged_df = data_manager_default.merge_activeregionpatchs(full_disk_data, cutout_data)
    expected_merged_data = pd.DataFrame(
        {
            "datetime": [datetime(2023, 1, 1, 0, 0), datetime(2023, 1, 2, 0, 0)],
            "url": ["url1", "url2"],
            "datetime_arc": [None, None],
            "url_arc": [None, None],
        }
    ).dropna()
    # the mergeactiveregion patches drops any NaN as there are no matches to the fulldisk data

    # Check if both data frames are empty
    if merged_df.empty and expected_merged_data.empty:
        assert True  # Empty data frames are considered equivalent
    else:
        assert merged_df.equals(expected_merged_data)


def test_merge_activeregionpatchs_multiple_cutouts(data_manager_default):
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
    merged_df = data_manager_default.merge_activeregionpatchs(full_disk_data, cutout_data)
    expected_merged_data = pd.DataFrame(
        {
            "datetime": [datetime(2023, 1, 1, 0, 0), datetime(2023, 1, 3, 0, 0), datetime(2023, 1, 3, 0, 0)],
            "url": ["url1", "url3", "url3"],
            "datetime_arc": [datetime(2023, 1, 1, 0, 0), datetime(2023, 1, 3, 0, 0), datetime(2023, 1, 3, 0, 0)],
            "url_arc": ["cutout_url1", "cutout_url2", "cutout_url3"],
        }
    ).dropna()
    assert merged_df.equals(expected_merged_data)


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

        # Example assertion related to datetime columns
        assert "datetime_srs" in merged_df.columns
        assert "datetime_hmi" in merged_df.columns
        assert "datetime_mdi" in merged_df.columns

        # Example assertion related to NaN values
        assert not any(merged_df["datetime_srs"].isna())  # No NaN in srs datetime

        assert (
            merged_df[["datetime_hmi", "datetime_mdi"]].notna().any(axis=1).all()
        )  # At least one non-NaN datetime in HMI or MDI

        merged_df_subset = merged_df[["datetime_srs", "datetime_hmi", "datetime_mdi"]]
        # Comparing the merged_df with the expected DataFrame for the current tolerance
        pd.testing.assert_frame_equal(merged_df_subset, expected_df)
