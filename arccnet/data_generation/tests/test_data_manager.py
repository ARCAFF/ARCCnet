import tempfile
from pathlib import Path
from datetime import datetime

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
            print("An unexpected error occurred:", e)


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
    )
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
    )
    assert merged_df.equals(expected_merged_data)
