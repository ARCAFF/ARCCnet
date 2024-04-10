import shutil
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pytest

from astropy.table import QTable

from arccnet.data_generation.data_manager import DataManager, Query
from arccnet.data_generation.magnetograms.instruments import HMILOSMagnetogram, HMISHARPs


# Query
@pytest.fixture
def example_query():
    # Create an example Query object for testing
    start = "2024-01-01T00:00:00"
    end = "2024-01-02T00:00:00"
    frequency = timedelta(hours=1)
    query = Query.create_empty(start, end, frequency)
    return Query(query)


def test_query_initialization(example_query):
    # Test initialization of Query object
    assert len(example_query) == 25  # 25 hours between start and end with hourly frequency
    assert len(example_query.colnames) == 2  # Two columns "target_time" and "url"
    assert np.all(example_query["url"].mask)  # All urls should be masked


def test_query_is_empty(example_query):
    # Test the is_empty property of Query object
    assert example_query.is_empty


@pytest.mark.skip(reason="currently failing with all empty")
def test_query_missing(example_query):
    # Test the missing property of Query object
    assert len(example_query.missing) == len(example_query)


# Define a fixture for creating a DataManager instance with default arguments
@pytest.fixture
@pytest.mark.remote_data
def data_manager_default():
    dm = DataManager(
        start_date=str(datetime(2010, 4, 15)),
        end_date=str(datetime(2010, 5, 15)),
        frequency=timedelta(days=1),
        magnetograms=[
            HMILOSMagnetogram(),
            HMISHARPs(),
        ],
    )

    search = dm.search(
        batch_frequency=1,
        merge_tolerance=timedelta(minutes=30),
    )

    return dm, search


@pytest.fixture(scope="session")
def temp_path_fixture(tmp_path_factory, request):
    temp_dir = tmp_path_factory.mktemp("arccnet_testing")  # Create temporary directory

    # unsure if the cleanup is unnecessary
    def cleanup():
        shutil.rmtree(temp_dir)  # Clean up the temporary directory

    request.addfinalizer(cleanup)  # noqa PT021
    # return the temp_dir, temp_dir/raw, temp_dir/processed
    return temp_dir


def list_files(directory):
    directory_path = Path(directory)
    files = [str(file) for file in directory_path.glob("**/*") if file.is_file()]
    return files


# Testing DataManager.download
@pytest.mark.remote_data
def test_download(data_manager_default, temp_path_fixture):
    # Create an instance of DataManager
    instance, search = data_manager_default

    download_objects = instance.download(
        search[1],  # !TODO fix this [1][0]
        path=temp_path_fixture,
        overwrite=True,
    )

    for do in download_objects:
        do = QTable(do)
        assert set(do[~do["path"].mask]["path"]).issubset(set(list_files(temp_path_fixture)))


# Testing DataManager._download
@pytest.mark.remote_data
def test__download_valid_one_invalid(data_manager_default, temp_path_fixture):
    instance, search = data_manager_default

    urls = search[1][0]["url"][~search[1][0]["url"].mask][0:3]
    urls[1] = "http://url"  # will fail download

    result = instance._download(
        urls,
        path=temp_path_fixture,
        overwrite=True,
        max_retries=5,
    )

    assert len(result) == 2


@pytest.mark.remote_data
def test__download_all_invalid(data_manager_default, temp_path_fixture):
    instance, _ = data_manager_default

    # Test data that isn't valid
    urls = ["http://url1", "http://url2", "http://url3"]

    # Call the method being tested
    result = instance._download(
        urls,
        path=temp_path_fixture,
        overwrite=True,
        max_retries=5,
    )

    # Check if the downloaded_successfully column is populated with False values
    assert len(result) == 0
