import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import patch

import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from arccnet.data_generation.catalogs.active_region_catalogs.swpc import NoDataError, SWPCCatalog, filter_srs


@pytest.fixture
def swpc_catalog():
    return SWPCCatalog()


@pytest.fixture
def valid_start_date():
    return datetime(2012, 1, 1)


@pytest.fixture
def valid_end_date():
    return datetime(2012, 1, 3)


def test_fetch_data_with_valid_dates(swpc_catalog, valid_start_date, valid_end_date):
    fetched_data = swpc_catalog.fetch_data(valid_start_date, valid_end_date)
    assert fetched_data is not None


def test_fetch_data_with_no_data(swpc_catalog):
    """
    test with 1st Jan 1980.
    """
    start_date = datetime(1980, 1, 1)
    end_date = datetime(1980, 1, 2)
    with pytest.raises(NoDataError):
        swpc_catalog.fetch_data(start_date, end_date)


@patch("arccnet.data_generation.catalogs.active_region_catalogs.swpc.dv")
def test_creat_catalog(mockdv, swpc_catalog, tmp_path):
    test_data_dir = Path(__file__).parent / "data"
    test_files = list(test_data_dir.glob("*.txt"))

    mockdv.NOAA_SRS_TEXT_EXCEPT_DIR = str(tmp_path / "bad")
    mockdv.SRS_FILEPATHS_IGNORED = ["19961209SRS.txt"]
    input = tmp_path / "input"
    input.mkdir()
    [shutil.copy(test_file, input) for test_file in test_files]
    files = input.glob("*.txt")

    swpc_catalog._fetched_data = list(files)
    raw, missing = swpc_catalog.create_catalog(save_csv=False, save_html=False)
    assert len(missing) == 1
    assert missing["filename"].values[0] == "20100228SRS.txt"
    assert len(raw) == 14
    assert set(raw.loc[raw.loaded_successfully == 1, "filename"]) == {"19990101SRS.txt"}


def test_create_catalog_with_valid_data(swpc_catalog, valid_start_date, valid_end_date):
    # Call fetch_data before create_catalog
    swpc_catalog.fetch_data(valid_start_date, valid_end_date)
    raw_catalog, raw_catalog_missing = swpc_catalog.create_catalog()
    assert raw_catalog is not None
    assert raw_catalog_missing is not None


def test_create_catalog_with_valid_data_srs_loading_issues(swpc_catalog):
    """
    Test times when SRS doesn't load
    """
    # Call fetch_data before create_catalog
    swpc_catalog.fetch_data(datetime(1999, 1, 1), datetime(1999, 1, 31))
    raw_catalog, raw_catalog_missing = swpc_catalog.create_catalog()
    assert raw_catalog is not None
    assert raw_catalog_missing is not None


def test_create_catalog_with_no_data(swpc_catalog):
    # Call fetch_data before create_catalog
    with pytest.raises(NoDataError):
        swpc_catalog.create_catalog()


def test_clean_catalog(swpc_catalog):
    with pytest.raises(NoDataError):
        swpc_catalog.clean_catalog()

    # Synthetic the raw_catalog
    swpc_catalog.raw_catalog = pd.DataFrame(
        {
            "Latitude": [15, 15, 15],
            "Longitude": [-74, -60, -46],
            "Number": [1234, 1234, 1234],
            "datetime": [datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2023, 1, 3)],
            "loaded_successfully": [True, True, False],
            "column1": [1, 2, None],
            "column2": [3, 4, 5],
            "Mag Type": ["Alpha", "Beta", "Gamma"],
            "ID": ["I", "IA", "II"],
            "Z": ["Axx", "Eac", "Eac"],
        }
    )
    catalog = swpc_catalog.clean_catalog()
    assert catalog is not None
    assert len(catalog) == 2
    assert catalog["column1"].isnull().sum() == 0


def test_filter_srs():
    # Synthetic the raw_catalog
    catalog = pd.DataFrame(
        {
            "Latitude": [70, 10, 10, 50, 20, -20, -20, -20],
            "Longitude": [40, 10, 40, 20, 34, -75, -61, -47],
            "Number": [0, 1, 1, 2, 2, 3, 3, 3],
            "datetime": [
                datetime(2023, 1, 1),
                datetime(2023, 1, 1),
                datetime(2023, 1, 2),
                datetime(2023, 1, 1),
                datetime(2023, 1, 2),
                datetime(2023, 1, 1),
                datetime(2023, 1, 2),
                datetime(2023, 1, 3),
            ],
        }
    )
    filtered, bad = filter_srs(catalog)
    assert len(filtered) == 3
    bad["bad_lat"][0] == 0
    assert len(bad["bad_lat"]) == 1
    assert_array_equal([3, 4], bad["bad_lat_rate"][0])
    assert_array_equal([1, 2], bad["bad_lon_rate"][0])
