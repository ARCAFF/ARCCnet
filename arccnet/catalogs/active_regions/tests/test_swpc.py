from pathlib import Path
from datetime import datetime

import pandas as pd
import pytest
from parfive import Results

import astropy.units as u
from astropy.table import MaskedColumn, QTable
from astropy.time import Time

from arccnet.catalogs.active_regions.swpc import Query, SWPCCatalog, filter_srs


@pytest.fixture
def swpc_catalog():
    return SWPCCatalog()


@pytest.fixture()
def fake_fido_search():
    fake_fido_search = QTable()
    fake_fido_search["start_time"] = Time("2022-01-01") + [0, 1, 2] * u.day
    fake_fido_search["end_time"] = fake_fido_search["start_time"] + (1 * u.day - 1 * u.ms)
    fake_fido_search.add_column(MaskedColumn(["1", "2", "3"], name="url"))
    for n, v in [("Instrument", "SOON"), ("Physobs", "SRS"), ("Source", "SWPC"), ("Provider", "NOAA")]:
        fake_fido_search[n] = v
    return fake_fido_search


@pytest.fixture()
def fake_fido_fetch(fake_fido_search):
    fake_fido_fetch = fake_fido_search.copy()
    fake_fido_fetch["start_time"] = Time("2022-01-01") + [0, 1, 2] * u.day
    fake_fido_fetch["end_time"] = fake_fido_fetch["start_time"] + (1 * u.day - 1 * u.ms)
    fake_fido_fetch.replace_column("url", MaskedColumn(["1", "2", "3"], name="url"))
    fake_fido_fetch.add_column(MaskedColumn(["a.txt", "b.txt", "c.txt"], name="path"))
    for n, v in [("Instrument", "SOON"), ("Physobs", "SRS"), ("Source", "SWPC"), ("Provider", "NOAA")]:
        fake_fido_fetch[n] = v
    return fake_fido_fetch


def test_create_empty_results():
    start, end = "2022-01-01", "2022-01-03"
    empty = Query.create_empty(start, end)
    assert len(empty) == 3
    assert empty[0]["start_time"] == Time(start)
    assert empty[-1]["start_time"] == Time(end)

    with pytest.raises(ValueError):
        Query.create_empty(start, "2022-01-03 10:00")


def test_search(swpc_catalog, fake_fido_search, mocker):
    fido_mock = mocker.patch("arccnet.catalogs.active_regions.swpc.Fido.search")
    start, end = "2022-01-01", "2022-01-03"
    empty = Query.create_empty(start, end)

    fido_mock.return_value = {"srs": fake_fido_search}

    # Normal path first time
    res = swpc_catalog.search(empty)
    assert len(res) == 3
    assert all(res["url"] == ["1", "2", "3"])
    fido_mock.assert_called_once()

    # Normal path less responses than expected
    fake_fido_search.remove_row(0)
    fido_mock.return_value = {"srs": fake_fido_search}

    res = swpc_catalog.search(empty)
    assert len(res) == 3
    assert all(res["url"].filled("") == ["", "2", "3"])
    assert fido_mock.call_count == 2

    # Existing data should just pass back and not call fido
    empty["url"] = MaskedColumn(["a", "b", "c"])
    res = swpc_catalog.search(empty)
    assert len(res) == 3
    assert all(res["url"] == ["a", "b", "c"])
    assert fido_mock.call_count == 2


def test_search_retry(swpc_catalog, fake_fido_search, mocker):
    fido_mock = mocker.patch("arccnet.catalogs.active_regions.swpc.Fido.search")
    start, end = "2022-01-01", "2022-01-03"
    empty = Query.create_empty(start, end)

    fido_mock.return_value = {"srs": fake_fido_search[1:]}

    empty["url"] = MaskedColumn(["1", "", ""], mask=[False, True, True])

    res = swpc_catalog.search(empty, retry_missing=True)
    assert len(res) == 3
    assert all(res["url"] == ["1", "2", "3"])
    fido_mock.assert_called_once()
    assert len(fido_mock.call_args_list[0].args[0].attrs) == 2  # two time ranges


def test_search_overwrite(swpc_catalog, fake_fido_search, mocker):
    fido_mock = mocker.patch("arccnet.catalogs.active_regions.swpc.Fido.search")
    start, end = "2022-01-01", "2022-01-03"
    empty = Query.create_empty(start, end)

    fido_mock.return_value = {"srs": fake_fido_search}

    res = swpc_catalog.search(empty, overwrite=True)
    assert len(res) == 3
    assert all(res["url"] == ["1", "2", "3"])
    fido_mock.assert_called_once()

    empty["url"] = MaskedColumn(["x", "y", "z"])
    res = swpc_catalog.search(empty, overwrite=True)
    assert len(res) == 3
    assert all(res["url"] == ["1", "2", "3"])
    assert fido_mock.call_count == 2


def test_download(swpc_catalog, fake_fido_fetch, mocker):
    fido_mock = mocker.patch("arccnet.catalogs.active_regions.swpc.Fido.fetch")
    results = swpc_catalog.download(fake_fido_fetch)
    fido_mock.assert_not_called()
    assert len(results) == 3
    assert all(results["path"] == ["a.txt", "b.txt", "c.txt"])

    fake_fido_fetch.remove_column("path")
    fido_mock.return_value = Results(["20220103SRS.txt", "20220101SRS.txt", "20220102SRS.txt"])
    results = swpc_catalog.download(fake_fido_fetch)
    fido_mock.assert_called_once()
    assert len(results) == 3
    assert all(results["path"] == ["20220101SRS.txt", "20220102SRS.txt", "20220103SRS.txt"])

    fido_mock.return_value = Results(["20220103SRS.txt", "20220101SRS.txt"])
    results = swpc_catalog.download(fake_fido_fetch)
    assert fido_mock.call_count == 2
    assert len(results) == 3
    assert all(results["path"] == ["20220101SRS.txt", "", "20220103SRS.txt"])


def test_download_overwrite(swpc_catalog, fake_fido_fetch, mocker):
    fido_mock = mocker.patch("arccnet.catalogs.active_regions.swpc.Fido.fetch")
    fido_mock.return_value = Results(["20220103SRS.txt", "20220101SRS.txt", "20220102SRS.txt"])
    results = swpc_catalog.download(fake_fido_fetch, overwrite=True)
    fido_mock.assert_called_once()
    assert len(results) == 3
    assert all(results["path"] == ["20220101SRS.txt", "20220102SRS.txt", "20220103SRS.txt"])


def test_download_retry_missing(swpc_catalog, fake_fido_fetch, mocker):
    fido_mock = mocker.patch("arccnet.catalogs.active_regions.swpc.Fido.fetch")
    fido_mock.return_value = Results(["20220103SRS.txt", "20220101SRS.txt", "20220102SRS.txt"])
    fake_fido_fetch["path"] = ["20220103SRS.txt", "", "20220102SRS.txt"]
    results = swpc_catalog.download(fake_fido_fetch, retry_missing=True)
    fido_mock.assert_called_once()
    assert len(results) == 3
    assert all(results["path"] == ["20220101SRS.txt", "20220102SRS.txt", "20220103SRS.txt"])


def test_create_catalog(swpc_catalog, fake_fido_fetch):
    data_dir = Path(__file__).parent / "data"
    fake_fido_fetch["path"] = MaskedColumn(
        [data_dir / file for file in ["20220104SRS.txt", "20220105SRS.txt", "20220106SRS.txt"]]
    )
    catalog = swpc_catalog.create_catalog(fake_fido_fetch)
    assert len(catalog) == 17


def test_filter_srs():
    # Synthetic the raw_catalog
    catalog = pd.DataFrame(
        {
            "id": ["I"] * 10,
            "magnetic_class": ["Alpha"] * 8 + ["keta", "beta"],
            "mcintosh_class": ["Fkc"] * 8 + ["Fkc", "Fck"],
            "latitude": [70, 10, 10, 70, 40, -20, -20, -20, 10, 10],
            "longitude": [40, 10, 40, 20, 34, -75, -61, -47, 10, 10],
            "number": [0, 1, 1, 2, 2, 3, 3, 3, 4, 5],
            "number_of_sunspots": [2] * 10,
            "time": [
                datetime(2023, 1, 1),
                datetime(2023, 1, 1),
                datetime(2023, 1, 2),
                datetime(2023, 1, 1),
                datetime(2023, 1, 2),
                datetime(2023, 1, 1),
                datetime(2023, 1, 2),
                datetime(2023, 1, 3),
                datetime(2023, 1, 3),
                datetime(2023, 1, 3),
            ],
            "path": ["a"] * 10,
            "url": ["b"] * 10,
        }
    )
    processed = filter_srs(QTable.from_pandas(catalog))
    assert processed["filtered"].sum() == 7

    assert processed["filter_reason"][0] == "bad_lat,"
    assert processed["filter_reason"][1] == "bad_lon_rate,"
    assert processed["filter_reason"][2] == "bad_lon_rate,"
    assert processed["filter_reason"][3] == "bad_lat,bad_lat_rate,"
    assert processed["filter_reason"][4] == "bad_lat_rate,"
    assert processed["filter_reason"][-2] == "invalid_magnetic_class,"
    assert processed["filter_reason"][-1] == "invalid_mcintosh_class,"
