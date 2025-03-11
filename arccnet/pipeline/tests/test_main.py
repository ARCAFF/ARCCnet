import pickle
from datetime import date

import pytest

from astropy.table import Table
from astropy.time import Time

import arccnet
from arccnet.pipeline.main import extract_flare_counts, process_hmi, process_mdi, process_srs


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):  # , request):
    # unsure if the cleanup is unnecessary
    # def cleanup():
    #     shutil.rmtree(temp_dir)  # Clean up the temporary directory

    # request.addfinalizer(cleanup)  # noqa PT021

    return tmp_path_factory.mktemp("arccnet_testing")


@pytest.fixture(scope="session")
def test_config(data_dir):
    # So don't change global config work on copy (copy(arccnet.config) didn't work
    rep = pickle.dumps(arccnet.config)
    config = pickle.loads(rep)
    config.set("paths", "data_root", str(data_dir))
    config.set("general", "start_date", "2010-06-01T00:00:00")
    config.set("general", "end_date", "2010-06-02T00:00:00")
    return config


@pytest.mark.remote_data
def test_process_srs(test_config):
    _ = process_srs(test_config)
    assert True


@pytest.mark.remote_data
def test_process_mdi(test_config):
    _ = process_mdi(test_config)
    assert True


@pytest.mark.remote_data
def test_process_hmi(test_config):
    _ = process_hmi(test_config)
    assert True


def test_extract_flare_counts():
    rows = [
        (Time("2024-01-01T00:00:00"), 1, "C1.1"),
        (Time("2024-01-01T23:59:59"), 1, "C1.2"),
        (Time("2024-01-02T00:00:00"), 1, "C1.1"),
        (Time("2024-01-02T10:00:00"), 1, "M1.1"),
        (Time("2024-01-02T23:00:00"), 2, "X1.1"),
    ]
    flares = Table(rows=rows, names=("peak_time", "noaa_number", "goes_class"))
    flare_counts = extract_flare_counts(flares)

    assert len(flare_counts) == 3
    assert flare_counts.iloc[0]["noaa_number"] == 1
    assert flare_counts.iloc[0]["date"] == date(2024, 1, 1)
    assert flare_counts.iloc[0]["C"] == 2

    assert flare_counts.iloc[1]["noaa_number"] == 1
    assert flare_counts.iloc[1]["date"] == date(2024, 1, 2)
    assert flare_counts.iloc[1]["M"] == 1

    assert flare_counts.iloc[2]["noaa_number"] == 2
    assert flare_counts.iloc[2]["date"] == date(2024, 1, 2)
    assert flare_counts.iloc[2]["X"] == 1
