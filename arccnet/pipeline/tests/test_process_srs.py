import pickle

import pytest

import arccnet
from arccnet.pipeline.main import process_hmi, process_mdi, process_srs


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("arccnet_testing")


@pytest.fixture(scope="session")
def test_config(data_dir):
    # So don't change global config work on copy (copy(arccnet.config) didn't work
    rep = pickle.dumps(arccnet.config)
    config = pickle.loads(rep)
    config.set("paths", "data_root", str(data_dir))
    config.set("general", "start_date", "2010-04-15T00:00:00")
    config.set("general", "end_date", "2010-05-15T00:00:00")
    # !TODO this shouldn't need to be a month of data to run
    # understand why we can't use just a day.
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
