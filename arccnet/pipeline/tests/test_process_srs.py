import pickle

import pytest

import arccnet
from arccnet.pipeline.main import process_hmi, process_mdi, process_srs


@pytest.fixture(scope="session")
def data_dir(tmp_path):
    return tmp_path


@pytest.fixture(scope="session")
def test_config(tmp_path):
    # So don't change global config work on copy (copy(arccnet.config) didn't work
    rep = pickle.dumps(arccnet.config)
    config = pickle.loads(rep)
    config.set("paths", "data_root", str(tmp_path))
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
