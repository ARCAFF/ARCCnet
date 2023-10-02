import pickle

import pytest

import arccnet
from arccnet.pipeline.main import process_srs


@pytest.fixture(scope="session")
def data_dir(tmp_path):
    return tmp_path


@pytest.mark.remote_data
def test_process_srs(tmp_path):
    # So don't change global config work on copy (copy(arccnet.config) didn't work
    rep = pickle.dumps(arccnet.config)
    config = pickle.loads(rep)
    config.set("paths", "data_root", str(tmp_path))
    config.set("general", "start_date", "2022-06-01T00:00:00")
    config.set("general", "end_date", "2022-06-02T00:00:00")
    query, results, raw_catalog, processed_catalog, clean_catalog = process_srs(config)
    assert True
