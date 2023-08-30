import pytest

from arccnet.pipeline.main import process_srs


@pytest.fixture(scope="session")
def data_dir(tmp_path):
    return tmp_path


def test_process_srs(tmp_path):
    config = {"paths": {"data_root": tmp_path}, "dates": {"start_date": "2022-01-01", "end_date": "2022-02-01"}}
    query, results, raw_catalog, processed_catalog, clean_catalog = process_srs(config)
    assert True
