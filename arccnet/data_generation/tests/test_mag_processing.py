import pandas as pd
import pytest
import sunpy
import sunpy.data.sample
import sunpy.map

from arccnet.data_generation.mag_processing import MagnetogramProcessor


@pytest.fixture
def mag_processor_default():
    return MagnetogramProcessor()


@pytest.fixture
def pd_dataframe():
    data = {
        "url_hmi": [
            sunpy.data.sample.HMI_LOS_IMAGE,
        ],  # Repeating the same value for demonstration
        "url_mdi": [
            sunpy.data.sample.HMI_LOS_IMAGE,
        ],
        "other": ["column"],
    }

    return pd.DataFrame(data)


# def test_read_datapaths(mag_processor_default, pd_dataframe):
#     # test the reading of datapaths
#     with tempfile.TemporaryDirectory() as temp_dir:
#         temp_dir_path = Path(temp_dir)
#         # save csv to file
#         csv_path = temp_dir_path / "data.csv"
#         pd_dataframe.to_csv(csv_path)

#         paths = mag_processor_default._read_datapaths(columns=["url_hmi", "url_mdi"], csv_file=csv_path)
#         assert all(isinstance(item, Path) for item in paths)


# def test_process_data_without_multiprocessing(mag_processor_default):
#     # test the reading of datapaths
#     with tempfile.TemporaryDirectory() as temp_dir:
#         mag_processor_default._process_data(paths, save_path=temp_dir, use_multiprocessing=False)

#         for path in paths:

#     isinstance(mag_processor_default._read_datapaths[0], Path)


def test_process_and_save_data():
    pass


def test_rotate_datum():
    pass
