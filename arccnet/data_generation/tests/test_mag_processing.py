import tempfile
from pathlib import Path

import pandas as pd
import pytest
import sunpy
import sunpy.data.sample
import sunpy.map

from arccnet.data_generation.mag_processing import MagnetogramProcessor

data = {
    "url_hmi": [
        sunpy.data.sample.HMI_LOS_IMAGE,
    ],  # Repeating the same value for demonstration
    "url_mdi": [
        sunpy.data.sample.HMI_LOS_IMAGE,
    ],
    "other": ["column"],
}
pd_dataframe = pd.DataFrame(data)


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


def test_read_datapaths(pd_dataframe):
    # test the reading of datapaths
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        csv_path = temp_dir_path / "data.csv"
        # save csv to temporary directory
        pd_dataframe.to_csv(csv_path)
        # load and save the sunpy file to the temp dir
        sunpy.map.Map(sunpy.data.sample.HMI_LOS_IMAGE).save(temp_dir_path / sunpy.data.sample.HMI_LOS_IMAGE.name)

        mp = MagnetogramProcessor(
            csv_file=csv_path, columns=["url_hmi", "url_mdi"], processed_data_dir=temp_dir_path, raw_data_dir=temp_dir
        )
        assert all(isinstance(item, Path) for item in mp.paths)


#!TODO more testing around multiprocessing with additional data


def test_process_data_without_multiprocessing(pd_dataframe):
    # test the reading of datapaths
    with tempfile.TemporaryDirectory() as raw_temp_dir:
        raw_temp_dir_path = Path(raw_temp_dir)
        csv_path = raw_temp_dir_path / "data.csv"
        print(raw_temp_dir_path)
        # save csv to temporary directory
        pd_dataframe.to_csv(csv_path)
        # load and save the sunpy file to the temp dir
        sunpy.map.Map(sunpy.data.sample.HMI_LOS_IMAGE).save(raw_temp_dir_path / sunpy.data.sample.HMI_LOS_IMAGE.name)

        with tempfile.TemporaryDirectory() as processed_temp_dir:
            processed_temp_dir_path = Path(processed_temp_dir)

            print(f"processed_data_dir={processed_temp_dir_path}")
            print(f"raw_data_dir={raw_temp_dir_path}")
            mp = MagnetogramProcessor(
                csv_file=csv_path,
                columns=["url_hmi", "url_mdi"],
                processed_data_dir=processed_temp_dir_path,
                raw_data_dir=raw_temp_dir_path,
            )

            mp.process_data(use_multiprocessing=False)

            [raw_temp_dir_path / path.name for path in mp.paths]
            processed_paths = [processed_temp_dir_path / path.name for path in mp.paths]
            raw_data = sunpy.map.Map(processed_paths[0])
            # repeat the processing steps
            processed_raw_data = raw_data.rotate()
            processed_raw_data.data[
                ~sunpy.map.coordinate_is_on_solar_disk(sunpy.map.all_coordinates_from_map(processed_raw_data))
            ] = 0.0

            processed_data = sunpy.map.Map(processed_paths[0])

            assert (processed_raw_data.data == processed_data.data).all()


def test_process_and_save_data():
    pass


def test_rotate_datum():
    pass
