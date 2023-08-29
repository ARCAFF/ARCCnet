import shutil
import tempfile
from pathlib import Path

import pandas as pd
import pytest
import sunpy
import sunpy.data.sample
import sunpy.map

from arccnet.data_generation.mag_processing import MagnetogramProcessor
from arccnet.data_generation.utils.utils import save_compressed_map


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
        csv_path = temp_dir_path / Path("data.csv")
        # save csv to temporary directory
        pd_dataframe.to_csv(csv_path)
        # load and save the sunpy file to the temp dir
        sunpy.map.Map(sunpy.data.sample.HMI_LOS_IMAGE).save(temp_dir_path / sunpy.data.sample.HMI_LOS_IMAGE.name)

        mp = MagnetogramProcessor(
            csv_file=csv_path,
            columns=["url_hmi", "url_mdi"],
            processed_data_dir=temp_dir_path,
            raw_data_dir=temp_dir_path,
        )
        assert all(isinstance(item, Path) for item in mp.paths)


def test_process_data_without_multiprocessing(pd_dataframe):
    """
    Test Processing without multiprocessing
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Save the dataframe to a temporary CSV file
        csv_path = temp_path / "data.csv"
        pd_dataframe.to_csv(csv_path)

        # Create the processed and raw data directories
        processed_temp_dir_path = temp_path / Path("processed")
        processed_temp_dir_path.mkdir()
        raw_temp_dir_path = temp_path / Path("raw")
        raw_temp_dir_path.mkdir()

        # Copy the HMI_LOS_IMAGE to the raw temporary directory
        hmi_los_image_path = raw_temp_dir_path / sunpy.data.sample.HMI_LOS_IMAGE.name
        shutil.copy(sunpy.data.sample.HMI_LOS_IMAGE, hmi_los_image_path)

        # Initialize the MagnetogramProcessor
        mp = MagnetogramProcessor(
            csv_file=csv_path,
            columns=["url_hmi", "url_mdi"],
            processed_data_dir=processed_temp_dir_path,
            raw_data_dir=raw_temp_dir_path,
        )

        # Process the data without multiprocessing
        mp.process_data(use_multiprocessing=False)

        # Construct paths for comparison
        processed_path = [processed_temp_dir_path / path.name for path in mp.paths][0]
        processed_data = sunpy.map.Map(processed_path)

        # Repeat the processing steps manually and save/load due to compression artifacts
        raw_path = [raw_temp_dir_path / path.name for path in mp.paths][0]
        processed_raw_data = sunpy.map.Map(raw_path)
        processed_raw_data = processed_raw_data.rotate()
        processed_raw_data.data[
            ~sunpy.map.coordinate_is_on_solar_disk(sunpy.map.all_coordinates_from_map(processed_raw_data))
        ] = 0.0
        processed_raw_path = processed_temp_dir_path / Path("raw_processed.fits")
        save_compressed_map(processed_raw_data, path=processed_raw_path)
        loaded_prd = sunpy.map.Map(processed_raw_path)

        # Assert processed data equality
        assert (loaded_prd.data == processed_data.data).all()


def test_process_data_with_multiprocessing(pd_dataframe):
    """
    Test Processing withmultiprocessing
    """
    pass


def test_process_and_save_data():
    pass


def test_rotate_datum():
    pass
