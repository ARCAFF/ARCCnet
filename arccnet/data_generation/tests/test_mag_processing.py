import uuid
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import sunpy
import sunpy.data.sample
import sunpy.map

from arccnet.data_generation.mag_processing import MagnetogramProcessor
from arccnet.data_generation.utils.utils import save_compressed_map


@pytest.fixture(scope="session")
def temp_path_fixture(request):
    temp_dir = tempfile.mkdtemp()  # Create temporary directory

    # Create raw and processed folders
    raw_data_dir = Path(temp_dir) / Path("raw")
    raw_data_dir.mkdir()
    processed_data_dir = Path(temp_dir) / Path("processed")
    processed_data_dir.mkdir()

    def cleanup():
        shutil.rmtree(temp_dir)  # Clean up the temporary directory

    request.addfinalizer(cleanup)  # noqa PT021
    return (Path(temp_dir), Path(raw_data_dir), Path(processed_data_dir))


@pytest.fixture
def sunpy_hmi_copies(temp_path_fixture):
    n = 10
    hmi_copies = []

    for _ in range(n):
        hmi_data = sunpy.map.Map(sunpy.data.sample.HMI_LOS_IMAGE)
        filename = f"{uuid.uuid4()}.fits"
        # save the files to the /raw/ folder in var
        file_path = temp_path_fixture[1] / filename
        hmi_data.save(file_path)
        hmi_copies.append(file_path)

    return hmi_copies


@pytest.fixture
def pd_dataframe(sunpy_hmi_copies):
    n = len(sunpy_hmi_copies)
    half_n = n // 2

    data = {
        "url_hmi": sunpy_hmi_copies[:half_n],
        "url_mdi": sunpy_hmi_copies[half_n:],
        "other": ["column"] * half_n,
    }

    df = pd.DataFrame(data)
    return df


def test_read_datapaths(pd_dataframe, temp_path_fixture):
    # test the reading of datapaths
    temp_dir_path, _, process_data_path = temp_path_fixture
    # save to the base of the tempdir
    # data is in raw/
    csv_path = temp_dir_path / Path("data.csv")
    pd_dataframe.to_csv(csv_path, index=False)

    # Initialize the MagnetogramProcessor
    mp = MagnetogramProcessor(
        csv_in_file=csv_path,
        columns=["url_hmi", "url_mdi"],
        processed_data_dir=process_data_path,
        process_data=False,
        use_multiprocessing=False,
    )

    assert all(isinstance(item, Path) for item in mp.paths)


@pytest.mark.parametrize("use_multiprocessing", [True, False])
def test_process_data(pd_dataframe, temp_path_fixture, use_multiprocessing):
    """
    Test Processing without multiprocessing
    """
    # Save the dataframe to a temporary CSV file
    temp_dir_path, _, process_data_path = temp_path_fixture
    csv_path = temp_dir_path / Path("data.csv")
    pd_dataframe.to_csv(csv_path, index=False)
    csv_out = temp_dir_path / Path("procesed_data.csv")

    # check that the processed dir is empty
    assert not list(process_data_path.glob("*.fits"))

    # Initialize the MagnetogramProcessor
    mp = MagnetogramProcessor(
        csv_in_file=csv_path,
        csv_out_file=csv_out,
        columns=["url_hmi", "url_mdi"],
        processed_data_dir=process_data_path,
        process_data=True,
        use_multiprocessing=use_multiprocessing,
    )

    # Construct paths for comparison
    raw_paths = mp.paths
    processed_paths = mp.processed_paths

    # sanity check to just ensure the raw and processed paths aren't the same
    assert (np.array(raw_paths) != np.array(processed_paths)).any()

    # sunpy maps
    for rpath, ppath in zip(raw_paths, processed_paths):
        # load raw, process
        raw_map = sunpy.map.Map(rpath)
        processed_raw_map = raw_map.rotate()
        processed_raw_map.data[
            ~sunpy.map.coordinate_is_on_solar_disk(sunpy.map.all_coordinates_from_map(processed_raw_map))
        ] = 0.0

        # ... save, read, delete
        processed_raw_path = process_data_path / Path("raw_processed.fits")
        save_compressed_map(processed_raw_map, path=processed_raw_path)
        loaded_prd = sunpy.map.Map(processed_raw_path)
        processed_raw_path.unlink()
        # load processed, delete
        processed_map = sunpy.map.Map(ppath)
        ppath.unlink()

        assert (loaded_prd.data == processed_map.data).all()
