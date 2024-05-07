import uuid
import shutil
import tempfile
from pathlib import Path
from datetime import datetime

import numpy as np
import pytest
import sunpy
import sunpy.data.sample
import sunpy.map

from astropy.table import MaskedColumn, QTable

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


@pytest.mark.remote_data
@pytest.fixture
def sunpy_hmi_copies(temp_path_fixture):
    n = 5
    hmi_copies = []

    for _ in range(n):
        hmi_data = sunpy.map.Map(sunpy.data.sample.HMI_LOS_IMAGE)
        filename = f"{uuid.uuid4()}.fits"
        # save the files to the /raw/ folder in var
        file_path = temp_path_fixture[1] / filename
        hmi_data.save(file_path)
        hmi_copies.append(str(file_path))

    return hmi_copies


@pytest.fixture
def data_qtable(sunpy_hmi_copies):
    return QTable(
        {
            "target_time": MaskedColumn(data=[datetime.now()] * len(sunpy_hmi_copies)),
            "url": MaskedColumn(data=(["something"] * len(sunpy_hmi_copies))),
            "path": MaskedColumn(data=sunpy_hmi_copies),
        }
    )


@pytest.mark.remote_data  # downloads sample data
@pytest.mark.parametrize("use_multiprocessing", [True, False])
def test_process_data(data_qtable, temp_path_fixture, use_multiprocessing):
    """
    Test Processing without multiprocessing
    """
    # Save the dataframe to a temporary CSV file
    temp_dir_path, _, process_data_path = temp_path_fixture

    # check that the processed dir is empty
    assert not list(process_data_path.glob("*.fits"))

    # Initialize the MagnetogramProcessor
    mp = MagnetogramProcessor(
        table=data_qtable,
        save_path=temp_dir_path,
        column_name="path",
    )

    merged_table = mp.process(use_multiprocessing=use_multiprocessing, merge_col_prefix="processed_", overwrite=True)

    raw_paths = merged_table["path"]
    processed_paths = merged_table["processed_path"]
    # sanity check to just ensure the raw and processed paths aren't the same
    assert (np.array(raw_paths) != np.array(processed_paths)).any()

    # sunpy maps
    for rpath, ppath in zip(raw_paths, processed_paths):
        # load raw, process
        raw_map = sunpy.map.Map(rpath)
        raw_map.data[~sunpy.map.coordinate_is_on_solar_disk(sunpy.map.all_coordinates_from_map(raw_map))] = 0.0
        processed_raw_map = raw_map.rotate()

        # ... save, read, delete
        processed_raw_path = process_data_path / Path("raw_processed.fits")
        save_compressed_map(processed_raw_map, path=processed_raw_path)
        loaded_prd = sunpy.map.Map(processed_raw_path)
        processed_raw_path.unlink()
        # load processed, delete
        ppath = Path(ppath)
        processed_map = sunpy.map.Map(ppath)
        ppath.unlink()

        assert np.array_equal(loaded_prd.data, processed_map.data, equal_nan=True)
