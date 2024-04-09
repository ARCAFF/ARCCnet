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

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import QTable

from arccnet.data_generation.region_detection import RegionDetection


@pytest.fixture(scope="session")
def temp_path_fixture(request):
    temp_dir = tempfile.mkdtemp()  # Create temporary directory

    # Create raw and processed folders
    fd_data_dir = Path(temp_dir) / Path("fulldisk")
    fd_data_dir.mkdir()
    cutout_data_dir = Path(temp_dir) / Path("cutout")
    cutout_data_dir.mkdir()

    def cleanup():
        shutil.rmtree(temp_dir)  # Clean up the temporary directory

    request.addfinalizer(cleanup)  # noqa PT021
    return (Path(temp_dir), Path(fd_data_dir), Path(cutout_data_dir))


@pytest.fixture
def sunpy_hmi_copies(temp_path_fixture):
    n = 10
    fulldisk_copies = []
    cutout_copies = []
    target_times = []

    for _ in range(n // 2):
        hmi_data = sunpy.map.Map(sunpy.data.sample.HMI_LOS_IMAGE)
        filename = f"{uuid.uuid4()}.fits"
        # save the files to the /raw/ folder in var
        file_path = temp_path_fixture[1] / filename
        hmi_data.save(file_path)
        fulldisk_copies.append(file_path)
        target_times.append(hmi_data.date.to_datetime())

    for _ in range(n // 2):
        hmi_data = sunpy.map.Map(sunpy.data.sample.HMI_LOS_IMAGE)
        # this should be fiinnee
        top_right = SkyCoord(
            np.random.uniform(-500, 500) * u.arcsec,
            np.random.uniform(-500, 500) * u.arcsec,
            frame=hmi_data.coordinate_frame,
        )
        bottom_left = SkyCoord(
            np.random.uniform(-500, 500) * u.arcsec,
            np.random.uniform(-500, 500) * u.arcsec,
            frame=hmi_data.coordinate_frame,
        )
        submap = hmi_data.submap(bottom_left, top_right=top_right)
        filename = f"{uuid.uuid4()}.fits"
        # save the files to the /raw/ folder in var
        file_path = temp_path_fixture[2] / filename
        submap.save(file_path)
        cutout_copies.append(file_path)

    return target_times, fulldisk_copies, cutout_copies


@pytest.fixture
def qtable_object(sunpy_hmi_copies):
    data = {
        "target_time": sunpy_hmi_copies[0],
        "processed_path": sunpy_hmi_copies[1],
        "path_arc": sunpy_hmi_copies[2],
    }

    data["processed_path"] = [str(dpth) for dpth in data["processed_path"]]
    data["path_arc"] = [str(dpth_a) for dpth_a in data["path_arc"]]

    df = pd.DataFrame(data)
    qt = QTable.from_pandas(df)  # RegionDetectionTable?
    return qt


@pytest.fixture
def region_detection_fixture(temp_path_fixture, qtable_object):
    # Create a temporary test CSV file with the test data
    test_parq_path = temp_path_fixture[0] / "test_data.parq"
    test_parq_path.parent.mkdir(exist_ok=True, parents=True)
    qtable_object.write(test_parq_path, format="parquet", overwrite=True)

    # Initialize RegionDetection with the test CSV file
    test_table = QTable.read(test_parq_path)
    region_detection = RegionDetection(
        table=test_table,
        col_group_path="processed_path",
        col_cutout_path="path_arc",
    )

    return region_detection


# Test methods using the fixture
@pytest.mark.remote_data
def test_initialization(region_detection_fixture):
    # Test if RegionDetection initializes correctly
    assert region_detection_fixture._loaded_data is not None


@pytest.mark.remote_data
def test_get_bboxes(region_detection_fixture, qtable_object):
    # Test the get_bboxes method using the fixture
    bboxes = region_detection_fixture.get_bboxes()
    assert len(bboxes[0]) == len(qtable_object["path_arc"])
