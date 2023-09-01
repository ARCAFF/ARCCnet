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

    for _ in range(n // 2):
        hmi_data = sunpy.map.Map(sunpy.data.sample.HMI_LOS_IMAGE)
        filename = f"{uuid.uuid4()}.fits"
        # save the files to the /raw/ folder in var
        file_path = temp_path_fixture[1] / filename
        hmi_data.save(file_path)
        fulldisk_copies.append(file_path)

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

    return fulldisk_copies, cutout_copies


@pytest.fixture
def pd_dataframe(sunpy_hmi_copies):
    data = {
        "download_path": sunpy_hmi_copies[0],
        "download_path_arc": sunpy_hmi_copies[1],
    }

    df = pd.DataFrame(data)
    return df


@pytest.fixture
def region_detection_fixture(temp_path_fixture, pd_dataframe):
    # Create a temporary test CSV file with the test data
    test_csv_path = temp_path_fixture[0] / "test_data.csv"
    pd_dataframe.to_csv(test_csv_path, index=False)

    # Initialize RegionDetection with the test CSV file
    region_detection = RegionDetection(filename=test_csv_path)

    return region_detection


# Test methods using the fixture
def test_initialization(region_detection_fixture):
    # Test if RegionDetection initializes correctly
    assert region_detection_fixture.loaded_data is not None


def test_get_bboxes(region_detection_fixture, pd_dataframe):
    # Test the get_bboxes method using the fixture
    bboxes = region_detection_fixture.get_bboxes(
        region_detection_fixture.loaded_data, "download_path", "download_path_arc"
    )

    # Add your assertions here to validate the results
    assert len(bboxes) == len(pd_dataframe["download_path_arc"])
