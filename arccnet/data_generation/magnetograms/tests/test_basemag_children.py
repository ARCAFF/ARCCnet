"""
Testing for the instrument classes that inherit from BaseMagnetogram
"""

import datetime

import numpy as np
import pandas as pd
import pytest

from arccnet.data_generation.magnetograms.instruments.hmi import HMILOSMagnetogram, HMISHARPs
from arccnet.data_generation.magnetograms.instruments.mdi import MDILOSMagnetogram, MDISMARPs


@pytest.mark.remote_data
@pytest.mark.parametrize(
    ("magnetogram_class", "expected_query", "frequency"),
    [
        (HMILOSMagnetogram, "hmi.M_720s[2023.01.01_00:00:00-2023.01.05_00:00:00@1d]", "1d"),
        (MDILOSMagnetogram, "mdi.fd_M_96m_lev182[2023.01.01_00:00:00-2023.01.05_00:00:00@1d]", "1d"),
        (HMILOSMagnetogram, "hmi.M_720s[2023.01.01_00:00:00-2023.01.05_00:00:00@6h]", "6h"),
        (MDILOSMagnetogram, "mdi.fd_M_96m_lev182[2023.01.01_00:00:00-2023.01.05_00:00:00@3h]", "3h"),
    ],
)
def test_generate_drms_query(magnetogram_class, expected_query, frequency):
    # Test the generate_drms_query method of HMILOSMagnetogram and MDILOSMagnetogram
    magnetogram = magnetogram_class()
    query = magnetogram.generate_drms_query(
        datetime.datetime(2023, 1, 1),
        datetime.datetime(2023, 1, 5),
        frequency=frequency,
    )
    assert query == expected_query


@pytest.mark.remote_data
@pytest.mark.parametrize("magnetogram_class", [HMILOSMagnetogram, MDILOSMagnetogram])
def test_fulldisk_get_matching_info_from_record(magnetogram_class):
    # Test the _get_matching_info_from_record method of HMILOSMagnetogram and MDILOSMagnetogram
    magnetogram = magnetogram_class()
    records = pd.Series(
        [
            "hmi.generic_name[2023.01.01_00:00:00_TAI]",
            "hmi.generic_name_also[2023.01.02_00:00:00_TAI][20]",
            "[2023.01.03_00:00:00_TAI][20]",
            "[2023.01.04_00:00:00_TAI]",
            "2023.01.05_00:00:00_TAI",
        ]
    )
    extracted_info = magnetogram._get_matching_info_from_record(records)
    assert extracted_info.equals(
        pd.DataFrame(
            {
                "T_REC": [
                    "2023.01.01_00:00:00_TAI",
                    "2023.01.02_00:00:00_TAI",
                    "2023.01.03_00:00:00_TAI",
                    "2023.01.04_00:00:00_TAI",
                    np.nan,
                ]
            }
        )
    )


@pytest.mark.remote_data
@pytest.mark.parametrize(
    ("magnetogram_class", "expected_columns"),
    [
        (HMISHARPs, ["HARPNUM", "T_REC"]),
        (MDISMARPs, ["TARPNUM", "T_REC"]),
    ],
)
def test_cutout_get_matching_info_from_record(magnetogram_class, expected_columns):
    # Test the _get_matching_info_from_record method of HMISHARPs and MDISMARPs
    magnetogram = magnetogram_class()
    records = pd.Series(
        [
            "inst.generic_name[01][2023.01.01_00:00:00_TAI]",
            "inst.generic_name[10][2023.01.02_00:00:00_TAI][20]",
            "[20][2023.01.03_00:00:00_TAI][20]instr.generic_name",
            "[30][2023.01.04_00:00:00_TAI]inst",
            "[2023.01.05_00:00:00_TAI]",
            "2023.01.06_00:00:00_TAI",
        ]
    )
    extracted_info = magnetogram._get_matching_info_from_record(records)
    extracted_values = extracted_info[expected_columns]

    # Define the generic column names and mapping for each class
    column_mapping = {
        HMISHARPs: {"XARPNUM": "HARPNUM"},
        MDISMARPs: {"XARPNUM": "TARPNUM"},
    }

    expected_values = pd.DataFrame(
        {
            "XARPNUM": [1, 10, 20, 30, np.nan, np.nan],
            "T_REC": [
                "2023.01.01_00:00:00_TAI",
                "2023.01.02_00:00:00_TAI",
                "2023.01.03_00:00:00_TAI",
                "2023.01.04_00:00:00_TAI",
                np.nan,
                np.nan,
            ],
        }
    )

    # !TODO understand if this has unintended consequences
    expected_values["XARPNUM"] = expected_values["XARPNUM"].astype("Int64")
    # the column is cast to Int64 as it can handle NaN values (as pd.NA)
    # while a string (object) column can handle np.nan

    # Rename the XARPNUM column based on the class
    expected_values = expected_values.rename(columns={"XARPNUM": column_mapping[magnetogram_class]["XARPNUM"]})
    # assert extracted_values.equals(expected_values)
    pd.testing.assert_frame_equal(extracted_values, expected_values)


@pytest.mark.remote_data
@pytest.mark.parametrize(
    ("magnetogram_class", "batch_frequency", "start_date", "end_date"),
    [
        (MDILOSMagnetogram, 4, datetime.datetime(1997, 1, 1), datetime.datetime(1997, 4, 1)),
        (MDILOSMagnetogram, 2, datetime.datetime(1999, 1, 1), datetime.datetime(1999, 4, 1)),
        (HMILOSMagnetogram, 1, datetime.datetime(2020, 1, 1), datetime.datetime(2020, 4, 1)),
        (HMILOSMagnetogram, 3, datetime.datetime(2021, 1, 1), datetime.datetime(2021, 4, 1)),
    ],
)
def test_fetch_metadata_v_batch(magnetogram_class, batch_frequency, start_date, end_date):
    # Test that the fetch_metadata and fetch_metadata_batch methods of
    # HMILOSMagnetogram and MDILOSMagnetogram provide the same output
    magnetogram = magnetogram_class()
    single_query = magnetogram.fetch_metadata(
        start_date=start_date,
        end_date=end_date,
        batch_frequency=batch_frequency,
        dynamic_columns=["url"],
    ).drop(columns="url")  # drop 'url' as it's dynamic
    batched_query = magnetogram.fetch_metadata_batch(
        start_date=start_date,
        end_date=end_date,
    ).drop(columns="url")  # drop 'url' as it's dynamic
    # assert single_query.equals(batched_query)
    pd.testing.assert_frame_equal(single_query, batched_query)


# Probably not really needed...
# HMI
@pytest.mark.remote_data
class TestHMILOSProperties:
    @pytest.fixture
    def hmi_instance(self):
        return HMILOSMagnetogram()  # Create an instance of your class for testing

    def test_series_name(self, hmi_instance):
        assert hmi_instance.series_name == "hmi.M_720s"

    def test_date_format(self, hmi_instance):
        assert hmi_instance.date_format == "%Y-%m-%dT%H:%M:%S.%fZ"

    def test_segment_column_name(self, hmi_instance):
        assert hmi_instance.segment_column_name == "magnetogram"

    def test_type(self, hmi_instance):
        assert hmi_instance._type == hmi_instance.__class__.__name__


@pytest.mark.remote_data
class TestHMISHARPsProperties:
    @pytest.fixture
    def sharp_instance(self):
        return HMISHARPs()  # Create an instance of your class for testing

    def test_series_name(self, sharp_instance):
        assert sharp_instance.series_name == "hmi.sharp_720s"

    def test_date_format(self, sharp_instance):
        assert sharp_instance.date_format == "%Y-%m-%dT%H:%M:%S.%fZ"

    def test_segment_column_name(self, sharp_instance):
        assert sharp_instance.segment_column_name == "bitmap"

    def test_type(self, sharp_instance):
        assert sharp_instance._type == sharp_instance.__class__.__name__


# MDI
@pytest.mark.remote_data
class TestMDILOSProperties:
    @pytest.fixture
    def mdi_instance(self):
        return MDILOSMagnetogram()  # Create an instance of your class for testing

    def test_series_name(self, mdi_instance):
        assert mdi_instance.series_name == "mdi.fd_M_96m_lev182"

    def test_date_format(self, mdi_instance):
        assert mdi_instance.date_format == "%Y-%m-%dT%H:%M:%SZ"

    def test_segment_column_name(self, mdi_instance):
        assert mdi_instance.segment_column_name == "data"

    def test_type(self, mdi_instance):
        assert mdi_instance._type == mdi_instance.__class__.__name__


@pytest.mark.remote_data
class TestMDISMARPsProperties:
    @pytest.fixture
    def smarp_instance(self):
        return MDISMARPs()  # Create an instance of your class for testing

    def test_series_name(self, smarp_instance):
        assert smarp_instance.series_name == "mdi.smarp_96m"

    def test_date_format(self, smarp_instance):
        assert smarp_instance.date_format == "%Y-%m-%dT%H:%M:%S.%fZ"

    def test_segment_column_name(self, smarp_instance):
        assert smarp_instance.segment_column_name == "bitmap"

    def test_type(self, smarp_instance):
        assert smarp_instance._type == smarp_instance.__class__.__name__
