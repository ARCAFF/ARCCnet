"""
Testing for the instrument classes that inherit from BaseMagnetogram
"""
import datetime

import numpy as np
import pandas as pd
import pytest

import arccnet.data_generation.utils.default_variables as dv
from arccnet.data_generation.magnetograms.instruments.hmi import HMILOSMagnetogram, HMISHARPs
from arccnet.data_generation.magnetograms.instruments.mdi import MDILOSMagnetogram, MDISMARPs


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
    print(query, expected_query)
    assert query == expected_query


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
                    pd.NA,
                ]
            }
        )
    )


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

    print(extracted_values)
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
                np.nan,  # !TODO do we actually want to return pd.NA?
            ],
        }
    )

    # !TODO understand if this has unintended consequences
    expected_values["XARPNUM"] = expected_values["XARPNUM"].astype("Int64")
    # the column is cast to Int64 as it can handle NaN values (as pd.NA)
    # while a string (object) column can handle np.nan

    # Rename the XARPNUM column based on the class
    expected_values = expected_values.rename(columns={"XARPNUM": column_mapping[magnetogram_class]["XARPNUM"]})
    assert extracted_values.equals(expected_values)


# Probably not really needed...
# HMI
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

    def test_metadata_save_location(self, hmi_instance):
        assert hmi_instance.metadata_save_location == dv.HMI_MAG_RAW_CSV


class TestHMISHARPsProperties:
    @pytest.fixture
    def hmi_instance(self):
        return HMISHARPs()  # Create an instance of your class for testing

    def test_series_name(self, hmi_instance):
        assert hmi_instance.series_name == "hmi.sharp_720s"

    def test_date_format(self, hmi_instance):
        assert hmi_instance.date_format == "%Y-%m-%dT%H:%M:%S.%fZ"

    def test_segment_column_name(self, hmi_instance):
        assert hmi_instance.segment_column_name == "bitmap"

    def test_metadata_save_location(self, hmi_instance):
        assert hmi_instance.metadata_save_location == dv.HMI_SHARPS_RAW_CSV


# MDI
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

    def test_metadata_save_location(self, mdi_instance):
        assert mdi_instance.metadata_save_location == dv.MDI_MAG_RAW_CSV


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

    def test_metadata_save_location(self, smarp_instance):
        assert smarp_instance.metadata_save_location == dv.MDI_SMARPS_RAW_CSV
