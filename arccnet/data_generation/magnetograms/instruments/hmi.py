import datetime

import arccnet.data_generation.utils.default_variables as dv
from arccnet.data_generation.magnetograms.base_magnetogram import BaseMagnetogram
from arccnet.data_generation.magnetograms.utils import datetime_to_jsoc

__all__ = ["HMIMagnetogram", "HMIContinuum"]


class HMIMagnetogram(BaseMagnetogram):
    def __init__(self):
        super().__init__()
        # According to JSOC: [DATE-OBS] DATE_OBS = T_OBS - EXPTIME/2.0

    def generate_drms_query(self, start_time: datetime.datetime, end_time: datetime.datetime, frequency="1d") -> str:
        """
        Returns
        -------
        str:
            JSOC Query string
        """
        # https://github.com/sunpy/drms/issues/98
        # https://github.com/sunpy/drms/issues/37
        # want to deal with quality after obtaining the data
        return f"{self.series_name}[{datetime_to_jsoc(start_time)}-{datetime_to_jsoc(end_time)}@{frequency}]"  # [? QUALITY=0 ?]"

    @property
    def series_name(self) -> str:
        """
        Returns
        -------
        str:
            JSOC series name
        """
        return "hmi.M_720s"

    @property
    def date_format(self) -> str:
        """
        Returns
        -------
        str:
            HMI date string format
        """
        return dv.HMI_DATE_FORMAT

    @property
    def segment_column_name(self) -> str:
        """
        Returns
        -------
        str:
            Name of the HMI data segment
        """
        return dv.HMI_SEG_COL

    @property
    def metadata_save_location(self) -> str:
        """
        Returns
        -------
        str:
            HMI directory path
        """
        return dv.HMI_MAG_RAW_CSV


class HMIContinuum(HMIMagnetogram):
    def __init__(self):
        super().__init__()

    @property
    def series_name(self) -> str:
        """
        Returns
        -------
        str:
            JSOC series name
        """
        return "hmi.Ic_720s"

    @property
    def segment_column_name(self) -> str:
        """
        Returns
        -------
        str:
            Name of the HMI data segment
        """
        return "continuum"

    @property
    def metadata_save_location(self) -> str:
        """
        Returns
        -------
        str:
            HMI directory path
        """
        return dv.HMI_IC_RAW_CSV
