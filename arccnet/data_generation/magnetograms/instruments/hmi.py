import re
import datetime

import arccnet.data_generation.utils.default_variables as dv
from arccnet.data_generation.magnetograms.base_magnetogram import BaseMagnetogram
from arccnet.data_generation.magnetograms.utils import datetime_to_jsoc

__all__ = ["HMILOSMagnetogram", "HMIMagnetogramNRT", "HMIContinuum", "HMISHARPs"]


class HMILOSMagnetogram(BaseMagnetogram):
    def __init__(self):
        super().__init__()
        # According to JSOC: [DATE-OBS] DATE_OBS = T_OBS - EXPTIME/2.0

    def generate_drms_query(self, start_time: datetime.datetime, end_time: datetime.datetime, frequency="1d") -> str:
        """
        Generate a JSOC query string for requesting observations within a specified time range.

        Parameters
        ----------
        start_time : datetime.datetime
            A datetime object representing the start time of the requested observations.

        end_time : datetime.datetime
            A datetime object representing the end time of the requested observations.

        frequency : str, optional
            A string representing the frequency of observations. Default is "1d" (1 day).
            Valid frequency strings can be specified, such as "1h" for 1 hour, "15T" for 15 minutes,
            "1M" for 1 month, "1Y" for 1 year, and more. Refer to the pandas documentation for a complete
            list of valid frequency strings: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases

        Returns
        -------
        str
            The JSOC query string for retrieving the specified observations.
        """
        # https://github.com/sunpy/drms/issues/98
        # https://github.com/sunpy/drms/issues/37
        # want to deal with quality after obtaining the data
        return f"{self.series_name}[{datetime_to_jsoc(start_time)}-{datetime_to_jsoc(end_time)}@{frequency}]"  # [? QUALITY=0 ?]"

    @property
    def _get_matching_info_from_record(self, record: str) -> tuple[tuple[str, str], list[str]]:
        # Extract both the date and HARPNUM from the specific record format
        # For example, you can use regular expressions to extract these values
        # Here's a hypothetical implementation, adjust it to match your actual data format
        match = re.match(r"\[(.*?)\]", record)
        if match:
            date = match.group(1)
            merge_columns = ["T_REC"]
            return (date), merge_columns
        else:
            raise ValueError()

    @property
    def series_name(self) -> str:
        """
        Get the JSOC series name.

        Returns
        -------
        str
            The JSOC series name.
        """
        return "hmi.M_720s"

    @property
    def date_format(self) -> str:
        """
        Get the HMI date string format.

        Returns
        -------
        str
            The HMI date string format.
        """
        return dv.HMI_DATE_FORMAT

    @property
    def segment_column_name(self) -> str:
        """
        Get the name of the HMI data segment.

        Returns
        -------
        str
            The name of the HMI data segment.
        """
        return dv.HMI_SEG_COL

    @property
    def metadata_save_location(self) -> str:
        """
        Get the HMI directory path for saving metadata.

        Returns
        -------
        str
            The HMI directory path for saving metadata.
        """
        return dv.HMI_MAG_RAW_CSV


class HMIBMagnetogram(HMILOSMagnetogram):
    def __init__(self):
        super().__init__()

    @property
    def series_name(self) -> str:
        """
        Get the JSOC series name.

        Returns
        -------
        str
            The JSOC series name.
        """
        return "hmi.B_720s"


class HMIContinuum(HMILOSMagnetogram):
    def __init__(self):
        super().__init__()

    @property
    def series_name(self) -> str:
        """
        Get the JSOC series name.

        Returns
        -------
        str
            The JSOC series name.
        """
        return "hmi.Ic_720s"

    @property
    def segment_column_name(self) -> str:
        """
        Get the name of the HMI data segment.

        Returns
        -------
        str
            The name of the HMI data segment.
        """
        return "continuum"

    @property
    def metadata_save_location(self) -> str:
        """
        Get the HMI directory path for saving metadata.

        Returns
        -------
        str
            The HMI directory path for saving metadata.
        """
        return dv.HMI_IC_RAW_CSV


class HMISHARPs(HMILOSMagnetogram):
    def __init__(self):
        super().__init__()

        # /Users/pjwright/Documents/work/ARCCnet/arccnet/data_generation/magnetograms/base_magnetogram.py:103:
        # PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert`
        # many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead.
        # To get a de-fragmented frame, use `newframe = frame.copy()` keys["magnetogram_fits"] = magnetogram_fits

    def generate_drms_query(self, start_time: datetime.datetime, end_time: datetime.datetime, frequency="1d") -> str:
        """
        Generate a JSOC query string for requesting observations within a specified time range.

        Parameters
        ----------
        start_time : datetime.datetime
            A datetime object representing the start time of the requested observations.

        end_time : datetime.datetime
            A datetime object representing the end time of the requested observations.

        frequency : str, optional
            A string representing the frequency of observations. Default is "1d" (1 day).
            Valid frequency strings can be specified, such as "1h" for 1 hour, "15T" for 15 minutes,
            "1M" for 1 month, "1Y" for 1 year, and more. Refer to the pandas documentation for a complete
            list of valid frequency strings: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases

        Returns
        -------
        str
            The JSOC query string for retrieving the specified observations.
        """
        # for SHARPs this needs to be of the for
        # `hmi.sharp_720s[<HARPNUM>][2010.05.01_00:00:00_TAI]`
        return f"{self.series_name}[][{datetime_to_jsoc(start_time)}-{datetime_to_jsoc(end_time)}@{frequency}]"  # [? QUALITY=0 ?]"

    @property
    def series_name(self) -> str:
        """
        Get the JSOC series name.

        Returns
        -------
        str
            The JSOC series name.
        """
        return "hmi.sharp_720s"

    @property
    def segment_column_name(self) -> str:
        """
        Get the name of the HMI data segment.

        Returns
        -------
        str
            The name of the HMI data segment.
        """
        return "bitmap"

    @property
    def metadata_save_location(self) -> str:
        """
        Get the HMI directory path for saving metadata.

        Returns
        -------
        str
            The HMI directory path for saving metadata.
        """
        return dv.HMI_SHARPS_RAW_CSV


class HMIMagnetogramNRT(HMILOSMagnetogram):
    def __init__(self):
        raise NotImplementedError("Placeholder class for NRT HMI Magnetograms")
