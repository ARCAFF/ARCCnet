import datetime

import pandas as pd

import arccnet.data_generation.utils.default_variables as dv
from arccnet.data_generation.magnetograms.base_magnetogram import BaseMagnetogram
from arccnet.data_generation.magnetograms.utils import datetime_to_jsoc

__all__ = ["MDILOSMagnetogram"]


class MDILOSMagnetogram(BaseMagnetogram):
    def __init__(self):
        super().__init__()

    def generate_drms_query(self, start_time: datetime.datetime, end_time: datetime.datetime, frequency="1d") -> str:
        """
        Returns
        -------
        str:
            JSOC Query string
        """
        # Line-of-sight magnetic field from 30-second observations in full-disc mode,
        # sampled either once in a minute or averaged over five consecutive minute samples.
        # Whether the data are form a single observation or an average of five is given by
        # the value of the keyword INTERVAL, the length of the sampling interval in seconds.
        # The data are acquired as part of the regular observing program.
        return f"{self.series_name}[{datetime_to_jsoc(start_time)}-{datetime_to_jsoc(end_time)}@{frequency}]"  # [? QUALITY=0 ?]"

    def _get_matching_info_from_record(self, records: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        # Extract both the date and HARPNUM from the specific record format
        # For example, you can use regular expressions to extract these values
        # Here's a hypothetical implementation, adjust it to match your actual data format]
        extracted_info = records.str.extract(r"\[(.*?)\]")

        # !TODO tidy this up by returning a df
        return extracted_info, ["T_REC"]

    @property
    def series_name(self) -> str:
        """
        Returns
        -------
        str:
            JSOC series name
        """
        return "mdi.fd_M_96m_lev182"

    @property
    def date_format(self) -> str:
        """
        Returns
        -------
        str:
            MDI date string format
        """
        return dv.MDI_DATE_FORMAT

    @property
    def segment_column_name(self) -> str:
        """
        Returns
        -------
        str:
            Name of the MDI data segment
        """
        return dv.MDI_SEG_COL

    @property
    def metadata_save_location(self) -> str:
        """
        Returns
        -------
        str:
            MDI directory path
        """
        return dv.MDI_MAG_RAW_CSV


class MDISMARPs(MDILOSMagnetogram):
    def __init__(self):
        super().__init__()

        # self._add_magnetogram_urls(keys, segs, url=dv.JSOC_BASE_URL, column_name="magnetogram_fits")
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
        return f"{self.series_name}[{datetime_to_jsoc(start_time)}-{datetime_to_jsoc(end_time)}@{frequency}]"  # [? QUALITY=0 ?]"

    def _get_matching_info_from_record(self, records: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        # Extract both the date and HARPNUM from the specific record format
        # For example, you can use regular expressions to extract these values
        # Here's a hypothetical implementation, adjust it to match your actual data format]
        extracted_info = records.str.extract(r"\[(.*?)\]\[(.*?)\]")
        extracted_info[0] = extracted_info[0].astype("Int64")  # !TODO fix this hack

        # !TODO tidy this up by returning a df
        return extracted_info, ["TARPNUM", "T_REC"]

    @property
    def series_name(self) -> str:
        """
        Get the JSOC series name.

        Returns
        -------
        str
            The JSOC series name.
        """
        return "mdi.smarp_96m"

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
        return dv.HMI_SMARPS_RAW_CSV
