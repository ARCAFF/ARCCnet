import datetime

import pandas as pd

from arccnet import config
from arccnet.data_generation.magnetograms.base_magnetogram import BaseMagnetogram
from arccnet.data_generation.magnetograms.utils import datetime_to_jsoc

__all__ = ["HMILOSMagnetogram", "HMIMagnetogramNRT", "HMIContinuum", "HMISHARPs"]


class HMILOSMagnetogram(BaseMagnetogram):
    def __init__(self):
        super().__init__()

    def generate_drms_query(self, start_time: datetime.datetime, end_time: datetime.datetime, frequency="1d") -> str:
        """
        Generate a JSOC query string for requesting observations within a specified time range.

        Parameters
        ----------
        start_time : datetime.datetime
            A datetime object representing the start time of the requested observations.

        end_time : datetime.datetime
            A datetime object representing the end time of the requested observations.

        frequency : `str`, optional
            A string representing the frequency of observations. Default is "1d" (1 day).
            Valid frequency strings can be specified, such as "1h" for 1 hour, "15T" for 15 minutes,
            "1M" for 1 month, "1Y" for 1 year, and more. Refer to the pandas documentation for a complete
            list of valid frequency strings: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases

        Returns
        -------
        str
            The JSOC query string for retrieving the specified observations.

        Notes
        -----
        - According to JSOC: [DATE-OBS] DATE_OBS = T_OBS - EXPTIME/2.0
        - Quality flags needs to be addressed and utilised
            - https://github.com/sunpy/drms/issues/37
        - https://github.com/sunpy/drms/issues/98;
            - Fixed in https://github.com/sunpy/drms/pull/102
        """
        # https://github.com/sunpy/drms/issues/98; Fixed in https://github.com/sunpy/drms/pull/102
        return f"{self.series_name}[{datetime_to_jsoc(start_time)}-{datetime_to_jsoc(end_time)}@{frequency}]"  # [? QUALITY=0 ?]"

    def _get_matching_info_from_record(self, records: pd.Series) -> tuple[pd.DataFrame, list[str]]:
        """
        Extract matching information from records in a DataFrame.

        This method processes a DataFrame containing records and extracts relevant information,
        such as dates and other identifiers, using regular expressions. The information extraction
        process involves searching for specific patterns within each record.

        Parameters
        ----------
        records : pd.Series
            A DataFrame column containing records to extract information from.

        Returns
        -------
        pd.DataFrame
            A DataFrame with extracted information.

        Notes
        -----
        Regular expressions (regex) are powerful tools for pattern matching in strings. In this method,
        we use a regex pattern to extract specific information enclosed within square brackets from each record.

        Here's a breakdown of the regex pattern:
        - `\\[` : Matches the opening square bracket character.
        - `(.*?)` : This is a capturing group that matches any characters (.*), but the ? makes it non-greedy
                   so that it captures the shortest sequence possible.
        - `\\]` : Matches the closing square bracket character.

        If the pattern is found in a record, the matched content is extracted and stored in the results.
        If the pattern is not found (or no matches are found), a default value of None is added to the results.

        Please note that the `str.extract()` method captures only the first occurrence of the pattern in each row.
        If there are multiple occurrences within a single record and you need to capture all of them, you might consider
        using other methods like `str.findall()`.

        For those new to regex, here's a simple explanation of how it works:
        - The opening \\[ matches the literal character "[".
        - (.*?) is a capturing group that captures any characters in a non-greedy manner.
        - The closing \\] matches the literal character "]".

        If you'd like to learn more about regex and its syntax, you can refer to the Python `re` module documentation:
        https://docs.python.org/3/library/re.html
        """
        extracted_info = records.str.extract(r"\[(.*?)\]")
        extracted_info.columns = ["T_REC"]

        return extracted_info

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

        Notes
        -----
        This is used for converting DATE-OBS to a datetime. This perhaps isn't
        the ideal way to do this.
        """
        return "%Y-%m-%dT%H:%M:%S.%fZ"

    @property
    def segment_column_name(self) -> str:
        """
        Get the name of the HMI data segment.

        Returns
        -------
        str
            The name of the HMI data segment.
        """
        return "magnetogram"

    @property
    def metadata_save_location(self) -> str:
        """
        Get the HMI directory path for saving metadata.

        Returns
        -------
        str
            The HMI directory path for saving metadata.
        """
        return config["paths"]["hmi_mag_raw_csv"]


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
        return config["paths"]["hmi_ic_raw_csv"]


class HMISHARPs(HMILOSMagnetogram):
    def __init__(self):
        super().__init__()

    def generate_drms_query(self, start_time: datetime.datetime, end_time: datetime.datetime, frequency="1d") -> str:
        """
        Generate a JSOC query string for requesting observations within a specified time range.

        Parameters
        ----------
        start_time : datetime.datetime
            A datetime object representing the start time of the requested observations.

        end_time : datetime.datetime
            A datetime object representing the end time of the requested observations.

        frequency : `str`, optional
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

    def _get_matching_info_from_record(self, records: pd.Series) -> pd.DataFrame:
        """
        Extract matching information from records in a DataFrame.

        This method processes a DataFrame containing records and extracts relevant information,
        such as dates and other identifiers, using regular expressions. The information extraction
        process involves searching for specific patterns within each record.

        Parameters
        ----------
        records : pd.Series
            A DataFrame column containing records to extract information from.

        Returns
        -------
        pd.DataFrame
            A DataFrame with extracted information.

        Notes
        -----
        Regular expressions (regex) are powerful tools for pattern matching in strings. In this method,
        we use a regex pattern to extract specific information enclosed within two sets of square brackets from each record.

        Here's a breakdown of the regex pattern:
        - `\\[` : Matches the opening square bracket character.
        - `(.*?)` : This is a capturing group that matches any characters (.*), but the ? makes it non-greedy
                   so that it captures the shortest sequence possible.
        - `\\]` : Matches the closing square bracket character.
        - `\\[` : Matches the opening square bracket character of the second set.
        - `(.*?)` : This is another capturing group for the second set.
        - `\\]` : Matches the closing square bracket character of the second set.

        If the pattern is found in a record, the matched content is extracted and stored in the results.
        If the pattern is not found (or no matches are found), a default value of None is added to the results.

        Please note that the `str.extract()` method captures only the first occurrence of the pattern in each row.
        If there are multiple occurrences within a single record and you need to capture all of them, consider
        using other methods like `str.findall()`.

        If you'd like to learn more about regex and its syntax, you can refer to the Python `re` module documentation:
        https://docs.python.org/3/library/re.html
        """
        extracted_info = records.str.extract(r"\[(.*?)\]\[(.*?)\]")
        extracted_info.columns = ["HARPNUM", "T_REC"]
        # cast to Int64 as NaN isn't represented in an int column
        extracted_info["HARPNUM"] = extracted_info["HARPNUM"].astype("Int64")

        return extracted_info

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
        return config["paths"]["hmi_sharps_raw_csv"]


class HMIMagnetogramNRT(HMILOSMagnetogram):
    def __init__(self):
        raise NotImplementedError("Placeholder class for NRT HMI Magnetograms")
