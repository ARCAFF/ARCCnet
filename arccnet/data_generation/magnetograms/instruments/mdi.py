import datetime

import pandas as pd

from arccnet.data_generation.magnetograms.base_magnetogram import BaseMagnetogram
from arccnet.data_generation.magnetograms.utils import datetime_to_jsoc

__all__ = ["MDILOSMagnetogram", "MDISMARPs"]


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
        # Line-of-sight magnetic field from 30-second observations in full-disk mode,
        # sampled either once in a minute or averaged over five consecutive minute samples.
        # Whether the data are form a single observation or an average of five is given by
        # the value of the keyword INTERVAL, the length of the sampling interval in seconds.
        # The data are acquired as part of the regular observing program.
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
        Regular expressions (regex) are powerful tools for working with patterns in strings. In this method,
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

        Notes
        -----
        This is used for converting DATE-OBS to a datetime. This perhaps isn't
        the ideal way to do this.
        """
        return "%Y-%m-%dT%H:%M:%SZ"

    @property
    def segment_column_name(self) -> str:
        """
        Returns
        -------
        str:
            Name of the MDI data segment
        """
        return "data"


class MDISMARPs(MDILOSMagnetogram):
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
        such as dates and identifiers, using regular expressions. The information extraction
        process involves searching for specific patterns within each record.

        Parameters
        ----------
        records : pd.DataFrame
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
        extracted_info.columns = ["TARPNUM", "T_REC"]
        extracted_info["TARPNUM"] = pd.to_numeric(extracted_info["TARPNUM"])  # !TODO fix this hack

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
        return "mdi.smarp_96m"

    @property
    def date_format(self) -> str:
        """
        Get the MDI date string format.

        Returns
        -------
        str
            The MDI date string format.

        Notes
        -----
        In SMARPs, the format is the same as HMI, not the same as MDI.

        DATE-OBS format for reference:
        - MDI: '%Y-%m-%dT%H:%M:%SZ'
        - HMI: '%Y-%m-%dT%H:%M:%S.%fZ'
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
        return "bitmap"


class MDIContinuum(MDILOSMagnetogram):
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
        return "mdi.fd_ic_interp"

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
