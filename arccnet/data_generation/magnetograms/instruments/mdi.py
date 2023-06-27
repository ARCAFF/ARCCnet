import datetime

import arccnet.data_generation.utils.default_variables as dv
from arccnet.data_generation.magnetograms.base_magnetogram import BaseMagnetogram
from arccnet.data_generation.magnetograms.utils import datetime_to_jsoc

__all__ = ["MDIMagnetogram"]


class MDIMagnetogram(BaseMagnetogram):
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
        return dv.MDI_MAG_DIR
