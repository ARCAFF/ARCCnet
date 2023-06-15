import datetime

import arccnet.data_generation.utils.default_variables as dv
from arccnet.data_generation.magnetograms.base_magnetogram import BaseMagnetogram
from arccnet.data_generation.magnetograms.utils import datetime_to_jsoc

__all__ = ["HMIMagnetogram"]


class HMIMagnetogram(BaseMagnetogram):
    def __init__(self):
        super().__init__()

    def query(self, start_time: datetime.datetime, end_time: datetime.datetime, frequency="1d") -> str:
        # https://github.com/sunpy/drms/issues/98
        # https://github.com/sunpy/drms/issues/37
        # want to deal with quality after obtaining the data
        return (
            f"hmi.M_720s[{datetime_to_jsoc(start_time)}-{datetime_to_jsoc(end_time)}@{frequency}]"  # [? QUALITY=0 ?]"
        )

    @property
    def date_format(self) -> str:
        return dv.HMI_DATE_FORMAT
        # According to JSOC: [DATE-OBS] DATE_OBS = T_OBS - EXPTIME/2.0

    @property
    def segment_column_name(self) -> str:
        return dv.HMI_SEG_COL

    @property
    def metadata_save_location(self) -> str:
        return dv.HMI_MAG_DIR
