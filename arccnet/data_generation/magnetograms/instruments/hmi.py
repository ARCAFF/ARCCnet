import drms
import pandas as pd

import arccnet.data_generation.utils.default_variables as dv
from arccnet.data_generation.magnetograms.base_magnetogram import BaseMagnetogram
from arccnet.data_generation.magnetograms.utils import datetime_to_jsoc
from arccnet.data_generation.utils.data_logger import logger

__all__ = ["HMIMagnetogram"]


class HMIMagnetogram(BaseMagnetogram):
    def __init__(self):
        super().__init__()

    def fetch_metadata(self, start_date, end_date) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get the fulldisk HMI data.

        Returns
        -------
        Tuple[dict, dict]
            A tuple containing a dictionary of keys (metadata for the HMI image)
            and a dictionary of segments
        """

        magnetogram_string = (
            f"hmi.M_720s[{datetime_to_jsoc(start_date)}-{datetime_to_jsoc(end_date)}@1d]"  # [? QUALITY=0 ?]"
        )
        # https://github.com/sunpy/drms/issues/98
        # https://github.com/sunpy/drms/issues/37
        # want to deal with quality after obtaining the data

        logger.info(f">> HMI Query: {magnetogram_string}")
        keys, seg = self._c.query(magnetogram_string, key=drms.const.all, seg="magnetogram")
        logger.info(f">> ... {len(keys)}, {len(seg)}")

        # Obtain the segments and set into the keys
        magnetogram_fits = dv.JSOC_BASE_URL + seg.magnetogram
        keys["magnetogram_fits"] = magnetogram_fits

        # as we combine the magnetogram_fits and keys DataFrame, assure they're the same length
        assert len(magnetogram_fits) == len(keys)

        # raise error if there are no keys returned
        if len(keys) == 0:
            raise ValueError("returns no results!")

        # Making a sunpy.map with fits
        r = self._c.export(magnetogram_string + "{magnetogram}", method="url", protocol="fits")
        # keys["magnetogram_query_string"] = magnetogram_string

        logger.info(len(r.urls), len(keys))  # the naming is different to other data..

        #!TODO move to separate method & use default variables
        # keys["datetime"] = [datetime.datetime.strptime(date, "%Y-%m-%dT%H:%M:%S.%fZ") for date in keys["DATE-OBS"]]
        keys["datetime"] = [
            pd.to_datetime(date, format="%Y-%m-%dT%H:%M:%S.%fZ", errors="coerce") for date in keys["DATE-OBS"]
        ]  # According to JSOC: [DATE-OBS] DATE_OBS = T_OBS - EXPTIME/2.0

        # keys is the keys, with links to the magnetogram
        # r.urls are urls of pure fits files.
        return keys  # , r.urls
