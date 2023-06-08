import drms
import pandas as pd

import arccnet.data_generation.utils.default_variables as dv
from arccnet.data_generation.magnetograms.base_magnetogram import BaseMagnetogram
from arccnet.data_generation.magnetograms.utils import datetime_to_jsoc
from arccnet.data_generation.utils.data_logger import logger

__all__ = ["MDIMagnetogram"]


class MDIMagnetogram(BaseMagnetogram):
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
            f"mdi.fd_M_96m_lev182[{datetime_to_jsoc(start_date)}-{datetime_to_jsoc(end_date)}@1d]"  # [? QUALITY=0 ?]"
        )
        # Line-of-sight magnetic field from 30-second observations in full-disc mode,
        # sampled either once in a minute or averaged over five consecutive minute samples.
        # Whether the data are form a single observation or an average of five is given by
        # the value of the keyword INTERVAL, the length of the sampling interval in seconds.
        # The data are acquired as part of the regular observing program.

        logger.info(f">> MDI Query: {magnetogram_string}")
        keys, seg = self._c.query(magnetogram_string, key=drms.const.all, seg="data")

        # raise error if there are no keys returned
        if len(keys) == 0:
            raise ValueError("returns no results!")

        # Obtain the segments and set into the keys
        magnetogram_fits = dv.JSOC_BASE_URL + seg.data
        keys["magnetogram_fits"] = magnetogram_fits

        keys["datetime"] = [
            pd.to_datetime(date, format=dv.MDI_DATE_FORMAT, errors="coerce") for date in keys["DATE-OBS"]
        ]
        # as we combine the magnetogram_fits and keys DataFrame, assure they're the same length
        # assert len(magnetogram_fits) == len(keys)

        # Making a sunpy.map with fits
        self._c.export(magnetogram_string + "{data}", method="url", protocol="fits")
        # keys["magnetogram_query_string"] = magnetogram_string

        # assert len(r.urls) == len(keys)  # the naming is different to other data..

        # keys is the keys, with links to the magnetogram
        # r.urls are urls of pure fits files.
        return keys  # , r.urls
