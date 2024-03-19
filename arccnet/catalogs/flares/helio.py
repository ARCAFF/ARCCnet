from typing import Union, Optional
from datetime import datetime

from sunpy.net import Fido
from sunpy.net import attrs as a
from sunpy.net.helio import HECResponse
from sunpy.time import TimeRange

from astropy.table import vstack
from astropy.time import Time

from arccnet.catalogs.flares.common import FlareCatalog
from arccnet.data_generation.utils.data_logger import get_logger

__all__ = ["HECFlareCatalog"]

logger = get_logger(__name__)

# Supported catalogs
CATALOGS = {
    "gevloc": (a.helio.TableName("gevloc_sxr_flare"), a.helio.MaxRecords(99999)),
    "goes": (a.helio.TableName("goes_sxr_flare"), a.helio.MaxRecords(99999)),
}


# Mapping columns from HEK to flare catalog
COLUM_MAPPINGS = {
    "time_start": "start_time",
    "time_end": "end_time",
    "time_peak": "peak_time",
    "xray_class": "goes_class",
    "long_hg": "hgs_longitude",
    "lat_hg": "hgs_latitude",
    "nar": "noaa_number",
}


class HECFlareCatalog:
    r"""
    GEVLOC flare catalog provided by HEC
    """

    def __init__(self, catalog: str, **kwargs):
        if catalog not in CATALOGS.keys():
            raise ValueError(f"Unknown catalog {catalog}")
        self.catalog = f"hec_{catalog}"
        self.query = CATALOGS[catalog]

    def search(
        self,
        start_time: Union[Time, datetime, str],
        end_time: Union[Time, datetime, str],
        n_splits: Optional[int] = None,
    ):
        r"""
        Search the HEC GEVLOC catalog for flares between the start and end times.

        Note
        ----
        Seems to be a hard limit on HEC server side of ~20,000 results so split into ~6-month chunks.

        Parameters
        ----------
        start_time :
            Start time
        end_time
            End time
        n_splits : optional int
            Number of windows to split the time range over, by default ~6-months windows.

        Returns
        -------

        """
        time_range = TimeRange(start_time, end_time)
        if n_splits is None:
            n_splits = int((time_range.end - time_range.start).to_value("year"))
            if n_splits == 0:
                windows = [time_range]
            else:
                windows = time_range.split(
                    min(2, n_splits * 2)
                )  # slit into ~6-month intervals to keep queries reasonable

        flares = []
        for window in windows:
            cur_flares = Fido.search(a.Time(window.start, window.end), *self.query)
            if len(cur_flares["hec"]) >= 20000:
                raise ValueError("Hitting hard limit on HEC")
            flares.append(cur_flares["hec"])

        # Remove meta (can't stack otherwise)
        for flare in flares:
            flare.meta = None
        stacked_flares = vstack(flares)

        return HECResponse(stacked_flares)

    def create_catalog(self, query: HECResponse) -> FlareCatalog:
        r"""
        Create a FlareCatalog from the give HEC query.

        Essentially a map column name and types into common format

        Parameters
        ----------
        query

        Returns
        -------

        """
        query.meta = None
        query.rename_columns(list(COLUM_MAPPINGS.keys()), list(COLUM_MAPPINGS.values()))
        query["source"] = self.catalog
        return FlareCatalog(query)
