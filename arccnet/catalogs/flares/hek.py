from typing import Union, Optional
from datetime import datetime

from sunpy.net import Fido
from sunpy.net import attrs as a
from sunpy.net.hek import HEKClient, HEKTable
from sunpy.time import TimeRange

from astropy.table import Table, vstack
from astropy.time import Time

from arccnet.catalogs.flares.common import FlareCatalog
from arccnet.data_generation.utils.data_logger import get_logger

__all__ = ["HEKFlareCatalog"]


logger = get_logger(__name__)

# Supported catalogs
CATALOGS = {
    "ssw_latest": (a.hek.FL, a.hek.FRM.Name == "SSW Latest Events"),
    "swpc": (a.hek.FL, a.hek.FRM.Name == "SWPC"),
}


# Mapping columns from HEK to flare catalog
COLUM_MAPPINGS = {
    "event_starttime": "start_time",
    "event_endtime": "end_time",
    "event_peaktime": "peak_time",
    "fl_goescls": "goes_class",
    "hgs_x": "hgs_longitude",
    "hgs_y": "hgs_latitude",
    "ar_noaanum": "noaa_number",
}


class HEKFlareCatalog:
    r"""
    SSW Latest Events Catalog provided by HEK
    """

    def __init__(self, catalog: str):
        if catalog not in CATALOGS.keys():
            raise ValueError(f"Unknown catalog: {catalog}")
        self.catalog = f"hek_{catalog}"
        self.query = CATALOGS[catalog]

    def search(
        self,
        start_time: Union[Time, datetime, str],
        end_time: Union[Time, datetime, str],
        n_splits: Optional[int] = None,
    ):
        r"""
        Search the HEK SSW Latest Events catalog for flares between the start and end times.

        Note
        ----
        Seems to be a hard limit on HEC server side of ~20,000 results so split into ~year chunks.

        Parameters
        ----------
        n_splits
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
            flares.append(cur_flares["hek"])

        # Remove meta (can't stack otherwise)
        for flare in flares:
            flare.meta = None
        stacked_flares = vstack(flares)

        return HEKTable(stacked_flares, client=HEKClient())

    def create_catalog(self, query: HEKTable) -> FlareCatalog:
        r"""
        Create a FlareCatalog from the give HEK query.

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

    def clean_catalog(self, catalog: Table) -> FlareCatalog:
        pass
