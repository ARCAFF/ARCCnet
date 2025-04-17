from datetime import datetime

import numpy as np
from sunpy.net import Fido
from sunpy.net import attrs as a
from sunpy.net.hek import HEKTable
from sunpy.time import TimeRange

from astropy.table import Table, vstack
from astropy.time import Time

from arccnet.catalogs.flares.common import FlareCatalog
from arccnet.utils.logging import get_logger

__all__ = ["HEKFlareCatalog"]


logger = get_logger(__name__, level="DEBUG")

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
    HEK Flare Catalog
    """

    def __init__(self, catalog: str):
        if catalog not in CATALOGS.keys():
            raise ValueError(f"Unknown catalog: {catalog}")
        self.catalog = f"hek_{catalog}"
        self.query = CATALOGS[catalog]

    def search(
        self,
        start_time: Time | datetime | str,
        end_time: Time | datetime | str,
        n_splits: int | None = None,
    ):
        r"""
        Search the HEK SSW Latest Events catalog for flares between the start and end times.

        Parameters
        ----------
        start_time :
            Start time
        end_time
            End time
        n_splits : int, optional
            Number of windows to split the time range over, by default spits into ~6-months windows.

        Returns
        -------

        """
        time_range = TimeRange(start_time, end_time)
        if n_splits is None:
            n_splits = int((time_range.end - time_range.start).to_value("year"))
            if n_splits == 0:
                windows = [time_range]
            else:
                n_splits = n_splits * 2
                windows = time_range.split(max(2, n_splits))  # slit into ~6-month intervals to keep queries reasonable
            logger.debug(f"Splitting query from {time_range.start} to {time_range.end} into {n_splits} windows.")

        flares = []
        for i, window in enumerate(windows):
            logger.debug(f"Searching for flares {i}: {window.start} - {window.end}")
            cur_flares = Fido.search(a.Time(window.start, window.end), *self.query)
            logger.debug(f"Found {len(cur_flares['hek'])} flares {i}: {window.start} - {window.end}")
            flares.append(cur_flares["hek"])

        # Remove meta (can't stack otherwise)
        for flare in flares:
            flare.meta = None
            # 'refs' can be serialised, sometimes it has different shapes
            # 'event_probability' is sometimes None so drop - could replace with -1 if need in future
            # 'event_avg_rating', 'event_importance' are sometimes None or 1
            for col_to_remove in ["refs", "event_probability", "event_avg_rating", "event_importance"]:
                try:
                    flare.remove_column(col_to_remove)
                except KeyError:
                    logger.debug(f"No {col_to_remove} column in: {flare.columns}, len: {len(flare)}")

        stacked_flares = vstack([f for f in flares if len(f) > 0])  # In case some time windows are empty

        # Remove columns which are all none
        col_to_remove = []
        for col in stacked_flares.columns:
            if np.all(stacked_flares[col] == None):  # noqa
                col_to_remove.append(col)
        if len(col_to_remove) > 0:
            logger.debug(f"Dropping columns {col_to_remove} as are all None")
            stacked_flares.remove_columns(col_to_remove)

        return Table(stacked_flares.as_array())

    def create_catalog(self, query: HEKTable) -> FlareCatalog:
        r"""
        Create a FlareCatalog from the give Fido query.

        Essentially map column names and types into common format `arccnet.catalogs.flares.common.FlareCatalog`

        Parameters
        ----------
        query :
            The search query
        Returns
        -------

        """
        query.meta = None
        # Could be an empty table
        if len(query.columns) > 0:
            query.rename_columns(list(COLUM_MAPPINGS.keys()), list(COLUM_MAPPINGS.values()))
        query["source"] = self.catalog
        return FlareCatalog(query)

    def clean_catalog(self, catalog: Table) -> FlareCatalog:
        r"""
        Clean the given catalog.

        Parameters
        ----------
        catalog

        Returns
        -------

        """
        pass
