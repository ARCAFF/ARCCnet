from time import sleep
from typing import Union, Optional
from datetime import datetime

from sunpy.net import Fido
from sunpy.net import attrs as a
from sunpy.net.helio import HECResponse
from sunpy.time import TimeRange

import astropy.units as u
from astropy.table import QTable, vstack
from astropy.time import Time

from arccnet.catalogs.flares.common import FlareCatalog
from arccnet.utils.logging import get_logger

__all__ = ["HECFlareCatalog"]

logger = get_logger(__name__, level="DEBUG")

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
    HEC flare catalogs
    """

    def __init__(self, catalog: str):
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

        # There is a hard server side limit of 20,000 records, making many small queries will be blocked so use
        # different size windows for different time periods so for 2022 on use 15 day intervals
        years = [tr.end.to_datetime().year for tr in windows]
        if 2022 in years:
            first_2022 = years.index(2022)
            tr_2022_plus = TimeRange(windows[first_2022].start, windows[first_2022 + 1].end)
            new_windows = tr_2022_plus.window(5 * u.day, 5 * u.day)
            windows = windows[: first_2022 - 1] + new_windows + windows[first_2022 + 1 :]

        flares = []
        for i, window in enumerate(windows):
            logger.debug(f"Searching for flares {i}: {window.start} - {window.end}")
            cur_flares = Fido.search(a.Time(window.start, window.end), *self.query)
            if i % 4 == 0 and i > 0:
                sleep(10)
            cur_num_flares = len(cur_flares["hec"])
            logger.debug(f"Found {cur_num_flares} flares {i}: {window.start} - {window.end}")
            if cur_num_flares >= 20000:
                logger.error(f"Hitting hard limit on HEC for in interval {i}, {window.start} - {window.end}")
            flares.append(cur_flares["hec"])

        # Remove meta (can't stack otherwise)
        for flare in flares:
            flare.meta = None
        stacked_flares = vstack(flares)

        # Convert string time columns to masked Time columns
        for col in ["time_start", "time_end", "time_peak"]:
            stacked_flares.replace_column(col, Time(stacked_flares[col].tolist()))

        # Convert to a fixed length string array so can serialise
        stacked_flares.replace_column("xray_class", stacked_flares["xray_class"].tolist())
        if self.catalog == "hec_gevloc":
            stacked_flares.replace_column("ename", stacked_flares["ename"].tolist())
            stacked_flares.replace_column("url_nar", stacked_flares["url_nar"].tolist())
            stacked_flares.replace_column("url_flare", stacked_flares["url_flare"].tolist())
        else:
            stacked_flares.replace_column("optical_class", stacked_flares["optical_class"].tolist())

        return QTable(stacked_flares.as_array())

    def create_catalog(self, query: HECResponse) -> FlareCatalog:
        r"""
        Create a catalog from the given queried data.

        Essentially a map column name and types into common format

        Parameters
        ----------
        query

        Returns
        -------

        """
        query.meta = None
        # Could be an empty table
        if len(query.columns) > 0:
            query.rename_columns(list(COLUM_MAPPINGS.keys()), list(COLUM_MAPPINGS.values()))
        query["source"] = self.catalog
        return FlareCatalog(query)
