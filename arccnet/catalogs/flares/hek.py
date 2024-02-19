from typing import Union
from datetime import datetime

from sunpy.net import Fido
from sunpy.net import attrs as a
from sunpy.net.hek import HEKClient, HEKTable
from sunpy.net.helio import HECResponse

from astropy.table import Table, vstack
from astropy.time import Time

from arccnet.catalogs.flares.common import FlareCatalog, _generate_intervals
from arccnet.data_generation.utils.data_logger import get_logger

__all__ = ["HEKSWPCCatalog", "HEKSSWFlareCatalog"]


logger = get_logger(__name__)

# Mapping columbs from HEK to flare catalog
COLUM_MAPPINGS = {
    "event_starttime": "start_time",
    "event_endtime": "end_time",
    "event_peaktime": "peak_time",
    "fl_goescls": "goes_class",
    "hgs_x": "hgs_longitude",
    "hgs_y": "hgs_latitude",
    "ar_noaanum": "noaa_number",
}


class HEKSWPCCatalog:
    r"""
    HEC GEVLOC catalog
    """

    def search(self, start_time: Union[Time, datetime, str], end_time: Union[Time, datetime, str], **kwargs):
        r"""
        Search the HEK SWPC catalog for flares between the start and end times.


        Parameters
        ----------
        start_time :
            Start time
        end_time
            End time
        kwargs

        Returns
        -------

        """
        start_time = Time(start_time)
        end_time = Time(end_time)
        duration = end_time - start_time
        duration_years = duration.to_value("year")

        if duration_years < 1:
            res = Fido.search(a.Time(start_time, end_time), a.hek.FL, a.hek.FRM.Name == "SWPC")
            return res["hek"]

        starts, ends = _generate_intervals(start_time, end_time, int(duration_years))
        flares = []
        for start, end in zip(starts, ends):
            cur_flares = Fido.search(a.Time(start, end), a.hek.FL, a.hek.FRM.Name == "SWPC")
            flares.append(cur_flares)

        # Remove meta (can't stack otherwise)
        for flare in flares:
            flare[0].meta = None
        stacked_flares = vstack([f[0] for f in flares])

        return HECResponse(stacked_flares, client=HEKClient())

    def create_catalog(self, query: HECResponse) -> FlareCatalog:
        pass

    def clean_catalog(self, catalog: Table) -> FlareCatalog:
        pass


class HEKSSWFlareCatalog:
    r"""
    SSW Latest Events Catalog provided by HEK
    """

    def search(self, start_time: Union[Time, datetime, str], end_time: Union[Time, datetime, str], **kwargs):
        r"""
        Search the HEK SSW Latest Events catalog for flares between the start and end times.

        Note
        ----
        Seems to be a hard limit on HEC server side of ~20,000 results so split into ~year chunks.

        Parameters
        ----------
        start_time :
            Start time
        end_time
            End time
        kwargs

        Returns
        -------

        """
        start_time = Time(start_time)
        end_time = Time(end_time)
        duration = end_time - start_time
        duration_years = duration.to_value("year")

        if duration_years < 1:
            res = Fido.search(a.Time(start_time, end_time), a.hek.FL, a.hek.FRM.Name == "SSW Latest Events")
            return res["hek"]

        starts, ends = _generate_intervals(start_time, end_time, int(duration_years))
        flares = []
        for start, end in zip(starts, ends):
            cur_flares = Fido.search(a.Time(start, end), a.hek.FRM.Name == "SSW Latest Events")
            flares.append(cur_flares)

        # Remove meta (can't stack otherwise)
        for flare in flares:
            flare[0].meta = None
        stacked_flares = vstack([f[0] for f in flares])

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
        query["source"] = "hek_ssw_latest"
        return FlareCatalog(query)

    def clean_catalog(self, catalog: Table) -> FlareCatalog:
        pass
