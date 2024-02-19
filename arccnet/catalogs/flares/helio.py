from typing import Union
from pathlib import Path
from datetime import datetime

from sunpy.net import Fido
from sunpy.net import attrs as a
from sunpy.net.helio import HECResponse

from astropy.table import Table, vstack
from astropy.time import Time

from arccnet.catalogs.flares.common import _generate_intervals
from arccnet.data_generation.utils.data_logger import get_logger

logger = get_logger(__name__)


class HECGEVLOCFlareCatalog:
    r"""
    GEVLOC flare catalog provided by HEC
    """

    catalog_name = "gevloc"
    source = "HEC"

    def search(self, start_time: Union[Time, datetime, str], end_time: Union[Time, datetime, str], **kwargs):
        r"""
        Search the HEC GEVLOC catalog for flares between the start and end times.

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
            return Fido.search(
                a.Time(start_time, end_time), a.helio.TableName("gevloc_sxr_flare"), a.helio.MaxRecords(9999)
            )

        starts, ends = _generate_intervals(start_time, end_time, int(duration_years))

        flares = []
        for start, end in zip(starts, ends):
            cur_flares = Fido.search(
                a.Time(start, end), a.helio.TableName("gevloc_sxr_flare"), a.helio.MaxRecords(9999)
            )
            flares.append(cur_flares)

        # Remove meta (can't stack otherwise)
        flares[0][0].meta
        for flare in flares:
            flare[0].meta = None

        stacked_flares = vstack([f[0] for f in flares])
        return HECResponse(stacked_flares)

    def fetch(self, result) -> list[Path]:
        return NotImplemented("No fetch stage for flare catalogs")

    def create_catalog(self, files: list[Path]) -> Table:
        pass

    def clean_catalog(self, catalog: Table) -> Table:
        pass


class HECGOESFlareCatalog:
    r"""
    GOES flare catalog provided by HEC
    """
    catalog_name = "goes"
    source = "HEC"

    def search(self, start_time: Union[Time, datetime, str], end_time: Union[Time, datetime, str], **kwargs):
        r"""
        Search the HEC GEVLOC catalog for flares between the start and end times.

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
            return Fido.search(
                a.Time(start_time, end_time), a.helio.TableName("goes_sxr_flare"), a.helio.MaxRecords(9999)
            )

        starts, ends = _generate_intervals(start_time, end_time, int(duration_years))

        flares = []
        for start, end in zip(starts, ends):
            cur_flares = Fido.search(a.Time(start, end), a.helio.TableName("goes_sxr_flare"), a.helio.MaxRecords(9999))
            flares.append(cur_flares)

        # Remove meta (can't stack otherwise)
        flares[0][0].meta
        for flare in flares:
            flare[0].meta = None

        stacked_flares = vstack([f[0] for f in flares])
        return HECResponse(stacked_flares)
