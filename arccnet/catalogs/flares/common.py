from __future__ import annotations

from pathlib import Path

import astropy.units as u
from astropy.table import QTable
from astropy.time import Time


class FlareCatalog(QTable):
    r"""
    Active region classification catalog.
    """
    required_column_types = {
        "start_time": Time,
        "peak_time": Time,
        "end_time": Time,
        "goes_class": str,
        "hgs_longitude": u.deg,
        "hgs_latitude": u.deg,
        "source": str,
        "noaa_number": int,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not set(self.colnames).issuperset(set(self.required_column_types.keys())):
            raise ValueError(
                f"{self.__class__.__name__} must contain " f"{list(self.required_column_types.keys())} columns"
            )

    @classmethod
    def read(cls, *args, **kwargs) -> FlareCatalog:
        r"""
        Read the catalog from a file.
        """
        table = QTable.read(*args, **kwargs)
        paths = [Path(p) for p in table["path"]]
        table.replace_column("path", paths)
        return cls(table)

    def write(self, *args, **kwargs) -> None:
        r"""
        Write the catalog to a file.
        """
        paths = [str(p) for p in self["path"]]
        self["path"] = paths
        return super(QTable, self).write(*args, **kwargs)


def _generate_intervals(start: Time, end: Time, num_intervals: int):
    r"""
    Split a time range in roughly n intervals.

    Parameters
    ----------
    start : `Time`
        Start time
    end : `Time`
        End time
    num_intervals : `int`
        Number of intervals

    Returns
    -------

    """
    timedelta = (end - start) / num_intervals
    starts = []
    ends = []
    for i in range(num_intervals):
        cur_start = start + (i * timedelta)
        cur_end = start + ((i + 1) * timedelta)
        starts.append(cur_start)
        ends.append(cur_end)

    return starts, ends
