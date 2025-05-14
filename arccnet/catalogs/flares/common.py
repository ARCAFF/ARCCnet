from __future__ import annotations

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
                f"{self.__class__.__name__} must contain {list(self.required_column_types.keys())} columns"
            )

    @classmethod
    def read(cls, *args, **kwargs) -> FlareCatalog:
        r"""
        Read the catalog from a file.
        """
        table = QTable.read(*args, **kwargs)
        return cls(table)

    def write(self, *args, **kwargs) -> None:
        r"""
        Write the catalog to a file.
        """
        return super(QTable, self).write(*args, **kwargs)
