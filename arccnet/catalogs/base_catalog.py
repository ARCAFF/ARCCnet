from abc import ABC, abstractmethod

__all__ = ["BaseCatalog"]

from pathlib import Path

from astropy.table import Table  # type: ignore


class BaseCatalog(ABC):
    def __init__(self):
        self._table = Table()

    @abstractmethod
    def search(self, start, end, **kwargs) -> Table:
        """
        Search for data for a given time range.
        This method must be implemented by concrete subclasses.
        """
        pass

    @abstractmethod
    def fetch(self, result) -> list[Path]:
        """
        Fetch data for a given time range.
        This method must be implemented by concrete subclasses.
        """
        pass

    @abstractmethod
    def create_catalog(self, files: list[Path]) -> Table:
        """
        create data catalog from fetched data.
        This method must be implemented by concrete subclasses.
        """
        pass

    @abstractmethod
    def clean_catalog(self, catalog: Table) -> Table:
        """
        Clean the data catalog.
        This method must be implemented by concrete subclasses.
        """
        pass

    def run_deepchecks(self):
        """
        Run deepchecks on the catalog data.
        """
        pass
