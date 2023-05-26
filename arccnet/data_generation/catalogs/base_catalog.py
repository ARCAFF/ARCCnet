from abc import ABC, abstractmethod

__all__ = ["BaseCatalog"]


class BaseCatalog(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def fetch_data(self):
        """
        Fetch data for a given time range.
        This method must be implemented by concrete subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def create_catalog(self):
        """
        create data catalog from fetched data.
        This method must be implemented by concrete subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def clean_catalog(self):
        """
        Clean the data catalog.
        This method must be implemented by concrete subclasses.
        """
        raise NotImplementedError

    def run_deepchecks(self):
        """
        Run deepchecks on the catalog data.
        """
        raise NotImplementedError
