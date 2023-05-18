from abc import ABC, abstractmethod

__all__ = ["BaseCatalog"]


class BaseCatalog(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def fetch_data(self):
        """
        Fetch data from the catalog for a given time range.
        This method must be implemented by concrete subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def clean_data(self):
        """
        Clean the data in the catalog.
        This method must be implemented by concrete subclasses.
        """
        raise NotImplementedError

    def run_deepchecks(self):
        """
        Run deepchecks on the catalog data.
        """
        raise NotImplementedError
