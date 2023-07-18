import datetime
from abc import ABC, abstractmethod
from pathlib import Path

import drms
import pandas as pd

import arccnet.data_generation.utils.default_variables as dv
from arccnet.data_generation.utils.data_logger import logger

__all__ = ["BaseMagnetogram"]


class BaseMagnetogram(ABC):
    def __init__(self) -> None:
        super().__init__()
        self._drms_client = drms.Client(debug=False, verbose=False, email=dv.JSOC_DEFAULT_EMAIL)

    @abstractmethod
    def generate_drms_query(self, start_time, end_time, frequency) -> str:
        """
        Returns
        -------
        str:
            JSOC Query string
        """
        raise NotImplementedError("This is the required method in the child class.")

    @property
    @abstractmethod
    def series_name(self) -> str:
        """
        Returns
        -------
        str:
            JSOC series name
        """
        raise NotImplementedError("This is the required method in the child class.")

    @property
    @abstractmethod
    def date_format(self) -> str:
        """
        Returns
        -------
        str:
            instrument date string format
        """
        raise NotImplementedError("This is the required method in the child class.")

    @property
    @abstractmethod
    def segment_column_name(self) -> str:
        """
        Returns
        -------
        str:
            Name of the data segment
        """
        raise NotImplementedError("This is the required method in the child class.")

    @property
    @abstractmethod
    def metadata_save_location(self) -> str:
        """
        Returns
        -------
        str:
            instrument directory path
        """
        raise NotImplementedError("This is the required method in the child class.")

    def _type(self):
        """
        Returns
        -------
        str:
            instantiated class name (e.g. child class if inherited)
        """
        return self.__class__.__name__

    def fetch_metadata(self, start_date: datetime.datetime, end_date: datetime.datetime) -> pd.DataFrame:
        """
        Fetch metadata from JSOC.

        Returns
        -------
        keys: pd.DataFrame
            A `pd.DataFrame` containing all keys (`drms.const.all`) for a JSOC query string and urls corresponding to the request segments (`seg`).
            The required segment is defined in the child class (for a magnetogram, `seg="magnetogram"` for HMI, and `seg="data"` for MDI).

            The `pd.DataFrame` also contains `urls` to the complete `.fits` files (magnetogram + metadata) that are staged by JSOC for download.

        """

        q = self.generate_drms_query(start_date, end_date)
        logger.info(f">> {self._type()} Query: {q}")
        keys, segs = self._drms_client.query(q, key=drms.const.all, seg=self.segment_column_name)
        logger.info(f"\t {len(keys)} entries")

        # Obtain the segments and set into the keys
        magnetogram_fits = dv.JSOC_BASE_URL + segs[self.segment_column_name]
        keys["magnetogram_fits"] = magnetogram_fits

        # raise error if there are no keys returned
        if len(keys) == 0:
            # !TODO implement custom error message
            raise (f"No results return for the query: {q}!")

        # Export the  .fits (data + metadata) for the same query
        export_response = self._drms_client.export(
            q + "{" + self.segment_column_name + "}", method="url", protocol="fits"
        )
        export_response.wait()

        # extract the `record` and strip the square brackets to return a T_REC-like time (in TAI)
        self.r_urls = export_response.urls.copy()
        self.r_urls["extracted_record_timestamp"] = self.r_urls["record"].str.extract(r"\[(.*?)\]")
        # merge on keys['T_REC'] so that there we can later get the files.
        # !TODO add testing for this merge
        keys = pd.merge(
            left=keys, right=self.r_urls, left_on="T_REC", right_on="extracted_record_timestamp", how="left"
        )

        # keys["datetime"] = [datetime.datetime.strptime(date, "%Y-%m-%dT%H:%M:%S.%fZ") for date in keys["DATE-OBS"]]
        keys["datetime"] = [
            pd.to_datetime(date, format=self.date_format, errors="coerce")
            for date in keys["DATE-OBS"]  # ensure we want errors="coerce"
        ]  # According to JSOC: [DATE-OBS] DATE_OBS = T_OBS - EXPTIME/2.0

        directory_path = Path(self.metadata_save_location)
        if not directory_path.exists():
            directory_path.mkdir(parents=True)

        keys.to_csv(directory_path / "raw.csv")

        return keys
