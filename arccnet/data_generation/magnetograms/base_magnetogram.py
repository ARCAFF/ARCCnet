from abc import ABC, abstractmethod

import drms
import pandas as pd

import arccnet.data_generation.utils.default_variables as dv
from arccnet.data_generation.utils.data_logger import logger

__all__ = ["BaseMagnetogram"]


class BaseMagnetogram(ABC):
    def __init__(self) -> None:
        super().__init__()
        self._c = drms.Client(debug=False, verbose=False, email=dv.JSOC_DEFAULT_EMAIL)

    @abstractmethod
    def query(self, start_time, end_time, frequency) -> str:
        """
        Returns
        -------
        str:
            JSOC Query string
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
            instantiate class name (e.g. child class if inherited)
        """
        return self.__class__.__name__

    def fetch_metadata(self, start_date, end_date) -> pd.DataFrame:  # tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch metadata from JSOC.

        Returns
        -------
        keys: pd.DataFrame
            A `pd.DataFrame` containing all keys (`drms.const.all`) for a JSOC query string and urls corresponding to the request segments (`seg`).
            The required segment is defined in the child class (for a magnetogram, `seg="magnetogram"` for HMI, and `seg="data"` for MDI).

            The `pd.DataFrame` also contains `urls` to the complete `.fits` files (magnetogram + metadata) that are staged by JSOC for download.

        """

        q = self.query(start_date, end_date)
        logger.info(f">> {self._type()} Query: {q}")
        keys, seg = self._c.query(q, key=drms.const.all, seg=self.segment_column_name)
        assert len(keys) == len(seg)
        logger.info(f"\t {len(keys)} entries")

        # Obtain the segments and set into the keys
        magnetogram_fits = dv.JSOC_BASE_URL + seg[self.segment_column_name]
        keys["magnetogram_fits"] = magnetogram_fits

        # as we combine the magnetogram_fits and keys DataFrame, assure they're the same length
        # assert len(magnetogram_fits) == len(keys)

        # raise error if there are no keys returned
        if len(keys) == 0:
            # !TODO implement custom error message
            raise (f"No results return for the query: {q}!")

        # Export the  .fits (data + metadata) for the same query
        r = self._c.export(q + "{" + self.segment_column_name + "}", method="url", protocol="fits")

        # extract the `record` and strip the square brackets to return a T_REC-like time (in TAI)
        self.r_urls = r.urls.copy()
        self.r_urls["extracted_record_timestamp"] = self.r_urls["record"].str.extract(r"\[(.*?)\]")
        # merge on keys['T_REC'] so that there we can later get the files.
        # !TODO add testing for this merge
        keys = pd.merge(
            left=keys, right=self.r_urls, left_on="T_REC", right_on="extracted_record_timestamp", how="left"
        )

        # keys["datetime"] = [datetime.datetime.strptime(date, "%Y-%m-%dT%H:%M:%S.%fZ") for date in keys["DATE-OBS"]]
        keys["datetime"] = [
            pd.to_datetime(date, format=self.date_format, errors="coerce") for date in keys["DATE-OBS"]
        ]  # According to JSOC: [DATE-OBS] DATE_OBS = T_OBS - EXPTIME/2.0

        keys.to_csv(self.metadata_save_location)

        return keys
