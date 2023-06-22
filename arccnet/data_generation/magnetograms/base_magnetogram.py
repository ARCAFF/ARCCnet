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
        raise NotImplementedError("This is the required method in the child class.")

    @property
    @abstractmethod
    def date_format(self) -> str:
        raise NotImplementedError("This is the required method in the child class.")

    @property
    @abstractmethod
    def segment_column_name(self) -> str:
        raise NotImplementedError("This is the required method in the child class.")

    @property
    @abstractmethod
    def metadata_save_location(self) -> str:
        raise NotImplementedError("This is the required method in the child class.")

    def _type(self):
        return self.__class__.__name__

    def fetch_metadata(self, start_date, end_date) -> pd.DataFrame:  # tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch metadata from JSOC.

        Returns
        -------
            ...
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
        assert len(magnetogram_fits) == len(keys)

        # raise error if there are no keys returned
        if len(keys) == 0:
            raise ValueError("returns no results!")

        # Making a sunpy.map with fits
        r = self._c.export(q + "{" + self.segment_column_name + "}", method="url", protocol="fits")
        # trying to get the `record` to something similar to
        self.r_urls = r.urls.copy()
        self.r_urls["extracted_record_timestamp"] = self.r_urls["record"].str.extract(r"\[(.*?)\]")
        # extract record name (think this is close to the T_REC used elsewhere)
        # !TODO merge on keys['T_REC'] so that there we can later get the files.
        # !TODO check this...
        keys = pd.merge(
            left=keys, right=self.r_urls, left_on="T_REC", right_on="extracted_record_timestamp", how="left"
        )
        # ...

        #!TODO move to separate method & use default variables
        # keys["datetime"] = [datetime.datetime.strptime(date, "%Y-%m-%dT%H:%M:%S.%fZ") for date in keys["DATE-OBS"]]
        keys["datetime"] = [
            pd.to_datetime(date, format=self.date_format, errors="coerce") for date in keys["DATE-OBS"]
        ]  # According to JSOC: [DATE-OBS] DATE_OBS = T_OBS - EXPTIME/2.0

        # logger.info(
        #     f"length of `r.urls`: {len(r.urls)}; length of `keys`: {len(keys)}`"
        # )  # the naming is different to other data..

        keys.to_csv(self.metadata_save_location)

        # keys is the keys, with links to the magnetogram
        # r.urls are urls of pure fits files.
        return keys  # , r.urls
