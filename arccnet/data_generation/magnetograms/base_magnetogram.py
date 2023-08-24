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

    @property
    @abstractmethod
    def _get_matching_info_from_record(self, record: str) -> tuple[tuple[str, str], list[str]]:
        raise NotImplementedError("This is the required method in the child class.")

    def _query_jsoc(self, query: str) -> tuple[pd.DataFrame, pd.Series]:
        keys, segs = self._drms_client.query(
            query, key=drms.const.all, seg=self.segment_column_name
        )  # !TODO make the drms key quert optional
        return keys, segs

    def _add_magnetogram_urls(
        self, keys: pd.DataFrame, segs: pd.Series, url: str = dv.JSOC_BASE_URL, column_name: str = "magnetogram_fits"
    ) -> None:
        magnetogram_fits = url + segs[self.segment_column_name]
        keys[column_name] = magnetogram_fits

    def _export_files(
        self,
        query: str,
        # column_name: str = "extracted_record_timestamp",
        **kwargs,
    ) -> None:
        export_response = self._drms_client.export(query, method="url", protocol="fits", **kwargs)
        export_response.wait()
        # extract the `record` and strip the square brackets to return a T_REC-like time (in TAI)
        r_urls = export_response.urls.copy()
        # `self.r_urls["record"].str.extract(r"\[(.*?)\]")`` will only extract the HARP num from:
        #   `hmi.sharp_720s[318][2011.01.01_00:00:00_TAI]`
        # so using r"\[([^\[\]]*?(?=\]))\][^\[\]]*$" to extract last one.
        # r_urls[column_name] = self.r_urls["record"].str.extract(
        #     r"\[(.*?)\]"
        # )  #!TODO this nedes to be put into it's own thing
        # merge on keys['T_REC'] so that there we can later get the files.
        return r_urls

    def _merge_urls_with_keys(self, keys: pd.DataFrame, right: pd.DataFrame, left_cols, right_cols) -> pd.DataFrame:
        """
        merge dataframes on columns
        """
        return pd.merge(
            left=keys,
            right=right,
            left_on=left_cols,  # , "HARPNUM"],
            right_on=right_cols,  # ["extracted_date"],  # , "extracted_harpnum"],
            how="left",
        )

    def _save_metadata_to_csv(self, keys: pd.DataFrame) -> None:
        file = Path(self.metadata_save_location)

        logger.info(f"The metadata save file is {file}")

        # !TODO make a utility function here
        directory_path = file.parent
        if not directory_path.exists():
            directory_path.mkdir(parents=True)

        keys.to_csv(file)

    def fetch_metadata(self, start_date: datetime.datetime, end_date: datetime.datetime, to_csv=False) -> pd.DataFrame:
        """
        Fetch metadata from JSOC.

        Returns
        -------
        keys: pd.DataFrame
            A `pd.DataFrame` containing all keys (`drms.const.all`) for a JSOC query string and urls corresponding to the request segments (`seg`).
            The required segment is defined in the child class (for a magnetogram, `seg="magnetogram"` for HMI, and `seg="data"` for MDI).

            The `pd.DataFrame` also contains `urls` to the complete `.fits` files (magnetogram + metadata) that are staged by JSOC for download.

        """

        query = self.generate_drms_query(start_date, end_date)
        logger.info(f">> {self._type()} Query: {query}")

        keys, segs = self._query_jsoc(query)
        if len(keys) == 0:
            # !TODO implement custom error message
            raise (f"No results return for the query: {query}!")
        else:
            logger.info(f"\t {len(keys)} entries")

        self._add_magnetogram_urls(keys, segs, url=dv.JSOC_BASE_URL, column_name="magnetogram_fits")
        r_urls = self._export_files(query)  # , column_name="extracted_record_timestamp")

        # r_urls["extracted_record_timestamp"] = self.r_urls["record"].str.extract(
        #     r"\[(.*?)\]"
        # ) # extract_

        # We need to match the record to the keys.
        # The following code uses the `_get_matching_info_from_record` method in the child class
        # to extract the relevant part of the record string that corresponds to columns in `keys`
        # such that keys and r_urls can be merged.
        # For an instrument like HMI or MDI, this is straight forward and can be done on T_REC.
        # For cutouts, where there are multiple regions for a given T_REC, another property e.g. HARPNUM
        # also needs to be extracted.
        extracted_data, merge_columns = r_urls["record"].apply(self._get_matching_info_from_record)
        column_names = [f"extracted_{i}" for i in range(merge_columns)]
        r_urls[column_names] = pd.DataFrame(extracted_data.to_list(), columns=column_names)

        keys = self._merge_urls_with_keys(keys, r_urls, left_cols=merge_columns, right_cols=column_names)

        # .....................................................................
        # keys = self._merge_urls_with_keys(
        #     keys, r_urls, right_cols=["extracted_record_timestamp"]
        # )  # !TODO fix this so it's the solution
        # # !TODO here we need to extract the urls and matching based on TREC/date and HARPNUM (or others)
        # #       as the following only works on HMI/MDI, not SHARPs.
        # # r_urls["extracted_record_timestamp"] = self.r_urls["record"].apply(self._get_date_from_record)
        # keys = pd.merge(
        #     left=keys, right=r_urls, left_on="T_REC", right_on="extracted_record_timestamp", how="left"
        # )  # This doesn't work on SHARP data
        # .....................................................................

        keys["datetime"] = pd.to_datetime(keys["DATE-OBS"], format=self.date_format, errors="coerce")
        # keys["datetime"] = [datetime.datetime.strptime(date, "%Y-%m-%dT%H:%M:%S.%fZ") for date in keys["DATE-OBS"]]
        # keys["datetime"] = [
        #     pd.to_datetime(date, format=self.date_format, errors="coerce")
        #     for date in keys["DATE-OBS"]  # ensure we want errors="coerce"
        # ]  # According to JSOC: [DATE-OBS] DATE_OBS = T_OBS - EXPTIME/2.0

        if to_csv:
            self._save_metadata_to_csv(keys)

    def validate_metadata(self):
        # not sure how to validate
        raise NotImplementedError()
