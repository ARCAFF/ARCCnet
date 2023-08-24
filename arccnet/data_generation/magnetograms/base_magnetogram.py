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

    @abstractmethod
    def _get_matching_info_from_record(self, records: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """
        Extract matching information from records in a DataFrame.

        This method processes a DataFrame containing records and extracts relevant information,
        such as dates and HARPNUMs/TARPNUMs, using regular expressions.

        Parameters
        ----------
        records : pd.DataFrame
            A DataFrame column containing records to extract information from.

        Returns
        -------
        tuple[pd.DataFrame, list[str]]
            A tuple containing a DataFrame with extracted information and a list of column names for the extracted data.

        """
        raise NotImplementedError("This is the required method in the child class.")

    def _query_jsoc(self, query: str) -> tuple[pd.DataFrame, pd.Series]:
        keys, segs = self._drms_client.query(
            query,
            key="**ALL**",  # drms.const.all = '**ALL**'
            seg=self.segment_column_name,  # !TODO remove this from here as it will pull everything...
        )
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
        # !TODO, shouldn't have to do this; the query should be the query
        if isinstance(self.segment_column_name, list):
            formatted_string = "{" + ", ".join([f"{seg}" for seg in self.segment_column_name]) + "}"
        else:
            formatted_string = f"{{{self.segment_column_name}}}"
        logger.info(f"exporting the query: {query + formatted_string}")

        export_response = self._drms_client.export(query + formatted_string, method="url", protocol="fits", **kwargs)
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

    def _save_metadata_to_csv(self, keys: pd.DataFrame, filepath: str = None) -> None:
        """
        Save metadata to CSV
        """
        if filepath is None:
            filepath = self.metadata_save_location

        file = Path(filepath)
        logger.info(f"The metadata save filepath is {file}")

        # !TODO make a utility function here
        directory_path = file.parent
        if not directory_path.exists():
            directory_path.mkdir(parents=True)

        keys.to_csv(file)

    def _add_extracted_columns_to_df(
        self, df: pd.DataFrame, df_colname: str = "record"
    ) -> tuple[pd.DataFrame, list[str], list[str]]:
        """
        Add extracted information to a pandas DataFrame

        This method extracts relevant information from the specified source column in the DataFrame,
        processes the data using the `_get_matching_info_from_record` method in the child class,
        and adds the extracted data to the DataFrame with appropriate column names.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to which extracted information will be added.
        source_col : str, optional
            The name of the source column containing the data to extract from. D
            Defaults to "record".

        Returns
        -------
            tuple[pd.DataFrame, list[str], list[str]]: A tuple containing the updated DataFrame,
            a list of merge columns, and a list of the corresponding column names.
        """

        original_column = df[df_colname]
        extracted_data = self._get_matching_info_from_record(records=original_column)
        merged_columns = extracted_data.columns.tolist()
        column_names = [f"{df_colname}_{col}" for col in merged_columns]
        df[column_names] = extracted_data
        return df, merged_columns, column_names

    def fetch_metadata(
        self, start_date: datetime.datetime, end_date: datetime.datetime, to_csv: bool = True
    ) -> pd.DataFrame:
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
            raise ValueError(f"No results return for the query: {query}!")
        else:
            logger.info(f"\t {len(keys)} entries")

        self._add_magnetogram_urls(keys, segs, url=dv.JSOC_BASE_URL, column_name="magnetogram_fits")
        r_urls = self._export_files(query)

        # extract info e.g. date, active region number from the `r_url["record"]`
        # and insert back into r_urls as additional column names for merging
        r_urls_plus, merge_columns, column_names = self._add_extracted_columns_to_df(df=r_urls, df_colname="record")

        keys_merged = pd.merge(
            left=keys,
            right=r_urls_plus,
            left_on=merge_columns,
            right_on=column_names,
            how="left",
        )

        # Check for duplicated rows
        if keys_merged.duplicated(subset=[column_names]).sum() > 0:
            logger.warn(
                f"keys_merged.duplicated(subset=[column_names]).sum() is {keys_merged.duplicated(subset=[column_names]).sum()}"
            )

        keys["datetime"] = pd.to_datetime(
            keys["DATE-OBS"], format=self.date_format, errors="coerce"
        )  # !TODO investigate coerce
        # keys["datetime"] = [datetime.datetime.strptime(date, "%Y-%m-%dT%H:%M:%S.%fZ") for date in keys["DATE-OBS"]]
        # keys["datetime"] = [
        #     pd.to_datetime(date, format=self.date_format, errors="coerce")
        #     for date in keys["DATE-OBS"]  # ensure we want errors="coerce"
        # ]  # According to JSOC: [DATE-OBS] DATE_OBS = T_OBS - EXPTIME/2.0

        if to_csv:
            self._save_metadata_to_csv(keys_merged)

        return keys_merged

    def validate_metadata(self):
        # not sure how to validate
        raise NotImplementedError("Metadata validation is not implemented")
