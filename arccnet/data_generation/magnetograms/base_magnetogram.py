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
    def generate_drms_query(self, start_time: datetime, end_time: datetime, frequency: str) -> str:
        """
        Generate a JSOC query string for requesting observations within a specified time range.

        Parameters
        ----------
        start_time : datetime.datetime
            A datetime object representing the start time of the requested observations.

        end_time : datetime.datetime
            A datetime object representing the end time of the requested observations.

        frequency : str, optional
            A string representing the frequency of observations. Default is "1d" (1 day).
            Valid frequency strings can be specified, such as "1h" for 1 hour, "15T" for 15 minutes,
            "1M" for 1 month, "1Y" for 1 year, and more. Refer to the pandas documentation for a complete
            list of valid frequency strings: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases

        Returns
        -------
        str
            The JSOC query string for retrieving the specified observations.

        Raises
        ------
        NotImplementedError
            If this property is not implemented in the child class.
        """
        raise NotImplementedError("This is the required method in the child class.")

    @property
    @abstractmethod
    def series_name(self) -> str:
        """
        Get the JSOC series name.

        Returns
        -------
        str:
            JSOC series name

        Raises
        ------
        NotImplementedError
            If this property is not implemented in the child class.
        """
        raise NotImplementedError("This is the required method in the child class.")

    @property
    @abstractmethod
    def date_format(self) -> str:
        """
        Get the date string format used by the instrument.

        Returns
        -------
        str:
            instrument date string format

        Raises
        ------
        NotImplementedError
            If this property is not implemented in the child class.
        """
        raise NotImplementedError("This is the required method in the child class.")

    @property
    @abstractmethod
    def segment_column_name(self) -> str:
        """
        Get the name of the data segment.

        Returns
        -------
        str:
            Name of the data segment

        Raises
        ------
        NotImplementedError
            If this property is not implemented in the child class.
        """
        raise NotImplementedError("This is the required method in the child class.")

    @property
    @abstractmethod
    def metadata_save_location(self) -> str:
        """
        Get the directory path for saving metadata.

        Returns
        -------
        str:
            The directory path for saving metadata

        Raises
        ------
        NotImplementedError
            If this property is not implemented in the child class.
        """
        raise NotImplementedError("This is the required method in the child class.")

    def _type(self):
        """
        Get the name of the instantiated class.

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

        Raises
        ------
        NotImplementedError
            If this property is not implemented in the child class.
        """
        raise NotImplementedError("This is the required method in the child class.")

    def _query_jsoc(self, query: str) -> tuple[pd.DataFrame, pd.Series]:
        """
        Query JSOC to retrieve keys and segments.

        This method sends a query to JSOC using the DRMS client to retrieve keys and segments based on the provided query.

        Parameters
        ----------
        query : str
            The JSOC query string.

        Returns
        -------
        tuple[pd.DataFrame, pd.Series]
            A tuple containing a DataFrame with keys and a Series with segments.
        """
        keys, segs = self._drms_client.query(
            query,
            key="**ALL**",  # drms.const.all = '**ALL**'
            seg=self.segment_column_name,
        )
        return keys, segs

    def _add_magnetogram_urls(
        self, keys: pd.DataFrame, segs: pd.Series, url: str = dv.JSOC_BASE_URL, column_name: str = "magnetogram_fits"
    ) -> pd.DataFrame:
        """
        Add magnetogram URLs to the DataFrame.

        This method generates magnetogram URLs based on the provided segments and adds them to the DataFrame.

        Parameters
        ----------
        keys : pd.DataFrame
            A DataFrame containing keys.

        segs : pd.Series
            A Series containing segments.

        url : str, optional
            The base URL for constructing the magnetogram URLs.

        column_name : str, optional
            The name of the column to store the magnetogram URLs.

        Returns
        -------
        pd.DataFrame
            The updated DataFrame with the added magnetogram URLs.
        """
        magnetogram_fits = url + segs[self.segment_column_name]
        new_column = pd.DataFrame({column_name: magnetogram_fits})
        keys_with_url_column = pd.concat([keys, new_column], axis=1)
        return keys_with_url_column

    def _export_files(
        self,
        query: str,
        **kwargs,
    ) -> None:
        """
        Export data files.

        This method exports data files from JSOC based on the provided query and additional keyword arguments.

        Parameters
        ----------
        query : str
            The JSOC query string.

        **kwargs
            Additional keyword arguments for exporting files.

        Returns
        -------
        None
        """
        # !TODO, shouldn't have to do this; the query should be the query
        if isinstance(self.segment_column_name, list):
            formatted_string = "{" + ", ".join([f"{seg}" for seg in self.segment_column_name]) + "}"
        else:
            formatted_string = f"{{{self.segment_column_name}}}"
        logger.info(f"exporting the query: {query + formatted_string}")

        export_response = self._drms_client.export(query + formatted_string, method="url", protocol="fits", **kwargs)
        export_response.wait()
        r_urls = export_response.urls.copy()
        return r_urls

    def _save_metadata_to_csv(self, keys: pd.DataFrame, filepath: str = None) -> None:
        """
        Save metadata to a CSV file.

        This method saves the metadata DataFrame to a CSV file.

        Parameters
        ----------
        keys : pd.DataFrame
            A DataFrame containing metadata keys.

        filepath : str, optional
            The file path for saving the CSV file. If not provided, the default location is used (see `metadata_save_location`).

        Returns
        -------
        None
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
            The name of the source column containing the data to extract from.
            Defaults to "record".

        Returns
        -------
            tuple[pd.DataFrame, list[str], list[str]]
                A tuple containing the updated DataFrame, a list of merge columns,
                and a list of the corresponding column names.

        See Also
        --------
        _get_matching_info_from_record : Method in the child class that extracts matching information from records.
        """

        original_column = df[df_colname]
        extracted_data = self._get_matching_info_from_record(records=original_column)
        merged_columns = extracted_data.columns.tolist()
        column_names = [f"{df_colname}_{col}" for col in merged_columns]
        df[column_names] = extracted_data
        return df, merged_columns, column_names

    def fetch_metadata(
        self,
        start_date: datetime.datetime,
        end_date: datetime.datetime,
        batch_frequency: int = 6,
        to_csv: bool = True,
        dynamic_columns=["url"],
    ) -> pd.DataFrame:
        """
        Fetch metadata from JSOC in batches.

        This method retrieves metadata from the Joint Science Operations Center (JSOC) based on the specified time range.
        It fetches metadata in batches, where each batch covers a specified time frequency within the overall range.

        Parameters
        ----------
        start_date : datetime.datetime
            The start datetime for the desired time range of observations.

        end_date : datetime.datetime
            The end datetime for the desired time range of observations.

        batch_frequency : int, optional
            The frequency for each batch. Default is 6 (6 months).

        to_csv : bool, optional
            Whether to save the fetched metadata to a CSV file. Defaults to True.

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame containing metadata and URLs for requested data segments.

        Raises
        ------
        ValueError
            If no results are returned from any of the JSOC queries.

        Notes
        -----
        This method breaks down the overall time range into smaller batches based on the specified frequency.
        For each batch, it calls the fetch_metadata_batch method with the corresponding batch's time range.
        The fetched metadata for all batches is then concatenated into a single DataFrame.

        Duplicate rows are checked and logged as warnings if found.

        If `to_csv` is True, the fetched metadata is saved to a CSV file using the `_save_metadata_to_csv` method.

        See Also
        --------
        fetch_metadata_batch
        """
        logger.info(f">> batching requests into {batch_frequency} months")
        batch_start = start_date
        all_metadata = []

        while batch_start < end_date:
            batch_end = batch_start + pd.offsets.DateOffset(months=batch_frequency)
            if batch_end > end_date:
                batch_end = end_date

            logger.info(f">>    {batch_start, batch_end}")
            metadata_batch = self.fetch_metadata_batch(batch_start, batch_end, to_csv=False)
            all_metadata.append(metadata_batch)

            batch_start = batch_end

        combined_metadata = pd.concat(all_metadata, ignore_index=True)  # test this

        # Check for duplicated rows in the combined metadata because we might be doing this accidentally
        # the "url" column is dynamic, and will not match (will the urls persist until we download them?)
        columns_to_check = [col for col in combined_metadata.columns if col not in dynamic_columns]
        combined_metadata = combined_metadata.drop_duplicates(subset=columns_to_check)
        # !TODO we need a better way of dealing with situations like this
        duplicate_count = combined_metadata.duplicated(subset=columns_to_check).sum()
        if duplicate_count > 0:
            raise ValueError(f"There are {duplicate_count} duplicated rows in the DataFrame.")

        if to_csv:
            self._save_metadata_to_csv(combined_metadata)

        return combined_metadata

    def fetch_metadata_batch(
        self,
        start_date: datetime.datetime,
        end_date: datetime.datetime,
        to_csv: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch metadata batch from JSOC.

        This method retrieves metadata from the Joint Science Operations Center (JSOC) based on the specified time range.
        It constructs a query, queries the JSOC database, and fetches metadata and URLs for requested data segments.

        Parameters
        ----------
        start_date : datetime.datetime
            The start datetime for the desired time range of observations.

        end_date : datetime.datetime
            The end datetime for the desired time range of observations.

        to_csv : bool, optional
            Whether to save the fetched metadata to a CSV file. Defaults to True.

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame containing metadata and URLs for requested data segments.
            The DataFrame has columns corresponding to metadata keys, URLs, and additional extracted information.

        Raises
        ------
        ValueError
            If no results are returned from the JSOC query.

        Notes
        -----
        The required data segment (`seg`) is defined in the child class. For instance, for magnetograms, `seg="magnetogram"`,
        and for HMI and MDI, `seg="data"`.

        The DataFrame includes the following columns:
        - Metadata keys corresponding to a JSOC query string.
        - URLs pointing to the complete `.fits` files (magnetogram + metadata) staged by JSOC for download.
        - Extracted information such as dates and identifiers.

        Duplicate rows are checked and logged as warnings if found.

        If `to_csv` is True, the fetched metadata is saved to a CSV file using the `_save_metadata_to_csv` method.

        See Also
        --------
        generate_drms_query, _query_jsoc, _add_magnetogram_urls, _export_files, _add_extracted_columns_to_df, _save_metadata_to_csv
        """

        query = self.generate_drms_query(start_date, end_date)
        logger.info(f">> {self._type()} Query: {query}")

        keys, segs = self._query_jsoc(query)
        if len(keys) == 0:
            # !TODO implement custom error message
            raise ValueError(f"No results return for the query: {query}!")
        else:
            logger.info(f"\t {len(keys)} entries")

        keys = self._add_magnetogram_urls(keys, segs, url=dv.JSOC_BASE_URL, column_name="magnetogram_fits")
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

        # Check for duplicated rows of merge_columns/column_names pairs
        if keys_merged.duplicated(subset=column_names + merge_columns).any():
            duplicate_count = keys_merged.duplicated(subset=column_names).sum()
            logger.warn(f"There are {duplicate_count} duplicated rows in the DataFrame.")

        # keys_merged["datetime"] = pd.to_datetime(
        #     keys_merged["DATE-OBS"], format=self.date_format, errors="coerce"
        # )
        # /Users/pjwright/Documents/work/ARCCnet/arccnet/data_generation/magnetograms/base_magnetogram.py:482: PerformanceWarning:
        # DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.
        # Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`

        # Replaced with:
        datetime_column = pd.to_datetime(
            keys_merged["DATE-OBS"], format=self.date_format, errors="coerce"  # !TODO investigate coerce
        )  # is DATE-OBS what we want to use? # According to JSOC: [DATE-OBS] DATE_OBS = T_OBS - EXPTIME/2.0
        datetime_df = pd.DataFrame({"datetime": datetime_column})
        # Concatenate the new datetime_df with keys_merged
        keys_merged = pd.concat([keys_merged, datetime_df], axis=1)

        if to_csv:
            self._save_metadata_to_csv(keys_merged)

        return keys_merged

    def validate_metadata(self):
        # not sure how to validate
        raise NotImplementedError("Metadata validation is not implemented")
