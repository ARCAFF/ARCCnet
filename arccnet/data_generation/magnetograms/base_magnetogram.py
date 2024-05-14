import time
import urllib
import datetime
import http.client
from abc import ABC, abstractmethod

import drms
import pandas as pd

from arccnet import config
from arccnet.utils.logging import logger

__all__ = ["BaseMagnetogram"]


class BaseMagnetogram(ABC):
    def __init__(self) -> None:
        super().__init__()
        self._drms_client = drms.Client(debug=False, verbose=False, email=config["jsoc"]["jsoc_default_email"])

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

        frequency : `str`, optional
            A string representing the frequency of observations. Default is "1d" (1 day).
            Valid frequency strings can be specified, such as "1h" for 1 hour, "15T" for 15 minutes,
            "1M" for 1 month, "1Y" for 1 year, and more. Refer to the pandas documentation for a complete
            list of valid frequency strings: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases

        Returns
        -------
        str
            The JSOC query string for retrieving the specified observations.
        """
        pass

    @property
    @abstractmethod
    def series_name(self) -> str:
        """
        Get the JSOC series name.

        Returns
        -------
        str:
            JSOC series name
        """
        pass

    @property
    @abstractmethod
    def date_format(self) -> str:
        """
        Get the date string format used by the instrument.

        Returns
        -------
        str:
            instrument date string format
        """
        pass

    @property
    @abstractmethod
    def segment_column_name(self) -> str:
        """
        Get the name of the data segment.

        Returns
        -------
        str:
            Name of the data segment
        """
        pass

    @property
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
    def _get_matching_info_from_record(self, records: pd.Series) -> tuple[pd.DataFrame, list[str]]:
        """
        Extract matching information from records in a DataFrame.

        This method processes a DataFrame containing records and extracts relevant information,
        such as dates and HARPNUMs/TARPNUMs, using regular expressions.

        Parameters
        ----------
        records : pd.Series
            A DataFrame column containing records to extract information from.

        Returns
        -------
        tuple[pd.DataFrame, list[str]]
            A tuple containing a DataFrame with extracted information and a list of column names for the extracted data.
        """
        pass

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
            key="**ALL**",  # the needed columns vary, **ALL** returns all available keys
            seg=self.segment_column_name,
        )
        return keys, segs

    def _add_magnetogram_urls(
        self,
        df: pd.DataFrame,
        segments: pd.Series,
        url: str = config["jsoc"]["jsoc_base_url"],
        column_name: str = "magnetogram_fits",
    ) -> pd.DataFrame:
        """
        Add magnetogram URLs to the DataFrame.

        This method generates magnetogram URLs based on the provided segments and adds them to the DataFrame under the
        specified column name, if the column doesn't already exist.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to which magnetogram URLs should be added.

        segments : pd.Series
            A Series containing filenames of JSOC series segments.

        url : `str`, optional
            The base URL for constructing the magnetogram URLs from the segments.

        column_name : `str`, optional
            The name of the column to store the magnetogram URLs. This must not already exist in the DataFrame.

        Returns
        -------
        pd.DataFrame
            The updated DataFrame with the added magnetogram URLs, if the column was added successfully.

        Raises
        ------
        ValueError
            If the specified column already exists in the DataFrame.
        """
        magnetogram_fits = url + segments[self.segment_column_name]

        if column_name not in df.columns:
            new_column = pd.DataFrame({column_name: magnetogram_fits})
            df_with_url_column = pd.concat([df, new_column], axis=1)
            return df_with_url_column
        else:
            raise ValueError(f"Column '{column_name}' already exists in the DataFrame.")

    def _data_export_request_with_retry(
        self, query: str, max_retries=5, retry_delay=60, drms_export_delay=60, **kwargs
    ) -> pd.DataFrame:
        """
        Submit a data export request with retries and return the URLs.

        This method provides a wrapper to `_data_export_request` to include a retry mechanism in case of connection issues.

        Parameters
        ----------
        query : str
            The JSOC query string.

        max_retries : `int`, optional
            The maximum number of retries before raising an exception (default is 3).

        retry_delay : `int`, optional
            The delay (in seconds) between retries (default is 60).

        **kwargs
            Additional keyword arguments for exporting files URLs.

        Returns
        -------
        result : object or None
            The result of the data export request if successful; None if all retries fail.

        Raises
        ------
        DataExportRequestError
            If the data export request fails even after the maximum number of retries.
        """
        retries = 1
        while retries <= max_retries:
            try:
                return self._data_export_request(query, **kwargs)
            except Exception as e:
                # Two primary Exceptions have been raised in testing:
                # 1. http.client.RemoteDisconnected
                # 2. drms.exceptions.DrmsExportError
                # The latter occurs after the former due to the request still pending
                if isinstance(e, http.client.RemoteDisconnected):
                    logger.warning(
                        f"\t ... Exception: '{e}' raised. Retrying in {retry_delay} seconds: retry {retries} of {max_retries}."
                    )
                    time.sleep(retry_delay)
                    retries += 1
                elif isinstance(e, drms.exceptions.DrmsExportError):
                    if "pending export requests" in str(e):
                        logger.info(
                            f"\t ... waiting {drms_export_delay} seconds for pending export requests to complete."
                        )
                        time.sleep(drms_export_delay)
                    else:
                        logger.warning(
                            f"\t ... Exception: '{e}' raised. Retrying in {retry_delay} seconds: retry {retries} of {max_retries}."
                        )
                        time.sleep(retry_delay)
                        retries += 1
                elif isinstance(e, urllib.error.HTTPError) and e.code == 504:
                    logger.info(f"\t ... HTTP Error 504: waiting {drms_export_delay} seconds before trying again.")
                    time.sleep(drms_export_delay)
                    retries += 1
                else:
                    raise e

        raise DataExportRequestError("Failed to export data after multiple retries")

    def _data_export_request(
        self,
        query: str,
        **kwargs,
    ) -> None:
        """
        Submit a data export request and return the urls.

        This method submoits a data export request to JSOC based on the provided query and additional keyword arguments.

        Parameters
        ----------
        query : str
            The JSOC query string.

        **kwargs
            Additional keyword arguments for exporting files urls.

        Returns
        -------
        r_urls : pd.DataFrame
            urls extracted from the export response
        """
        # !TODO, shouldn't have to do this; the query should be the query
        if isinstance(self.segment_column_name, list):
            formatted_string = "{" + ", ".join([f"{seg}" for seg in self.segment_column_name]) + "}"
        else:
            formatted_string = f"{{{self.segment_column_name}}}"
        logger.info(f"\t ... requesting {self.segment_column_name} urls from JSOC")

        export_response = self._drms_client.export(query + formatted_string, method="url", protocol="fits", **kwargs)
        export_response.wait()
        r_urls = export_response.urls.copy()
        return r_urls

    def _add_extracted_columns_to_df(
        self, df: pd.DataFrame, df_colname: str = "record"
    ) -> tuple[pd.DataFrame, list[str], list[str]]:
        """
        Add extracted information to a pandas DataFrame.

        This method extracts relevant information from the specified source column in the DataFrame,
        processes the data using the `_get_matching_info_from_record` method in the child class,
        and adds the extracted data to the DataFrame with appropriate column names.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to which extracted information will be added.

        df_colname : `str`, optional
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
        # !TODO this can be tidied up considerably
        original_column = df[df_colname]
        extracted_data = self._get_matching_info_from_record(records=original_column)
        merged_columns = extracted_data.columns.tolist()
        column_names = [f"{df_colname}_{col}" for col in merged_columns]

        # Check if the columns already exist in the DataFrame
        existing_columns = [col for col in column_names if col in df.columns]

        if existing_columns:
            raise ValueError(f"Columns {', '.join(existing_columns)} already exist in the DataFrame.")

        # Prefix the column names of the extracted data with df_colname
        extracted_data = extracted_data.add_prefix(f"{df_colname}_")

        # Concatenate extracted columns to the DataFrame
        # !TODO what happens if these are different sizes?
        df = pd.concat([df, extracted_data], axis=1)

        return df, merged_columns, column_names

    def fetch_metadata(
        self,
        start_date: datetime.datetime,
        end_date: datetime.datetime,
        batch_frequency: int = 3,
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

        batch_frequency : `int`, optional
            The frequency for each batch.
            Default is 3 (3 months), empirically determined based on the density of files seen in SHARPs queries.

        dynamic_columns : list[str]
            Columns that will be different with each request. This is used for dropping duplicates.
            For example, JSOC prepares the set of files at each request, returning unique URLs.
            Default is ["url"].

        Returns
        -------
        `DataFrame` or None
            A pandas DataFrame containing metadata and URLs for requested data segments.
            Returns None if there is no metadata.

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

        See Also
        --------
        fetch_metadata_batch
        """
        logger.info(
            f">> Fetching metadata for {self._type}. The requests are batched into batches of {batch_frequency} months"
        )
        batch_start = start_date
        all_metadata = []

        counter = 0
        while batch_start < end_date:
            batch_end = batch_start + pd.offsets.DateOffset(months=batch_frequency)
            if batch_end > end_date:
                batch_end = end_date

            metadata_batch = self.fetch_metadata_batch(batch_start, batch_end)

            if metadata_batch is not None:  # Check if the batch is not empty or None
                all_metadata.append(metadata_batch)

            batch_start = batch_end
            counter += 1

        if len(all_metadata) > 0:
            combined_metadata = pd.concat(all_metadata, ignore_index=True)  # test this
        else:
            logger.warning("No metadata from this query")
            return None

        # Check for duplicated rows in the combined metadata because we might be doing this accidentally
        # the "url" column is dynamic, and will not match (will the urls persist until we download them?)
        # !TODO we need a better way of dealing with situations like this
        columns_to_check = [col for col in combined_metadata.columns if col not in dynamic_columns]
        combined_metadata = combined_metadata.drop_duplicates(subset=columns_to_check).reset_index(drop=True)

        logger.debug(combined_metadata.shape)
        return combined_metadata

    def fetch_metadata_batch(
        self,
        start_date: datetime.datetime,
        end_date: datetime.datetime,
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


        Returns
        -------
        `DataFrame` or None
            A pandas DataFrame containing metadata and URLs for requested data segments.
            The DataFrame has columns corresponding to metadata keys, URLs, and additional extracted information.
            If no results are returned from the JSOC query, `None` is returned.

        Notes
        -----
        The required data segment (`seg`) is defined in the child class. For instance, for magnetograms, `seg="magnetogram"`,
        and for HMI and MDI, `seg="data"`.

        The DataFrame includes the following columns:
        - Metadata keys corresponding to a JSOC query string.
        - URLs pointing to the complete `.fits` files (magnetogram + metadata) staged by JSOC for download.
        - Extracted information such as dates and identifiers.

        Duplicate rows are checked and logged as warnings if found.

        See Also
        --------
        generate_drms_query, _query_jsoc, _add_magnetogram_urls, _data_export_request, _add_extracted_columns_to_df
        """

        query = self.generate_drms_query(start_date, end_date)
        query_string = f"\t {self._type} Query: {query} "

        keys, _ = self._query_jsoc(query)
        if len(keys) == 0:
            # return None if there are no results
            logger.warning(query_string + f"returned {len(keys)} results")
            return None
        else:
            logger.info(query_string + f"returned {len(keys)} results")

        # NB: There are two files presented here that are essentially of the same thing.
        #   1. segs is a list of data files. The full .fits can be made with keys & segs. (commented out)
        #   2. r_urls provides the urls of the full .fits files
        # self._add_magnetogram_urls(keys, segs, url="http://jsoc.stanford.edu", column_name="magnetogram_fits")
        r_urls = self._data_export_request_with_retry(query)
        # extract info e.g. date, active region number from the `r_url["record"]` using `_get_matching_info_from_record`
        # and insert back into r_urls as additional column names.
        # for now, return `merge_columns` which are the columns in the original keys that correspond to `column_names`
        r_urls_plus, merge_columns, column_names = self._add_extracted_columns_to_df(df=r_urls, df_colname="record")
        # --

        keys_merged = pd.merge(
            left=keys,
            right=r_urls_plus,
            left_on=merge_columns,  # columns to merge on, e.g. "T_REC"
            right_on=column_names,  # column names in data, e.g. "record_T_REC"
            how="left",
        )

        # Check for duplicated rows of merge_columns/column_names pairs
        if keys_merged.duplicated(subset=column_names + merge_columns).any():
            duplicate_count = keys_merged.duplicated(subset=column_names).sum()
            logger.warning(f"There are {duplicate_count} duplicated rows in the DataFrame.")

        datetime_column = pd.to_datetime(  # if 'coerce', then invalid parsing will be set as NaT.
            keys_merged["DATE-OBS"], format=self.date_format, errors="coerce"
        )  # is DATE-OBS what we want to use? # According to JSOC: [DATE-OBS] DATE_OBS = T_OBS - EXPTIME/2.0

        if "datetime" not in keys_merged.columns:
            # being overly cautious about adding new columns
            datetime_df = pd.DataFrame({"datetime": datetime_column})
            # Concatenate the new datetime_df with keys_merged
            keys_merged = pd.concat([keys_merged, datetime_df], axis=1)
        else:
            raise ValueError("Column 'datetime' already exists in the DataFrame.")

        return keys_merged

    def validate_metadata(self):
        # not sure how to validate
        raise NotImplementedError("Metadata validation is not implemented")


class DataExportRequestError(Exception):
    pass
