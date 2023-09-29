from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from pandas import DataFrame, Timedelta
from sunpy.util.parfive_helpers import Downloader

from arccnet import config
from arccnet.catalogs.active_regions.swpc import SWPCCatalog

from datetime import timedelta
from arccnet.data_generation.magnetograms.base_magnetogram import BaseMagnetogram

from arccnet.data_generation.utils.data_logger import logger

from datetime import timedelta
import astropy.units as u
from astropy.table import MaskedColumn, QTable, join, vstack
from astropy.time import Time

__all__ = ["DataManager"]


class Query(QTable):
    r"""
    Query object define both the query and results.

    The query is defined by a row with 'start_time', 'end_time' and 'url'. 'url' is `MaskedColum` and where the
    mask is `True` can be interpreted as missing data.

    Notes
    -----
    Under the hood uses QTable and Masked columns to define if a expected result is present or missing

    """
    required_column_types = {"target_time": Time, "url": str}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not set(self.colnames).issuperset(set(self.required_column_types.keys())):
            raise ValueError(
                f"{self.__class__.__name__} must contain " f"{list(self.required_column_types.keys())} columns."
            )
        # !TODO this doesn't work and we can't enforce the column type!
        # for colname, coltype in self.required_column_types.items():
        #     if colname not in self.colnames or not all(isinstance(value, coltype) for value in self[colname]):
        #         raise ValueError(f"{colname} column must contain {coltype} values.")

    @property
    def is_empty(self) -> bool:
        r"""Is the query empty."""
        return np.all(self["url"].mask == np.full(len(self), True))

    @property
    def missing(self):
        r"""Rows which are missing."""
        return self[self["url"].mask == True]  # noqa

    @classmethod
    def create_empty(cls, start, end, frequency: timedelta): #, tolerance: timedelta):
        r"""
        Create an 'empty' Query.

        Parameters
        ----------
        start
            Start time, any format supported by `astropy.time.Time`
        end
            End time, any format supported by `astropy.time.Time`

        Returns
        -------
        Query
            An empty Query
        """
        start = Time(start)
        end = Time(end)
        start_pydate = start.to_datetime()
        end_pydate = end.to_datetime()
        dt = int((end_pydate - start_pydate) / frequency) # maybe?

        target_times = Time([start_pydate + i*frequency for i in range(dt + 1)]) # maybe?
        # start_times = target_times - tolerance 
        # end_times = target_times + tolerance

        # this breaks, should it be end_pydate?
        # if not start_times[-1].isclose(end):
        #     raise ValueError(f"Expected end time {times[-1]} does not match supplied end time: {end}")

        urls = MaskedColumn(data=[""] * len(target_times), mask=np.full(len(target_times), True))
        empty_query = cls(data=[target_times, urls], names=["target_time", "url"])
        return empty_query
    
class DataManager:
    """
    Main data management class.

    This class instantiates and handles data acquisition for the individual instruments and cutouts
    """

    def __init__(
        self,
        start_date: datetime,
        end_date: datetime,
        frequency: timedelta,
        magnetograms: list[BaseMagnetogram],
    ):
        """
        Initialize the DataManager.

        Parameters
        ----------
        start_date : datetime
            Start date of data acquisition period. Default is `arccnet.data_generation.utils.default_variables.DATA_START_TIME`.

        end_date : datetime
            End date of data acquisition period. Default is `arccnet.data_generation.utils.default_variables.DATA_END_TIME`.

            
        # merge_tolerance : pd.Timedelta, optional
        #     Time tolerance for merging operations. Default is pd.Timedelta("30m").

        # download_fits : bool, optional
        #     Whether to download FITS files. Default is True.
        """
        self._start_date = start_date
        self._end_date = end_date
        self._frequency = frequency

        # Check that all class objects are subclasses of `BaseMagnetogram`
        for class_obj in magnetograms:
            if not issubclass(class_obj, BaseMagnetogram):
                raise ValueError(f"{class_obj.__name__} is not a subclass of BaseMagnetogram")

        self._mag_objects = magnetograms
        self._query_objects = [Query.create_empty(self.start_date, self.end_date, self.frequency) for _ in self._mag_objects]

    @property
    def start_date(self):
        return self._start_date

    @property
    def end_date(self):
        return self._end_date

    @property
    def tolerance(self):
        return self._tolerance

    @property
    def frequency(self):
        return self._frequency
    
    # list | int 
    def search(self, batch_frequency: int = 4, merge_tolerance: timedelta = timedelta(minutes=12)) -> list[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Fetch and return data from various sources.

        Returns
        -------
        list(Query)
            List of Query objects
        """
        logger.debug("Entering search")

        # times = None # hmm...

        # Check if batch_frequency is a list or a single value
        if isinstance(batch_frequency, list):
            # Check if the length of batch_frequency matches the number of elements in self._mag_objects
            if len(batch_frequency) != len(self._mag_objects):
                raise ValueError("Length of batch_frequency list must match the number of Magnetogram objects")
            # If it's a list, use it for each data source individually
            metadata_list = [
                data_source.fetch_metadata(self.start_date, self.end_date, batch_frequency=bf)
                for data_source, bf in zip(self._mag_objects, batch_frequency)
            ]
        else:
            # If it's a single value, use it for all data sources
            metadata_list = [
                data_source.fetch_metadata(self.start_date, self.end_date, batch_frequency=batch_frequency)
                for data_source in self._mag_objects
            ]

        for meta in metadata_list:
            logger.debug(
                f"{meta.__class__.__name__}: \n{meta[['T_REC','T_OBS','DATE-OBS','datetime', 'url']]}"
            )

        results = [] 
        for meta, query in zip(metadata_list, self._query_objects):
            # do the join in pandas, and then convert to QTable?

            # removing url, but appending Query(...) without url column will complain
            # probably a better way to deal with this
            pd_query = query.to_pandas() # [['target_time']]
            
            # generate a mapping from target_time to datetime with a tolerance.
            # probably want a test that this whole code gets the same thing if there are no duplicates...
            merged_time = pd.merge_asof(
                left=pd_query[["target_time"]],
                right=meta[['datetime']].drop_duplicates(), 
                left_on=["target_time"],
                right_on=["datetime"],
                # suffixes=["_query", "_meta"],
                tolerance=timedelta(minutes=12),  # HMI is at 720s (12 min) cadence
                direction="nearest",
            )

            # for now, ensure that there are no duplicates of the same "datetime" in the df
            if len(merged_time['datetime'].unique()) != len(merged_time['datetime']):
                raise ValueError("there are duplicates of datetime from the right df")

            # find the rows in the metadata which match the exact datetime
            # which there may be multiple for cutouts at the same full-disk time 
            # ... and join
            matched_rows = meta[meta['datetime'].isin(merged_time['datetime'])]

            # -- Bit hacky to stop HARPNUM becoming a float
            #    I think Shane may have found a better way to deal with this?
            # Convert int64 columns to Int64
            int64_columns = matched_rows.select_dtypes(include=['int64']).columns
            # Create a new DataFrame with Int64 data types
            new_df = matched_rows.copy()
            for col in int64_columns:
                new_df[col] = matched_rows[col].astype('Int64')

            merged_df = pd.merge(merged_time, new_df, on='datetime', how='left')
            
            result = Query(QTable.from_pandas(merged_df))
            # !TODO Replace NaN values in the "url" column with masked values or change this...
            # remove columns ? 
            # rename columns ?

            results.append(Query(result))

        # !TODO merge _query_objects and mag_tables
        logger.debug("Exiting search")
        return results
    
    def download(self, query_list: list(dict)):
        logger.debug("Entering download")

        downloads = []
        for query in query_list:
            # expand like swpc
            missing = query["path"] == ""
            new_query = query[missing]
            downloads = new_query[~new_query["url"].mask]
            # downloads = self._download(downloads, overwrite, path)
            # results = self._match(results, downloads)]
            # downloads.append(results)

        logger.debug("Exiting download")
        return downloads

    @staticmethod
    def download_from_column(
        urls_df: pd.DataFrame = None,
        column_name="url",
        suffix="",
        base_directory_path: Path = None,
        max_retries: int = 5,
        overwrite: bool = False,
    ) -> pd.DataFrame:
        """
        Download data from URLs in a DataFrame using parfive.

        Parameters
        ----------
        urls_df : pd.DataFrame
            DataFrame containing a "url" column with URLs to download.

        base_directory_path : Path
            Base directory path to save downloaded files. Default is `arccnet.data_generation.utils.default_variables.MAG_RAW_DATA_DIR`.

        max_retries : int, optional
            Maximum number of download retries. Default is 5.

        Returns
        -------
        urls_df
             DataFrame containing "download_path" and "downloaded_successfully" columns
        """
        if urls_df is None or not isinstance(urls_df, pd.DataFrame) or column_name not in urls_df.columns or base_directory_path:
            logger.warning(f"Invalid DataFrame format. Expected a DataFrame with a '{column_name}' column.")
            return None
        if base_directory_path is None:
            logger.warning(f"Provide a base_directory_path.")
            raise ValueError("base_directory_path is None")

        downloader = Downloader(
            max_conn=1,
            progress=True,
            overwrite=False,
            max_splits=1,
        )

        # setup dataframe with download_paths
        urls_df["download_path" + suffix] = [
            base_directory_path / Path(url).name if isinstance(url, str) else np.nan for url in urls_df[column_name]
        ]
        downloaded_successfully = []
        for path in urls_df["download_path" + suffix]:
            if isinstance(path, Path):
                downloaded_successfully.append(False)
            elif pd.isna(path):
                downloaded_successfully.append(np.nan)
            else:
                raise InvalidDownloadPathError(f"Invalid download path: {path}")

        urls_df["downloaded_successfully" + suffix] = downloaded_successfully

        fileskip_counter = 0
        for _, row in urls_df.iterrows():
            if row[column_name] is not np.nan:
                # download only if it doesn't exist or overwrite is True
                # this assumes that parfive deals with checking the integrity of files downloaded
                # and that none are corrupt
                if not row["download_path" + suffix].exists() or overwrite:
                    downloader.enqueue_file(row[column_name], filename=row["download_path" + suffix])
                else:
                    fileskip_counter += 1

        if fileskip_counter > 0:
            logger.info(f"{fileskip_counter} files already exist at the destination, and will not be overwritten.")

        results = downloader.download()

        if len(results.errors) != 0:
            logger.warning(f"results.errors: {results.errors}")
            retry_count = 0
            while len(results.errors) != 0 and retry_count < max_retries:
                logger.info("retrying...")
                downloader.retry(results)
                retry_count += 1
            if len(results.errors) != 0:
                logger.error("Failed after maximum retries.")
            else:
                logger.info("Errors resolved after retry.")
        else:
            logger.info("No errors reported by parfive")

        parfive_download_errors = [errors.url for errors in results.errors]

        urls_df["downloaded_successfully" + suffix] = [
            url not in parfive_download_errors if isinstance(url, str) else url for url in urls_df[column_name]
        ]

        return urls_df
    
def merge_activeregionpatches(
    full_disk_data,
    cutout_data,
) -> pd.DataFrame:
    """
    Merge active region patch data.

        Parameters
        ----------
        full_disk_data : `DataFrame`
            Data from full disk observations with columns;
            ["datetime", "url"]

        cutout_data : `DataFrame`
            Data from active region cutouts. The pd.DataFrame must contain a "datetime" column.

        Returns
        -------
        `DataFrame`
            Merged DataFrame of active region patch data.
        """
        expected_columns = ["datetime", "url"]

    if not set(expected_columns).issubset(full_disk_data.columns):
        logger.warn(
            "Currently, the columns in full_disk_data need to be ['datetime', 'url'] due to poor planning and thought."
        )
        raise NotImplementedError()

    if not set(expected_columns).issubset(cutout_data.columns):
        logger.warn(
            "Currently, the columns in cutout_data need to be ['datetime', 'url'] due to poor planning and thought."
        )
        raise NotImplementedError()

    full_disk_data_dropna = full_disk_data.dropna().reset_index(drop=True)

    # !TODO can probably just do this in the merge with suffix = ...
    cutout_data = cutout_data.add_suffix("_arc")
    cutout_data_dropna = cutout_data.dropna().reset_index(drop=True)

    # need to figure out if a left merge is what we want...
    merged_df = pd.merge(
        full_disk_data_dropna, cutout_data_dropna, left_on="datetime", right_on="datetime_arc"
    )  # no tolerance as should be exact!

    return merged_df

def merge_srshmimdi_metadata(
    srs_keys: pd.DataFrame,
    hmi_keys: pd.DataFrame,
    mdi_keys: pd.DataFrame,
    tolerance: pd.Timedelta = pd.Timedelta("30m"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge SRS, HMI, and MDI metadata.
    def merge_hmimdi_metadata(
        self,
        srs_keys: DataFrame,
        hmi_keys: DataFrame,
        mdi_keys: DataFrame,
        tolerance: Timedelta = pd.Timedelta("30m"),
    ) -> tuple[DataFrame, DataFrame]:
        """
        Merge SRS, HMI, and MDI metadata.

    This function merges NOAA SRS, Helioseismic and Magnetic Imager (HMI),
    and Michelson Doppler Imager (MDI) metadata based on datetime keys with a specified tolerance.

        Parameters
        ----------
        srs_keys : `DataFrame`
            DataFrame containing SRS metadata.

        hmi_keys : `DataFrame`
            DataFrame containing HMI metadata.

        mdi_keys : `DataFrame`
            DataFrame containing MDI metadata.

        tolerance : `Timedelta`, optional
            Time tolerance for merging operations. Default is pd.Timedelta("30m").

        Returns
        -------
        `DataFrame`
            Merged DataFrame of SRS, HMI, and MDI metadata.

        `DataFrame`
            DataFrame of dropped rows after merging.
        """
        srs_keys = srs_keys.copy(deep=True)
        # merge srs_clean and hmi
        mag_cols = ["datetime", "url"]

    # !TODO do a check for certain keys (no duplicates...)
    # extract only the relevant HMI keys, and rename
    # (should probably do this earlier on)
    hmi_keys = hmi_keys[mag_cols].copy(deep=True)
    hmi_keys = hmi_keys.add_suffix("_hmi")
    hmi_keys_dropna = hmi_keys.dropna().reset_index(drop=True)

    mdi_keys = mdi_keys[mag_cols].copy(deep=True)
    mdi_keys = mdi_keys.add_suffix("_mdi")
    mdi_keys_dropna = mdi_keys.dropna().reset_index(drop=True)

    # Rename columns in srs_clean with suffix "_srs" except for exclude_columns
    exclude_columns = [
        "ID",
        "Number",
        "Carrington Longitude",
        "Area",
        "Z",
        "Longitudinal Extent",
        "Number of Sunspots",
        "Mag Type",
        "Latitude",
        "Longitude",
    ]
    srs_keys.columns = [f"{col}_srs" if col not in exclude_columns else col for col in srs_keys.columns]
    #

    # both `pd.DataFrame` must be sorted based on the key !
    merged_df = pd.merge_asof(
        left=srs_keys,
        right=hmi_keys_dropna,
        left_on="datetime_srs",
        right_on="datetime_hmi",
        suffixes=["_srs", "_hmi"],
        tolerance=tolerance,  # HMI is at 720s (12 min) cadence
        direction="nearest",
    )

    # Merge the SRS-HMI df with MDI
    merged_df = pd.merge_asof(
        left=merged_df,
        right=mdi_keys_dropna,
        left_on="datetime_srs",
        right_on="datetime_mdi",
        suffixes=["_srs", "_mdi"],
        tolerance=tolerance,
        direction="nearest",
    )

    # with a SRS-HMI-MDI df, remove NaNs
    # Drop rows where there is no HMI or MDI match to SRS data
    dropped_rows = merged_df.copy()
    merged_df = merged_df.dropna(subset=["datetime_srs"])  # Don't think this is necessary
    merged_df = merged_df.dropna(subset=["datetime_hmi", "datetime_mdi"], how="all")
    dropped_rows = dropped_rows[~dropped_rows.index.isin(merged_df.index)].copy()

    logger.info(f"merged_df: \n{merged_df[['datetime_srs', 'datetime_hmi', 'datetime_mdi']]}")
    logger.info(f"dropped_rows: \n{dropped_rows[['datetime_srs', 'datetime_hmi', 'datetime_mdi']]}")
    logger.info(f"dates dropped: \n{dropped_rows['datetime_srs'].unique()}")

    # return the merged dataframes with dropped indices
    return merged_df.reset_index(drop=True), dropped_rows.reset_index(drop=True)

    def fetch_fits(
        self,
        urls_df: DataFrame = None,
        column_name="url",
        suffix="",
        base_directory_path: Path = Path(config["paths"]["mag_raw_data_dir"]),
        max_retries: int = 5,
        overwrite: bool = False,
    ) -> DataFrame:
        """
        Download data from URLs in a DataFrame using parfive.

        Parameters
        ----------
        urls_df : `DataFrame`
            DataFrame containing a "url" column with URLs to download.

        base_directory_path : `Path`, optional
            Base directory path to save downloaded files. Default is `arccnet.data_generation.utils.default_variables.MAG_RAW_DATA_DIR`.

        max_retries : `int`, optional
            Maximum number of download retries. Default is 5.

        Returns
        -------
        `DataFrame`
            New data frame containing "download_path" and "downloaded_successfully" columns
        """
        if urls_df is None or not isinstance(urls_df, pd.DataFrame) or column_name not in urls_df.columns:
            logger.warning(f"Invalid DataFrame format. Expected a DataFrame with a '{column_name}' column.")
            return None

        downloader = Downloader(
            max_conn=1,
            progress=True,
            overwrite=False,
            max_splits=1,
        )

        # setup dataframe with download_paths

        urls_df["download_path" + suffix] = [
            base_directory_path / Path(url).name if isinstance(url, str) else np.nan for url in urls_df[column_name]
        ]
        downloaded_successfully = []
        for path in urls_df["download_path" + suffix]:
            if isinstance(path, Path):
                downloaded_successfully.append(False)
            elif pd.isna(path):
                downloaded_successfully.append(np.nan)
            else:
                raise InvalidDownloadPathError(f"Invalid download path: {path}")

        urls_df["downloaded_successfully" + suffix] = downloaded_successfully

        fileskip_counter = 0
        for _, row in urls_df.iterrows():
            if row[column_name] is not np.nan:
                # download only if it doesn't exist or overwrite is True
                # this assumes that parfive deals with checking the integrity of files downloaded
                # and that none are corrupt
                if not row["download_path" + suffix].exists() or overwrite:
                    downloader.enqueue_file(row[column_name], filename=row["download_path" + suffix])
                else:
                    fileskip_counter += 1

        if fileskip_counter > 0:
            logger.info(f"{fileskip_counter} files already exist at the destination, and will not be overwritten.")

        results = downloader.download()

        if len(results.errors) != 0:
            logger.warning(f"results.errors: {results.errors}")
            retry_count = 0
            while len(results.errors) != 0 and retry_count < max_retries:
                logger.info("retrying...")
                downloader.retry(results)
                retry_count += 1
            if len(results.errors) != 0:
                logger.error("Failed after maximum retries.")
            else:
                logger.info("Errors resolved after retry.")
        else:
            logger.info("No errors reported by parfive")

        parfive_download_errors = [errors.url for errors in results.errors]

        urls_df["downloaded_successfully" + suffix] = [
            url not in parfive_download_errors if isinstance(url, str) else url for url in urls_df[column_name]
        ]

        return urls_df


class InvalidDownloadPathError(Exception):
    pass
