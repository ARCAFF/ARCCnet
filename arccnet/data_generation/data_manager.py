from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
from pandas import DataFrame, Timedelta
from sunpy.util.parfive_helpers import Downloader

import arccnet import config
from arccnet.catalogs.active_regions.swpc import SWPCCatalog

from astropy.table import MaskedColumn, QTable
from astropy.time import Time

from arccnet.data_generation.magnetograms.base_magnetogram import BaseMagnetogram
from arccnet.data_generation.utils.data_logger import logger

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
    def create_empty(cls, start, end, frequency: timedelta):  # , tolerance: timedelta):
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
        dt = int((end_pydate - start_pydate) / frequency)

        target_times = Time([start_pydate + (i * frequency) for i in range(dt + 1)])

        # set urls as a masked column
        urls = MaskedColumn(data=[""] * len(target_times), mask=np.full(len(target_times), True))
        empty_query = cls(data=[target_times, urls], names=["target_time", "url"])
        return empty_query


class Result(QTable):
    r"""
    Result object define both the result and download status.

    The value of the 'path' is used to encode if the corresponding file was downloaded or not.

    Notes
    -----
    Under the hood uses QTable and Masked columns to define if a file was downloaded or not

    """
    required_column_types = {"target_time": Time, "url": str, "path": str}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not set(self.colnames).issuperset(set(self.required_column_types.keys())):
            raise ValueError(
                f"{self.__class__.__name__} must contain " f"{list(self.required_column_types.keys())} columns"
            )


class DataManager:
    """
    Main data management class.

    This class instantiates and handles data acquisition for the individual instruments and cutouts
    """

    def __init__(
        self,
        start_date: str,
        end_date: str,
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
        self._start_date = Time(start_date)
        self._end_date = Time(end_date)
        self._frequency = frequency

        # Check that all class objects are subclasses of `BaseMagnetogram`
        for class_obj in magnetograms:
            if not issubclass(class_obj.__class__, BaseMagnetogram):
                raise ValueError(f"{class_obj.__name__} is not a subclass of BaseMagnetogram")

        self._mag_objects = magnetograms
        self._query_objects = [
            Query.create_empty(self.start_date, self.end_date, self.frequency) for _ in self._mag_objects
        ]

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

    @property
    def mag_objects(self):
        return self._mag_objects

    @property
    def query_objects(self):
        return self._query_objects

    def search(self, batch_frequency: int = 4, merge_tolerance: timedelta = timedelta(minutes=12)) -> list[Query]:
        """
        Fetch and return data from various sources.

        Returns
        -------
        list(Query)
            List of Query objects
        """
        logger.debug("Entering search")

        # times = None # hmm...

        # If it's a single value, use it for all data sources
        # !TODO consider implementing as a list (probably not needed)
        metadata_list = [
            # !TODO write this in a better way to not have to convert from astropy time to datetime is bad.
            data_source.fetch_metadata(
                self.start_date.to_datetime(), self.end_date.to_datetime(), batch_frequency=batch_frequency
            )
            for data_source in self._mag_objects
        ]

        for meta in metadata_list:
            logger.debug(f"{meta.__class__.__name__}: \n{meta[['T_REC','T_OBS','DATE-OBS','datetime', 'url']]}")

        results = []
        for meta, query in zip(metadata_list, self._query_objects):
            # do the join in pandas, and then convert to QTable?

            # removing url, but appending Query(...) without url column will complain
            # probably a better way to deal with this
            pd_query = query.to_pandas()  # [['target_time']]

            # check this dropping... how is datetime determined? are we dropping all missing?
            meta_datetime = meta[["datetime"]].drop_duplicates().dropna()

            # generate a mapping from target_time to datetime with the specified tolerance.
            # probably want a test that this whole code gets the same thing if there are no duplicates...
            merged_time = pd.merge_asof(
                left=pd_query[["target_time"]],
                right=meta_datetime,
                left_on=["target_time"],
                right_on=["datetime"],
                # suffixes=["_query", "_meta"],
                tolerance=merge_tolerance,  # HMI is at 720s (12 min) cadence
                direction="nearest",
            )

            merged_time = merged_time.dropna()  # can be NaT in the datetime column

            logger.debug(
                f"len(meta) {len(meta)}; len(meta_datetime), {len(meta_datetime)}; len(merged_time), {len(merged_time)}"
            )

            # for now, ensure that there are no duplicates of the same "datetime" in the df
            # this would happen if two `target_time` share a single `meta[datetime]`
            if len(merged_time["datetime"].dropna().unique()) != len(merged_time["datetime"].dropna()):
                raise ValueError("there are duplicates of datetime from the right df")

            # extract the rows in the metadata which match the exact datetime
            # which there may be multiple for cutouts at the same full-disk time, and join
            matched_rows = meta[meta["datetime"].isin(merged_time["datetime"])]

            # -- Bit hacky to stop H(T)ARPNUM becoming a float
            #    I think Shane may have found a better way to deal with this?
            # Convert int64 columns to Int64
            int64_columns = matched_rows.select_dtypes(include=["int64"]).columns
            # Create a new DataFrame with Int64 data types
            new_df = matched_rows.copy()
            for col in int64_columns:
                new_df[col] = matched_rows[col].astype("Int64")

            logger.debug(f"len(merged_time) {len(merged_time)}; len(new_df), {len(new_df)}")

            # merged_time <- this is the times that match between the query and output
            # new_df / matched_rows are the rows in the output at the same time as the query
            merged_df = pd.merge(merged_time, new_df, on="datetime", how="left")
            # I hope this isn't nonsense, and keeps the `urls` as a masked column
            # how does this work with sharps/smarps where same datetime for multiple rows

            # now merge with original query (only target_time)
            if len(pd_query.url.dropna().unique()) == 0:
                merged_df = pd.merge(pd_query["target_time"], merged_df, on="target_time", how="left")
            else:
                raise NotImplementedError("pd_query.url is not empty")

            logger.debug(f"len(merged_df) {len(merged_df)}")

            # !TODO Replace NaN values in the "url" column with masked values or change this...
            # remove columns ?
            # rename columns ?
            results.append(Query(QTable.from_pandas(merged_df)))

        logger.debug("Exiting search")
        return results

    def download(self, query_list: list[Query], path=None, overwrite=False, retry_missing=False):
        logger.debug("Entering download")

        downloads = []
        for query in query_list:
            # expand like swpc

            new_query = None
            results = query.copy()

            if overwrite is True or "path" not in query.colnames:
                logger.debug(f"Full download with overwrite: (overwrite = {overwrite})")
                new_query = QTable(query)

            if retry_missing is True:
                logger.debug(f"Downloading with retry_missing: (retry_missing = {retry_missing})")
                missing = query["path"] == ""
                new_query = query[missing]

            if new_query is not None:
                logger.debug("Downloading ...")
                downloads = self._download(new_query[~new_query["url"].mask]["url"].data.data, overwrite, path)
                results = self._match(results, downloads.data)  # should return a results object.

            downloads.append(Result(results))

        logger.debug("Exiting download")
        return downloads

    def _match(self, results: Query, downloads: np.array) -> Result:  # maybe?
        """ """
        logger.info("Downloads to query or new data")
        results = QTable(results)

        if "path" in results.colnames:
            results.remove_column("path")

        # make empty path column
        results["path"] = None

        results_df = QTable.to_pandas(results)
        results_df["temp_url_name"] = [Path(url).name if not pd.isna(url) else "" for url in results_df["url"]]
        downloads_df = pd.DataFrame({"temp_path": downloads})
        downloads_df["temp_path_name"] = downloads_df["temp_path"].apply(lambda x: Path(x).name)
        merged_df = pd.merge(results_df, downloads_df, left_on="temp_url_name", right_on="temp_path_name", how="left")

        merged_df.drop(columns=["temp_url_name", "temp_path_name"], inplace=True)

        results = QTable.from_pandas(merged_df)
        results["path"][results["path"] is None] = ""  # for masking
        # Table weirdness
        tmp_path = MaskedColumn(results["temp_path"].data.tolist())
        tmp_path.mask = results["url"].mask
        results.replace_column("path", tmp_path)
        results.remove_column("temp_path")
        results = Result(results.columns)
        return results

    def _download(
        column,  # is this urls or a table?
        path: Path,
        max_retries=5,
    ):
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
        UnifiedResponse
            The download results response
        """
        downloader = Downloader(
            max_conn=1,
            progress=True,
            overwrite=False,
            max_splits=1,
        )

        for url in column:
            if url != "":  # remove?
                # download only if it doesn't exist or overwrite is True
                # this assumes that parfive deals with checking the integrity of files downloaded
                # and that none are corrupt

                # check if exists before adding... (but then want to return the path of it existing...)
                downloader.enqueue_file(url=url, path=path)

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

        return results

    @staticmethod
    def download_from_df_column(
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
        if (
            urls_df is None
            or not isinstance(urls_df, pd.DataFrame)
            or column_name not in urls_df.columns
            or base_directory_path
        ):
            logger.warning(f"Invalid DataFrame format. Expected a DataFrame with a '{column_name}' column.")
            return None
        if base_directory_path is None:
            logger.warning("Provide a base_directory_path.")
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
