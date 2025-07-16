import logging
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
from pandas import DataFrame
from sunpy.util.parfive_helpers import Downloader

from astropy.table import MaskedColumn, QTable
from astropy.time import Time

from arccnet.data_generation.magnetograms.base_magnetogram import BaseMagnetogram
from arccnet.utils.logging import get_logger

# from arccnet.data_generation.utils.data_logger import logger # move to get_logger

logger = get_logger(__name__, logging.DEBUG)

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

    required_column_types = {"target_time": Time, "url": str}  # column types are not currently enforced

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not set(self.colnames).issuperset(set(self.required_column_types.keys())):
            raise ValueError(
                f"{self.__class__.__name__} must contain {list(self.required_column_types.keys())} columns."
            )

    @property
    def is_empty(self) -> bool:
        r"""Is the query empty."""
        return np.all(self["url"].mask == np.full(len(self), True))

    @property
    def missing(self):
        r"""Rows which are missing."""
        return self[self["url"].mask == True]  # noqa

    @classmethod
    def create_empty(cls, start, end, frequency: timedelta):
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


# could just import the Result object from SRS and adjust required_column_types, but this is more explicit
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
                f"{self.__class__.__name__} must contain {list(self.required_column_types.keys())} columns"
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
        start_date : `str`
            Start date of data acquisition period.

        end_date : `str`
            End date of data acquisition period.

        frequency : `timedelta`
            Observation frequency

        magnetograms : `list(BaseMagnetogram)`
            List of classes derived from BaseMagnetogram.
        """
        self._start_date = Time(start_date)
        self._end_date = Time(end_date)
        self._frequency = frequency

        # Check that all class objects are subclasses of `BaseMagnetogram`
        for class_obj in magnetograms:
            if not issubclass(class_obj.__class__, BaseMagnetogram):
                raise ValueError(f"{class_obj.__name__} is not a subclass of BaseMagnetogram")

        self._mag_objects = magnetograms

        # Generate empty query for each provided magnetogram object
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
    def frequency(self):
        return self._frequency

    @property
    def mag_objects(self):
        return self._mag_objects

    @property
    def query_objects(self):
        return self._query_objects

    def search(self, batch_frequency: int, merge_tolerance: timedelta) -> list[Query]:
        """
        Fetch and return data from various sources.

        Parameters
        ----------
        batch_frequency : `int`
            integer number of months to batch search.

        merge_tolerance : `timedelta`
            the tolerance on observation time to target target time.

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
            data_source.fetch_metadata(
                self.start_date.to_datetime(), self.end_date.to_datetime(), batch_frequency=batch_frequency
            )  # understand if that conversion from astropy time to datetime is bad.
            for data_source in self._mag_objects
        ]

        for meta in metadata_list:
            logger.debug(f"{meta.__class__.__name__}: \n{meta[['T_REC', 'T_OBS', 'DATE-OBS', 'datetime', 'url']]}")

        results = []
        for meta, query in zip(metadata_list, self._query_objects):
            # do the join in pandas, and then convert to QTable?

            # removing url, but appending Query(...) without url column will complain
            # probably a better way to deal with this
            pd_query = query.to_pandas()  # [['target_time']]

            # check this dropping... how is datetime determined? are we dropping all missing?
            meta_datetime = (
                meta[["datetime"]].drop_duplicates().dropna().sort_values("datetime").reset_index(drop=True)
            )  # adding sorting here... is this going to mess something up?
            if len(meta_datetime) == 0:
                results.append(
                    Query(
                        QTable(
                            names=[
                                "target_time",
                                "datetime",
                                "start_time",
                                "end_time",
                                "record",
                                "filename",
                                "url",
                                "record_T_REC",
                            ]
                        )
                    )
                )
                continue
            # generate a mapping between target_time to datetime with the specified tolerance.
            merged_time = pd.merge_asof(
                left=pd_query[["target_time"]],
                right=meta_datetime,
                left_on=["target_time"],
                right_on=["datetime"],
                tolerance=merge_tolerance,
                direction="nearest",
            )

            merged_time = merged_time.dropna()  # can be NaT in the datetime column

            # for now, ensure that there are no duplicates of the same "datetime" in the df
            # this would happen if two `target_time` share a single `meta[datetime]`
            if len(merged_time["datetime"].dropna().unique()) != len(merged_time["datetime"].dropna()):
                raise ValueError("there are duplicates of datetime from the right df")

            # extract the rows in the metadata which match the exact datetime
            # which there may be multiple for cutouts at the same full-disk time, and join
            matched_rows = meta[meta["datetime"].isin(merged_time["datetime"])]

            # merged_time <- this is the times that match between the query and output
            # matched_rows are the rows in the output at the same time as the query
            merged_df = pd.merge(merged_time, matched_rows, on="datetime", how="left")
            # I hope this isn't nonsense, and keeps the `urls` as a masked column
            # how does this work with sharps/smarps where same datetime for multiple rows

            # now merge with original query (only target_time)
            if len(pd_query.url.dropna().unique()) == 0:
                merged_df = pd.merge(pd_query["target_time"], merged_df, on="target_time", how="left")
            else:
                raise NotImplementedError("pd_query.url is not empty")

            # !TODO Replace NaN values in the "url" column with masked values or change this...
            # remove columns ?
            # rename columns ?
            results.append(Query(QTable.from_pandas(merged_df)))

        logger.debug("Exiting search")
        return metadata_list, results

    def download(self, query_list: list[Query], path: Path = None, overwrite: bool = False):
        """

        Parameters
        ----------
        query_list : `list[Query]`
            list of Query(QTable) objects

        path : `Path`
            download path

        overwrite : `bool`, optional
            overwrite files on download. Default is False

        Returns
        -------
        downloads : list[Result]
            list of Result objects
        """
        logger.debug("Entering download")

        downloads = []
        for query in query_list:
            # expand like swpc

            new_query = None
            results = query.copy()

            if overwrite is True or "path" not in query.colnames:
                logger.debug(f"Full download (overwrite = {overwrite})")
                new_query = QTable(query)

            # !TODO a way of retrying missing would be good, but JSOC URLs are temporary.
            if new_query is not None:
                if len(new_query) > 0:
                    downloaded_files = self._download(
                        data_list=new_query[~new_query.mask["url"]]["url"].data, path=path, overwrite=overwrite
                    )
                else:
                    downloaded_files = []
                results = self._match(results, downloaded_files)  # should return a results object.
            else:
                raise NotImplementedError("new_query is none.")

            downloads.append(Result(results))

        logger.debug("Exiting download")
        return downloads

    def _match(self, results: Result, downloads: np.array) -> Result:  # maybe?
        """
        match results against downloaded files

        Parameters
        ----------
        results : `Result`
            Result(QTable)

        downloads : `np.array`
            downloaded filenames

        """
        results = QTable(results)

        if "path" in results.colnames:
            results.remove_column("path")

        results_df = QTable.to_pandas(results)
        results_df["temp_url_name"] = [str(Path(url).name) if not pd.isna(url) else "" for url in results_df["url"]]
        downloads_df = DataFrame({"path": downloads})
        downloads_df["temp_path_name"] = downloads_df["path"].apply(lambda x: str(Path(x).name))
        merged_df = pd.merge(results_df, downloads_df, left_on="temp_url_name", right_on="temp_path_name", how="left")

        merged_df.drop(columns=["temp_url_name", "temp_path_name"], inplace=True)
        results = QTable.from_pandas(merged_df)
        return results

    @staticmethod
    def _download(
        data_list,
        path: Path,
        overwrite: bool = False,
        max_retries: int = 5,
    ):
        """
        Download data from URLs

        Parameters
        ----------
        data_list
            list of URLs to download

        path : `Path`
            Path to save downloaded files.

        overwrite : `bool`, optional
            Flag to overwrite files. Default is False

        max_retries : `int`, optional
            Maximum number of download retries. Default is 5.

        Returns
        -------
        'parfive.results.Results'
        """
        downloader = Downloader(
            max_conn=1,
            progress=True,
            overwrite=overwrite,
            max_splits=1,
        )

        existing_files = []
        # if file exists, add to list.
        for url in data_list:
            # Generate the full path where you want to save the file
            file_name = Path(url).name
            file_path = Path(path) / file_name

            # Check if the file already exists
            if file_path.exists() and not overwrite:
                existing_files.append(str(file_path))
            else:
                # If it doesn't exist, enqueue the file for downloading
                downloader.enqueue_file(url=url, path=path)

        logger.debug(f"{len(existing_files)}/{len(data_list)} files already exist (overwrite = {overwrite})")
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

        results = np.concatenate((existing_files, results.data), axis=0)
        results = np.sort(results, axis=0)

        return results
