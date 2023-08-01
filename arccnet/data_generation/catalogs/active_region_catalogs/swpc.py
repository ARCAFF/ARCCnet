import pathlib
import datetime
from typing import Union, Optional
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import parfive
import parfive.results

import arccnet.data_generation.utils.default_variables as dv
from arccnet.data_generation.catalogs.base_catalog import BaseCatalog
from arccnet.data_generation.utils.data_logger import logger
from arccnet.data_generation.utils.utils import check_column_values, save_df_to_html
from sunpy.io.special import srs
from sunpy.net import Fido
from sunpy.net import attrs as a

__all__ = ["SWPCCatalog", "NoDataError"]


class SWPCCatalog(BaseCatalog):
    """
    SWPCCatalog is a class for fetching and processing SWPC active region
    classification data.

    It provides methods to fetch data from SWPC, create a catalog from the
    fetched data, clean and validate the catalog, and perform other related
    operations.

    Attributes
    ----------
    text_format_template : None or `pandas.Series`
        The template defining the expected data types for each column in the
        catalog. Initially set to None and later populated based on the
        fetched data.

    _fetched_data : None or `parfive.results.Results`
        The fetched SWPC data, set by calling `fetch_data()`. Initially set to None.

    raw_catalog : None or `pandas.DataFrame`
        The raw catalog created from the fetched data, set by calling `create_catalog()`. Initially set to None.

    raw_catalog_missing : None or `pandas.DataFrame`
        The subset of the raw catalog that contains data files that were not
        loaded successfully, set by calling `create_catalog()`. Initially set to None.

    catalog : None or `pandas.DataFrame`
        The cleaned catalog without NaN values and checked for valid values,
        set by calling `clean_catalog()`. Initially set to None.

    Methods
    -------
    fetch_data(start_date=DATA_START_TIME, end_date=DATA_END_TIME)
        Fetches SWPC active region classification data for the specified time
        range.

    create_catalog(save_csv=True, save_html=True)
        Creates an SRS catalog from the fetched data.

    clean_catalog()
        Cleans and checks the validity of the SWPC active region classification
        data.
    """

    def __init__(self):
        self.text_format_template = None
        self._fetched_data = None
        self.raw_catalog = None
        self.raw_catalog_missing = None
        self.catalog = None

    def fetch_data(
        self,
        start_date: datetime.datetime = dv.DATA_START_TIME,
        end_date: datetime.datetime = dv.DATA_END_TIME,
    ) -> parfive.Results:
        """
        Fetches SWPC active region classification data
        for the specified time range.

        Parameters
        ----------
        start_date : `datetime.datetime`
            Start date for the data range.

        end_date : `datetime.datetime`
            End date for the data range.

        Returns
        -------
        `parfive.results.Results`
            DataFrame containing SWPC active region
            classification data for the specified time range.

        Raises
        ------
        NoDataError
            If the table returned by `Fido.fetch` is of length zero.

        """

        logger.info(f">> searching for SRS data between {start_date} and {end_date}")
        result = Fido.search(
            a.Time(start_date, end_date),
            a.Instrument.soon,
        )

        logger.info(f">> downloading SRS data to {dv.NOAA_SRS_TEXT_DIR}")

        # Sunpy issue throwing up errors during SRS download so monkey patch problematic method
        orig_get_ftp = parfive.downloader.get_ftp_size

        async def dummy_get_ftp(*args, **kwargs):
            return 0

        parfive.downloader.get_ftp_size = dummy_get_ftp

        table = Fido.fetch(
            result,
            path=dv.NOAA_SRS_TEXT_DIR,
            progress=True,
            overwrite=False,
        )

        # Replace original method - not sure if this is needed
        parfive.downloader.get_ftp_size = orig_get_ftp

        if len(table.errors) > 0:
            logger.warning(f">> the following errors were reported: {table.errors}")
            # !TODO re-run?
        else:
            logger.info(">> no errors reported in `fido.fetch`")

        if len(table) == 0:
            raise NoDataError

        # set _fetched_data, and return it
        self._fetched_data = table
        return table

    @staticmethod
    def _parse_srs(filepath: list[pathlib.Path]) -> tuple[pathlib.Path, Union[bool, pd.DataFrame]]:
        try:
            srs_table = srs.read_srs(filepath)
            result = srs_table.to_pandas()
        except Exception as e:
            logger.warning(f"Error reading file {str(filepath)}: {str(e)[0:65]}...")  # 0:65 truncates the error
            result = False

        return filepath, result

    def create_catalog(
        self,
        save_csv: Optional[bool] = True,
        save_html: Optional[bool] = True,
    ) -> "tuple[pd.DataFrame, pd.DataFrame]":
        """
        Creates an SRS catalog from `self._fetched_data`.

        Parameters
        ----------
        save_csv : Optional[bool], default=True
            Boolean for saving to CSV.

        save_html : Optional[bool], default=True
            Boolean for saving to HTML.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            Tuple containing the raw catalog, and the catalog with missing values.

        Raises
        ------
        NoDataError
            If `self._fetched_data` is `None`.

        """
        srs_dfs = []
        time_now = datetime.datetime.utcnow()

        if self._fetched_data is None:
            raise NoDataError

        logger.info(">> loading fetched data")
        # include filepaths to ignore
        downloaded_filepaths = [Path(filepath) for filepath in self._fetched_data]
        downloaded_filepaths = [
            path_obj for path_obj in downloaded_filepaths if path_obj.name not in dv.SRS_FILEPATHS_IGNORED
        ]

        with ProcessPoolExecutor() as exec:
            results = exec.map(self._parse_srs, downloaded_filepaths)

        for filepath, data in results:
            # instantiate a `pandas.DataFrame` based on our additional info
            # and assign to the SRS `pandas.DataFrame` if not empty.
            # Any issue reading will log the exception as a warning and move
            # the files to a separate directory, and flag them in the catalog.
            file_info_df = pd.DataFrame(
                [
                    {
                        "filepath": str(filepath),
                        "filename": filepath.name,
                        "loaded_successfully": False,
                        "catalog_created_on": time_now,
                    }
                ]
            )

            if data is not False:
                srs_table = srs.read_srs(filepath)
                srs_df = srs_table.to_pandas()
                file_info_df["loaded_successfully"] = True

                if self.text_format_template is None:
                    # Setting the format_template
                    cols = srs_df.select_dtypes(include="int64").columns
                    srs_df[cols] = srs_df[cols].astype("Int64")
                    # self.text_format_template = srs_df.dtypes.replace(
                    #     "int64", "Int64"
                    # )
                    # columns of dtype `int64` are replaced with dtype `Int64`
                    # (former doesn't support NaN values;
                    # https://pandas.pydata.org/docs/user_guide/integer_na.html)
                    # By default the `Number` column from `srs.read_srs` was
                    # being loaded as `int64` not `Int64`
                    # (`Sunspot Number` is `Int64` by default).
                    cols = srs_df.select_dtypes(include="int32").columns
                    srs_df[cols] = srs_df[cols].astype("Int32")
                    self.text_format_template = srs_df.dtypes
                    logger.info(f"SRS format: \n{self.text_format_template}")

                if srs_df.empty:
                    srs_dfs.append(file_info_df)
                else:
                    srs_dfs.append(srs_df.assign(**file_info_df.iloc[0]))

            elif data is False:
                # create the "except directory/folder" if it does not exist
                directory_path = Path(dv.NOAA_SRS_TEXT_EXCEPT_DIR)
                if not directory_path.exists():
                    directory_path.mkdir(parents=True)

                # Move the file to the "except directory"
                except_filepath = directory_path / filepath.name
                filepath.replace(except_filepath)
                file_info_df["filepath"] = str(except_filepath)

                srs_dfs.append(file_info_df)

        self.raw_catalog = pd.concat(srs_dfs, ignore_index=True)

        logger.info(f">> finished loading the `self._fetched_data`, of length {len(self._fetched_data)}")

        # reformat `pandas.DataFrame` based on `format_template`
        if self.text_format_template is not None:
            self.raw_catalog = self.raw_catalog.astype(self.text_format_template.to_dict())

        #!TODO move to separate method & use default variables
        self.raw_catalog["datetime"] = [
            datetime.datetime.strptime(filename.replace("SRS.txt", ""), "%Y%m%d").replace(hour=0, minute=0, second=0)
            for filename in self.raw_catalog["filename"]  # SRS valid at 00:00:00
        ]

        # extract subset of data that wasn't loaded successfully
        self.raw_catalog_missing = self.raw_catalog[~self.raw_catalog["loaded_successfully"]]

        logger.warning(
            f">> unsuccessful loading of {len(self.raw_catalog_missing ['filename'].unique())} (of {len(self.raw_catalog['filename'].unique())}) files"
        )

        # save to csv
        if save_csv:
            logger.info(f">> saving raw data to `{dv.NOAA_SRS_RAW_DATA_CSV}` and `{dv.NOAA_SRS_RAW_DATA_EXCEPT_CSV}`")
            self.raw_catalog.to_csv(dv.NOAA_SRS_RAW_DATA_CSV)
            self.raw_catalog_missing.to_csv(dv.NOAA_SRS_RAW_DATA_EXCEPT_CSV)

        # save to html
        if save_html:
            logger.info(f">> saving raw data to `{dv.NOAA_SRS_RAW_DATA_HTML}` and `{dv.NOAA_SRS_RAW_DATA_EXCEPT_HTML}`")
            save_df_to_html(df=self.raw_catalog, filename=dv.NOAA_SRS_RAW_DATA_HTML)
            save_df_to_html(df=self.raw_catalog_missing, filename=dv.NOAA_SRS_RAW_DATA_EXCEPT_HTML)

        return self.raw_catalog, self.raw_catalog_missing

    def clean_catalog(self, save_csv: Optional[bool] = True) -> pd.DataFrame:
        """
        Cleans and checks the validity of the SWPC active region classification data

        Parameters
        ----------
        save_csv : Optional[bool], default=True
            Boolean for saving to CSV.

        Returns
        -------
        `pandas.DataFrame`
            Cleaned catalog without NaNs, checked for valid values

        Raises
        ------
        NoDataError
            If no SWPC data is found. Call `fetch_data()` first to obtain the data

        """
        if self.raw_catalog is not None:
            # Drop rows with NaNs to remove `loaded_successfully` == False
            # Check columns against `VALID_SRS_VALUES`
            self.catalog = self.raw_catalog.dropna().reset_index(drop=True)

            # Fixing the case of Mag Type and Z for capitalised SRS files
            self.catalog["Mag Type"] = self.catalog["Mag Type"].str.title()
            self.catalog["Z"] = self.catalog["Z"].str.title()

            _ = check_column_values(
                catalog=self.catalog,
                valid_values=dv.VALID_SRS_VALUES,
            )
        else:
            raise NoDataError("No SWPC data found. Please call `fetch_data()` first to obtain the data.")

        # save to an intermediate location
        if save_csv:
            Path(dv.NOAA_SRS_INTERMEDIATE_DIR).mkdir(parents=True, exist_ok=True)
            logger.info(f">> saving cleaned catalog `{dv.NOAA_SRS_INTERMEDIATE_DATA_CSV}`")
            self.catalog.to_csv(dv.NOAA_SRS_INTERMEDIATE_DATA_CSV)

        if len(self.catalog) == 0:
            raise NoDataError("Cleaned SWPC Catalog is empty")

        return self.catalog


class NoDataError(Exception):
    """
    Raises an Exception when no data is available
    """

    def __init__(self, message="No data available."):
        super().__init__(message)
        logger.exception(message)
