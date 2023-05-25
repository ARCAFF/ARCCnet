import os
import datetime
from typing import Optional

import pandas as pd

import arccnet.data_generation.utils.default_variables as dv
from arccnet.data_generation.catalogs.base_catalog import BaseCatalog
from arccnet.data_generation.utils.data_logger import logger
from sunpy.io.special import srs
from sunpy.net import Fido
from sunpy.net import attrs as a

__all__ = ["SWPCCatalog", "NoDataError", "save_df_to_html", "check_column_values"]
# !TODO move these to another place


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

    _fetched_data : None or `sunpy.net.fido.results.QueryResponse`
        The fetched SWPC data. Initially set to None.

    raw_catalog : None or `pandas.DataFrame`
        The raw catalog created from the fetched data. Initially set to None.

    raw_catalog_missing : None or `pandas.DataFrame`
        The subset of the raw catalog that contains data files that were not
        loaded successfully. Initially set to None.

    catalog : None or `pandas.DataFrame`
        The cleaned catalog without NaN values and checked for valid values.
        Initially set to None.

    Methods
    -------
    fetch_data(start_date=DATA_START_TIME, end_date=DATA_END_TIME)
        Fetches SWPC active region classification data for the specified time
        range.

    create_catalog(save_csv=True, save_html=True)
        Creates an SRS catalog from the fetched data.

    clean_data()
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
    ) -> pd.DataFrame:
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
        `pandas.DataFrame`
            DataFrame containing SWPC active region
            classification data for the specified time range.

        Raises
        ------
        NoDataError
            If the table returned by `Fido.fetch` is of length zero.

        Examples
        --------
        >>> start_date = datetime.datetime(2022, 1, 1)
        >>> end_date = datetime.datetime(2022, 1, 7)
        >>> data = fetch_data(start_date, end_date)
        """

        logger.info(f">> searching for SRS data between {start_date} and {end_date}")
        result = Fido.search(
            a.Time(start_date, end_date),
            a.Instrument.soon,
        )

        logger.info(f">> downloading SRS data to {dv.NOAA_SRS_TEXT_DIR}")
        table = Fido.fetch(
            result,
            path=dv.NOAA_SRS_TEXT_DIR,
            progress=True,
            overwrite=False,
        )

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

    def create_catalog(
        self,
        save_csv: Optional[bool] = True,
        save_html: Optional[bool] = True,
    ) -> pd.DataFrame:
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
        pd.DataFrame
            The raw catalog.

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
        for filepath in self._fetched_data:
            # instantiate a `pandas.DataFrame` based on our additional info
            # and assign to the SRS `pandas.DataFrame` if not empty.
            # Any issue reading will log the exception as a warning and move
            # the files to a separate directory, and flag them in the catalog.
            file_info_df = pd.DataFrame(
                [
                    {
                        "filepath": filepath,
                        "filename": os.path.basename(filepath),
                        "loaded_successfully": False,
                        "catalog_created_on": time_now,
                    }
                ]
            )

            try:
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

            except Exception as e:
                logger.warning(f"Error reading file {filepath}: {str(e)[0:65]}...")  # 0:65 truncates the error

                # create the "except directory/folder" if it does not exist
                if not os.path.exists(dv.NOAA_SRS_TEXT_EXCEPT_DIR):
                    os.makedirs(dv.NOAA_SRS_TEXT_EXCEPT_DIR)

                # Move the file to the "except directory"
                except_filepath = os.path.join(dv.NOAA_SRS_TEXT_EXCEPT_DIR, os.path.basename(filepath))
                os.rename(filepath, except_filepath)
                file_info_df["filepath"] = except_filepath

                srs_dfs.append(file_info_df)

        self.raw_catalog = pd.concat(srs_dfs, ignore_index=True)

        logger.info(f">> finished loading the `self._fetched_data`, of length {len(self._fetched_data)}")

        # reformat `pandas.DataFrame` based on `format_template`
        if self.text_format_template is not None:
            self.raw_catalog = self.raw_catalog.astype(self.text_format_template.to_dict())

        #!TODO move to separate method & use default variables
        self.raw_catalog["datetime"] = [
            datetime.datetime.strptime(filename.replace("SRS.txt", ""), "%Y%m%d").replace(hour=0, minute=30, second=0)
            for filename in self.raw_catalog["filename"]
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

    def clean_data(self) -> pd.DataFrame:
        """
        Cleans and checks the validity of the SWPC active region classification data

        Returns
        -------
        `pandas.DataFrame`
            Cleaned catalog without NaNs, checked for valid values

        Raises
        ------
        NoDataError
            If no SWPC data is found. Call `fetch_data()` first to obtain the data

        """
        if self.raw_catalog is not None:  #! TODO not raw catalog, surely?
            # Drop rows with NaNs to remove `loaded_successfully` == False
            # Check columns against `VALID_SRS_VALUES`
            check_column_values(
                catalog=self.raw_catalog.dropna(),
                valid_values=dv.VALID_SRS_VALUES,
            )
        else:
            raise NoDataError("No SWPC data found. Please call `fetch_data()` first to obtain the data.")

        return self.catalog


class NoDataError(Exception):
    """
    Raises an Exception
    """

    def __init__(self, message="No data available."):
        super().__init__(message)
        logger.exception(message)


def save_df_to_html(df: pd.DataFrame, filename: str) -> None:
    """
    Save the provided `df` to an HTML file with the specified `filename`.

    Parameters
    ----------
    df : `pandas.DataFrame`
        a `pandas.DataFrame` to save to the HTML file

    filename : str
        the HTML filename

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If `filename` is not a string or `df` is not a DataFrame

    """

    if not isinstance(filename, str):
        raise ValueError("The `filename` must be a string.")

    if not isinstance(df, pd.DataFrame):
        raise ValueError("The provided object is not a `pandas.DataFrame`.")

    with open(filename, "w") as file:
        file.write(df.to_html())


def check_column_values(catalog: pd.DataFrame, valid_values: dict) -> pd.DataFrame:
    """
    Check column values against known (valid) values.

    First check if the columns in `valid_values` are present in the
    `catalog` DataFrame and verify that the corresponding values in those
    columns match the known valid values.

    Parameters
    ----------
    catalog : pandas.DataFrame
        a `pandas.DataFrame` that contains a set of columns

    valid_values : dict
        a dictionary containing the column names and valid values.
        The dictionary keys must be a subset of the `catalog.columns`

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If any columns in `valid_values` are not present in the `catalog`.

    Examples
    --------
    >>> catalog = pandas.DataFrame({'ID': ['I', 'I', 'II'], 'Value': [10, 20, 30]})
    >>> valid_values = {'ID': ['I', 'II'], 'Value': [10, 20, 30]}
    >>> check_column_values(catalog, valid_values)

    """

    # Check that the columns in `valid_values` are in `catalog``
    invalid_columns = set(valid_values.keys()) - set(catalog.columns)
    if invalid_columns:
        raise ValueError(f"Columns {list(invalid_columns)} in `valid_values` are not present in `catalog`.")

    # Checking values against the `valid_values`
    for col, vals in valid_values.items():
        result = catalog[col].isin(vals)
        invalid_vals = catalog.loc[~result, col].unique().tolist()
        if invalid_vals:
            msg = f"Invalid `{col}`; `{col}` = {invalid_vals}"
            logger.error(msg)
            # raise ValueError(msg) # !TODO reinstate ValueError

    if catalog["ID"].nunique() != 1 or catalog["ID"].unique()[0] != "I":
        raise ValueError("Invalid 'ID' values.")

    return catalog
