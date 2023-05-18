import os
from datetime import datetime

import pandas as pd

import arccnet.data_generation.utils.default_variables as dv
from arccnet.data_generation.catalogs.base_catalog import BaseCatalog
from arccnet.data_generation.utils.data_logger import logger
from sunpy.io.special import srs
from sunpy.net import Fido
from sunpy.net import attrs as a

__all__ = ["SWPCCatalog"]


class SWPCCatalog(BaseCatalog):
    def __init__(self):
        # -- setting the format template for SWPCC data to be None
        self.text_format_template = None

        self._fetched_data = None

        self.raw_catalog = None
        self.raw_catalog_missing = None

        self.catalog = None

    def fetch_data(
        self,
        start_date: datetime = dv.DATA_START_TIME,
        end_date: datetime = dv.DATA_END_TIME,
    ) -> pd.DataFrame:
        """
        Fetches SWPC active region classification data
        for the specified time range.

        Parameters
        ----------
            start_date (datetime): Start date for the data range.
            end_date (datetime): End date for the data range.

        Returns
        -------
            `pandas.DataFrame`: DataFrame containing SWPC active region
            classification data for the specified time range.
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

        self._fetched_data = table
        return table

    def create_catalog(
        self,
        save_html: bool = True,
    ) -> pd.DataFrame:
        """
        Creates an SRS catalog from `self._fetched_data`

        Parameters
        ----------
            save_html (bool): Boolean for saving to HTML. Default is True.

        Returns
        -------
            None

        """
        srs_dfs = []
        time_now = datetime.utcnow()

        if self._fetched_data is None:
            raise NoDataError()

        logger.info(">> loading fetched data")
        for filepath in self._fetched_data:
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

                    cols = srs_df.select_dtypes(include="int").columns
                    srs_df[cols] = srs_df[cols].astype("Int64")
                    self.text_format_template = srs_df.dtypes
                    # self.text_format_template = srs_df.dtypes.replace(
                    #     "int64", "Int64"
                    # )
                    # columns of dtype `int64` are replaced with dtype `Int64`
                    # (former doesn't support NaN values;
                    # https://pandas.pydata.org/docs/user_guide/integer_na.html)
                    # By default the `Number` column from `srs.read_srs` was
                    # being loaded as `int64` not `Int64`
                    # (`Sunspot Number` is `Int64` by default).
                    logger.info(f"SRS format: \n{self.text_format_template}")

                if srs_df.empty:
                    srs_dfs.append(file_info_df)
                else:
                    srs_dfs.append(srs_df.assign(**file_info_df.iloc[0]))

            except Exception as e:
                logger.warning(f"Error reading file {filepath}: {str(e)[0:65]}...")  # 0:65 truncates the error

                # create the folder if it doesn't exist
                if not os.path.exists(dv.NOAA_SRS_TEXT_EXCEPT_DIR):
                    os.makedirs(dv.NOAA_SRS_TEXT_EXCEPT_DIR)

                # Move the file to the `except` folder
                except_filepath = os.path.join(dv.NOAA_SRS_TEXT_EXCEPT_DIR, os.path.basename(filepath))
                os.rename(filepath, except_filepath)
                file_info_df["filepath"] = except_filepath

                srs_dfs.append(file_info_df)

        df = pd.concat(srs_dfs, ignore_index=True)

        logger.info(f">> finished loading the `self._fetched_data`, of length {len(self._fetched_data)}")

        # reformat based on format_template
        if self.text_format_template is not None:
            df = df.astype(self.text_format_template.to_dict())

        # extract subset of data that wasn't loaded successfully
        srs_unable_to_load = df[~df["loaded_successfully"]]

        logger.warning(
            f">> unsuccessful loading of {len(df[~df['loaded_successfully']]['filename'].unique())} (of {len(df['filename'].unique())}) files"
        )

        # save the dataframe with all data, and a dataframe with missing data
        logger.info(f">> saving raw data to `{dv.NOAA_SRS_RAW_DATA_CSV}` " + f"`{dv.NOAA_SRS_RAW_DATA_EXCEPT_CSV}`")
        df.to_csv(dv.NOAA_SRS_RAW_DATA_CSV)
        srs_unable_to_load.to_csv(dv.NOAA_SRS_RAW_DATA_EXCEPT_CSV)

        if save_html:
            logger.info(
                f">> saving raw data to `{dv.NOAA_SRS_RAW_DATA_HTML}` " + f"and `{dv.NOAA_SRS_RAW_DATA_EXCEPT_HTML}`"
            )
            # !TODO clean this up

            # write df to html
            text_file = open(dv.NOAA_SRS_RAW_DATA_HTML, "w")
            text_file.write(df.to_html())
            text_file.close()

            # write unable to load data to html
            text_file = open(dv.NOAA_SRS_RAW_DATA_EXCEPT_HTML, "w")
            text_file.write(srs_unable_to_load.to_html())
            text_file.close()

        self.raw_catalog = df
        self.raw_catalog_missing = srs_unable_to_load

        return df, srs_unable_to_load

    def clean_data(self) -> pd.DataFrame:
        """
        Cleans the SWPC active region classification data
        by dropping duplicates and sorting by date.
        """
        if self.raw_catalog is not None:
            # Drop rows with NaNs to remove `loaded_successfully` == False
            self.catalog = self.raw_catalog.dropna()

            valid_values = {
                "Mag Type": dv.HALE_CLASSES,
                "Z": dv.MCINTOSH_CLASSES,
                "ID": ["I"],  # , "IA", "II"],
            }

            # TEST THE BELOW CODE
            for col, vals in valid_values.items():
                result = self.catalog[col].isin(vals)
                invalid_vals = list(self.catalog.loc[~result, col].unique())
                if invalid_vals:
                    msg = f"Invalid `{col}`; `{col}` = {invalid_vals}"
                    logger.error(msg)
                    raise ValueError(msg)

            # ensuring that only `ID` == `I`
            assert self.catalog["ID"].unique()[0] == "I"

        else:
            raise NoDataError("No SWPC data found. " + "Please call `fetch_data()` first to obtain the data.")

        return self.catalog


class NoDataError(Exception):
    def __init__(self, message="No data available."):
        super().__init__(message)
        logger.exception(message)
