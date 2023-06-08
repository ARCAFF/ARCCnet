from datetime import datetime

import pandas as pd

import arccnet.data_generation.utils.default_variables as dv
from arccnet.data_generation.catalogs.active_region_catalogs.swpc import SWPCCatalog
from arccnet.data_generation.magnetograms.instruments import HMIMagnetogram, MDIMagnetogram
from arccnet.data_generation.utils.data_logger import logger

__all__ = ["DataManager"]


class DataManager:
    """
    Main data management class.

    This class instantiates and handles data acquisition for the individual instruments
    """

    def __init__(
        self,
        start_date: datetime = dv.DATA_START_TIME,
        end_date: datetime = dv.DATA_END_TIME,
    ):
        self.start_date = start_date
        self.end_date = end_date

        logger.info(f"Instantiated `DataManager` for {self.start_date} -> {self.end_date}")

        # instantiate classes
        self.swpc = SWPCCatalog()
        self.hmi = HMIMagnetogram()
        self.mdi = MDIMagnetogram()

        # 1. fetch metadata
        logger.info(">> Fetching Metadata")
        self.fetch_metadata()
        logger.info(f"\n{self.srs_raw}")

        # 2. clean metadata
        logger.info(">> Cleaning Metadata")
        self.clean_metadata()
        logger.info(f"\n{self.srs_clean}")

        # # 3. merge metadata sources
        # tol = pd.Timedelta("30m")
        # logger.info(f">> Merging Metadata with tolerance {tol}")
        # self.merged_data = self.merge_metadata_sources(tolerance=tol)

        # 4a. check if image data exists
        # ...

        # 4b. download image data
        # ...

        logger.info(">> Execution completed successfully")

    def fetch_metadata(self):
        """
        method to fetch and return data from various sources
        """

        # download the txt files and create an SRS catalog
        _ = self.swpc.fetch_data(self.start_date, self.end_date)
        self.srs_raw, self.srs_raw_missing = self.swpc.create_catalog()

        # HMI & MDI
        # self.hmi_k, self.hmi_urls = self.hmi.fetch_metadata(self.start_date, self.end_date)
        self.hmi_k = self.hmi.fetch_metadata(self.start_date, self.end_date)
        # logger.info(f"HMI Keys: \n{self.hmi_k}")
        logger.info(
            f"HMI Keys: \n{self.hmi_k[['T_REC','T_OBS','DATE-OBS','DATE__OBS','datetime','magnetogram_fits']]}"
        )  # the date-obs or date-avg
        self.mdi_k = self.mdi.fetch_metadata(self.start_date, self.end_date)
        # logger.info(f"MDI Keys: \n{self.mdi_k}")
        logger.info(
            f"MDI Keys: \n{self.mdi_k[['T_REC','T_OBS','DATE-OBS','DATE__OBS','datetime','magnetogram_fits']]}"
        )  # the date-obs or date-avg

    def clean_metadata(self):
        """
        clean data from each instrument
        """

        # clean the raw SRS catalog
        self.srs_clean = self.swpc.clean_catalog()

        # clean the raw HMI/MDI catalogs
        # ...

    def merge_metadata_sources(
        self,
        tolerance: pd.Timedelta = pd.Timedelta("30m"),
    ):
        """
        method to merge the data sources
        """

        # merge srs_clean and hmi

        # !TODO do a check for certain keys (no duplicates...)
        # extract only the relevant HMI keys, and rename
        # (should probably do this earlier on)
        hmi_keys = self.hmi_k[["magnetogram_fits", "datetime"]]
        hmi_keys = hmi_keys.rename(
            columns={
                "datetime": "datetime_hmi",
                "magnetogram_fits": "magnetogram_fits_hmi",
            }
        )
        hmi_keys_dropna = hmi_keys.dropna().reset_index(drop=True)

        # both `pd.DataFrame` must be sorted based on the key !
        self.merged_df = pd.merge_asof(
            left=self.srs_clean.rename(
                columns={
                    "datetime": "datetime_srs",
                    "filepath": "filepath_srs",
                    "filename": "filename_srs",
                    "loaded_successfully": "loaded_successfully_srs",
                    "catalog_created_on": "catalog_created_on_srs",
                }
            ),
            right=hmi_keys_dropna,
            left_on="datetime_srs",
            right_on="datetime_hmi",
            suffixes=["_srs", "_hmi"],
            tolerance=tolerance,  # HMI is at 720s (12 min) cadence
            direction="nearest",
        )

        mdi_keys = self.mdi_k[["magnetogram_fits", "datetime"]]
        hmi_keys = hmi_keys.rename(
            columns={
                "datetime": "datetime_mdi",
                "magnetogram_fits": "magnetogram_fits_mdi",
            }
        )
        self.merged_df = pd.merge_asof(
            left=self.merged_df,
            right=mdi_keys,
            left_on="datetime_srs",
            right_on="datetime_mdi",
            suffixes=["_srs2", "_mdi"],
            tolerance=tolerance,  # HMI is at 720s (12 min) cadence
            direction="nearest",
        )

        # do we want to wait until we merge with MDI before dropping nans?
        self.dropped_rows = self.merged_df.copy()
        self.merged_df = self.merged_df.dropna(subset=["datetime_srs", "datetime_hmi", "datetime_mdi"])
        self.dropped_rows = self.dropped_rows[~self.dropped_rows.index.isin(self.merged_df.index)].copy()

        logger.info(f"merged_df: \n{self.merged_df.head()}")
        logger.info(f"dropped_rows: \n{self.hmi_dropped_rows}")


if __name__ == "__main__":
    logger.info(f"Executing {__file__} as main program")
    _ = DataManager(dv.DATA_START_TIME, dv.DATA_END_TIME)
