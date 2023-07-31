from pathlib import Path
from datetime import datetime

import pandas as pd
from sunpy.util.parfive_helpers import Downloader

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
        merge_tolerance: pd.Timedelta = pd.Timedelta("30m"),
    ):
        self.start_date = start_date
        self.end_date = end_date

        logger.info(f"Instantiated `DataManager` for {self.start_date} -> {self.end_date}")

        # instantiate classes
        self.swpc = SWPCCatalog()
        # !TODO change this into an iterable
        self.hmi = HMIMagnetogram()
        self.mdi = MDIMagnetogram()

        # 1. fetch metadata
        logger.info(">> Fetching NOAA SRS Metadata")
        self.fetch_metadata()
        logger.info(f"\n{self.srs_raw}")

        # 2. clean metadata
        logger.info(">> Cleaning NOAA SRS Metadata")
        # self.clean_metadata()
        self.srs_clean = self.swpc.clean_catalog()
        logger.info(f"\n{self.srs_clean}")

        # 3. merge metadata sources
        # logger.info(f">> Merging Metadata with tolerance {merge_tolerance}")
        self.merge_metadata_sources(tolerance=merge_tolerance)

        # 4a. check if image data exists
        # !TODO implement this checking if each file that is expected exists.

        # # 4b. download image data
        _ = self.fetch_magnetograms(self.merged_df)
        # !TODO handle the output... want a csv with the filepaths

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
            f"HMI Keys: \n{self.hmi_k[['T_REC','T_OBS','DATE-OBS','DATE__OBS','datetime','magnetogram_fits', 'url']]}"
        )  # the date-obs or date-avg
        self.mdi_k = self.mdi.fetch_metadata(self.start_date, self.end_date)
        # logger.info(f"MDI Keys: \n{self.mdi_k}")
        logger.info(
            f"MDI Keys: \n{self.mdi_k[['T_REC','T_OBS','DATE-OBS','DATE__OBS','datetime','magnetogram_fits', 'url']]}"
        )  # the date-obs or date-avg

    # def clean_metadata(self):
    #     """
    #     clean metadata from each instrument
    #     """
    #
    #     # clean the raw SRS catalog
    #     self.srs_clean = self.swpc.clean_catalog()

    def merge_metadata_sources(
        self,
        tolerance: pd.Timedelta = pd.Timedelta("30m"),
    ):
        """
        method to merge the data sources
        """

        # merge srs_clean and hmi
        mag_cols = ["magnetogram_fits", "datetime", "url"]

        # !TODO do a check for certain keys (no duplicates...)
        # extract only the relevant HMI keys, and rename
        # (should probably do this earlier on)
        hmi_keys = self.hmi_k[mag_cols]
        hmi_keys = hmi_keys.add_suffix("_hmi")
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

        mdi_keys = self.mdi_k[mag_cols]
        mdi_keys = mdi_keys.add_suffix("_mdi")
        mdi_keys_dropna = mdi_keys.dropna().reset_index(drop=True)

        self.merged_df = pd.merge_asof(
            left=self.merged_df,
            right=mdi_keys_dropna,
            left_on="datetime_srs",
            right_on="datetime_mdi",
            suffixes=["_srs", "_mdi"],
            tolerance=tolerance,
            direction="nearest",
        )

        # do we want to wait until we merge with MDI before dropping nans?
        self.dropped_rows = self.merged_df.copy()
        # self.merged_df = self.merged_df.dropna(subset=["datetime_srs", "datetime_hmi", "datetime_mdi"])
        self.merged_df = self.merged_df.dropna(subset=["datetime_srs"])
        self.merged_df = self.merged_df.dropna(subset=["datetime_hmi", "datetime_mdi"], how="all")
        self.dropped_rows = self.dropped_rows[~self.dropped_rows.index.isin(self.merged_df.index)].copy()

        logger.info(f"merged_df: \n{self.merged_df[['datetime_srs', 'datetime_hmi', 'datetime_mdi']]}")
        logger.info(f"dropped_rows: \n{self.dropped_rows[['datetime_srs', 'datetime_hmi', 'datetime_mdi']]}")
        logger.info(f"dates dropped: \n{self.dropped_rows['datetime_srs'].unique()}")

        directory_path = Path(dv.MAG_INTERMEDIATE_DIR)
        if not directory_path.exists():
            directory_path.mkdir(parents=True)

        self.merged_df.to_csv(dv.MAG_INTERMEDIATE_DATA_CSV)

    def fetch_magnetograms(self, mag_df, max_retries=5):
        """
        download the magnetograms using parfive (with one connection),
        and return the list of files

        Returns
        -------
        results : List[str]
            List of filepaths (strings) for the downloaded files
        """
        base_directory_path = Path(dv.MAG_RAW_DATA_DIR)
        if not base_directory_path.exists():
            base_directory_path.mkdir(parents=True)

        # HMI/MDI
        # !TODO change this so that it's not as specific as `url_hmi`, `url_mdi`
        urls = list(mag_df.url_hmi.dropna().unique()) + list(mag_df.url_mdi.dropna().unique())

        # Only 1 parallel connection (`max_conn`, `max_splits`)
        # https://docs.sunpy.org/en/stable/_modules/sunpy/net/jsoc/jsoc.html#JSOCClient
        downloader = Downloader(
            max_conn=1,
            progress=True,
            overwrite=False,
            max_splits=1,
        )

        paths = []
        for url in urls:
            filename = url.split("/")[-1]  # Extract the filename from the URL
            paths.append(base_directory_path / filename)

        for aurl, fname in zip(urls, paths):
            downloader.enqueue_file(aurl, filename=fname, max_splits=1)

        results = downloader.download()

        if len(results.errors) != 0:
            logger.warn(f"results.errors: {results.errors}")
            # attempt a retry
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


if __name__ == "__main__":
    logger.info(f"Executing {__file__} as main program")
    _ = DataManager(dv.DATA_START_TIME, dv.DATA_END_TIME)
