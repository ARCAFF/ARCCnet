from datetime import datetime

import arccnet.data_generation.utils.default_variables as dv
from arccnet.data_generation.catalogs.active_region_catalogs.swpc import SWPCCatalog
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

        # 1. fetch metadata
        logger.info(">> Fetching Metadata")
        self.srs_raw, self.srs_raw_missing = self.fetch_metadata()
        logger.info(f"\n{self.srs_raw}")

        # 2. clean metadata
        logger.info(">> Cleaning Metadata")
        self.srs_clean = self.clean_metadata()
        logger.info(f"\n{self.srs_clean}")

        # 3. merge metadata sources
        self.merged_data = self.merge_metadata_sources()

        # 4. download image data
        # ...

        logger.info(">> Execution completed successfully")

    def fetch_metadata(self):
        """
        method to fetch and return data from various sources
        """

        # download the txt files and create an SRS catalog
        _ = self.swpc.fetch_data(self.start_date, self.end_date)
        srs_raw, srs_raw_missing = self.swpc.create_catalog()

        # HMI & MDI

        return srs_raw, srs_raw_missing

    def clean_metadata(self):
        """
        clean data from each instrument
        """

        # clean the raw SRS catalog
        srs_clean = self.swpc.clean_catalog()

        # clean the raw HMI/MDI catalogs
        # ...

        return srs_clean

    def merge_metadata_sources(self):
        """
        method to merge the data sources
        """
        # merge `pd.DataFrames`, for example
        pass


if __name__ == "__main__":
    logger.info(f"Executing {__file__} as main program")
    _ = DataManager(dv.DATA_START_TIME, dv.DATA_END_TIME)
