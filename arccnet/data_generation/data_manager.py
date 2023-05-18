from datetime import datetime

import arccnet.data_generation.utils.default_variables as dv
from arccnet.data_generation.catalogs.active_region_catalogs.swpc import SWPCCatalog
from arccnet.data_generation.utils.data_logger import logger

__all__ = ["DataManager"]


class DataManager:
    def __init__(
        self,
        start_date: datetime = dv.DATA_START_TIME,
        end_date: datetime = dv.DATA_END_TIME,
        # data_path: str = dv.BASE_DIR,
    ):
        # set dates
        self.start_date = start_date
        self.end_date = end_date

        # !TODO implement this shit.
        # dv.BASE_DIR = data_path
        # print(dv.BASE_DIR)
        # if not os.path.exists(data_path):
        #     os.makedirs(data_path)

        logger.info(f"Instantiated `DataManager` for {self.start_date}, {self.end_date}")

        # instantiate classes
        # self.hmi = HMIMagnetogram()
        # self.mdi = MDIMagnetogram()
        self.swpc = SWPCCatalog()

        logger.info(">> Fetching Data")
        self.catalog, self.catalog_missing = self.fetch_data(self.start_date, self.end_date)
        logger.info(f"\n{self.catalog}")

        logger.info(">> Cleaning Data")
        self.clean_catalog = self.clean_data()
        logger.info(f"\n{self.clean_catalog}")
        logger.info(">> Execution completed successfully ")

    def fetch_data(self, start_date, end_date):
        _ = self.swpc.fetch_data(start_date, end_date)
        # create catalog
        c, cm = self.swpc.create_catalog()
        return c, cm

    def clean_data(self):
        swpc_clean = self.swpc.clean_data
        return swpc_clean

    def combine_data(self):
        pass


if __name__ == "__main__":
    logger.info(f"Executing {__file__} as main program")
    dm = DataManager(dv.DATA_START_TIME, dv.DATA_END_TIME)
