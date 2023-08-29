from pathlib import Path
from datetime import datetime

import pandas as pd
from parfive import Results
from sunpy.util.parfive_helpers import Downloader

import arccnet.data_generation.utils.default_variables as dv
from arccnet.data_generation.catalogs.active_region_catalogs.swpc import SWPCCatalog
from arccnet.data_generation.magnetograms.instruments import (
    HMILOSMagnetogram,
    HMISHARPs,
    MDILOSMagnetogram,
    MDISMARPs,
)
from arccnet.data_generation.utils.data_logger import logger
from arccnet.data_generation.utils.utils import save_df_to_html

__all__ = ["DataManager"]


class DataManager:
    """
    Main data management class.

    This class instantiates and handles data acquisition for the individual instruments and cutouts
    """

    def __init__(
        self,
        start_date: datetime = dv.DATA_START_TIME,
        end_date: datetime = dv.DATA_END_TIME,
        merge_tolerance: pd.Timedelta = pd.Timedelta("30m"),
        download_fits: bool = True,
        save_to_csv: bool = True,
        save_to_html: bool = False,
    ):
        """
        Initialize the DataManager.

        Parameters
        ----------
        start_date : datetime, optional
            Start date of data acquisition period. Default is `arccnet.data_generation.utils.default_variables.DATA_START_TIME`.

        end_date : datetime, optional
            End date of data acquisition period. Default is `arccnet.data_generation.utils.default_variables.DATA_END_TIME`.

        merge_tolerance : pd.Timedelta, optional
            Time tolerance for merging operations. Default is pd.Timedelta("30m").

        download_fits : bool, optional
            Whether to download FITS files. Default is True.
        """
        self.start_date = start_date
        self.end_date = end_date
        self.save_to_csv = save_to_csv
        self.save_to_html = save_to_html
        logger.info(
            f"Instantiated `DataManager` for {self.start_date} -> {self.end_date}, "
            + f"with save_to_csv={self.save_to_csv}, save_to_html={self.save_to_html} "
        )

        # instantiate classes
        # SWPC Catalog
        self.swpc = SWPCCatalog()
        # !TODO change this into an iterable
        # Full-disk Magnetograms
        self.hmi = HMILOSMagnetogram()
        self.mdi = MDILOSMagnetogram()
        # Cutouts
        self.sharps = HMISHARPs()
        self.smarps = MDISMARPs()

        # 1. fetch metadata
        logger.info(">> Fetching metadata")
        (
            self.srs,
            self.mdi_keys,
            self.hmi_keys,
            self.sharp_keys,
            self.smarp_keys,
        ) = self.fetch_metadata()
        srs_raw, _ = self.srs
        logger.info(f"\n{srs_raw}")

        # 2. clean metadata
        self.srs_clean = self.swpc.clean_catalog()
        logger.info(f"\n{self.srs_clean}")

        # 3. merge metadata sources
        logger.info(f">> Merging full-disk metadata with tolerance {merge_tolerance}")
        self.merged_df, self.merged_df_dropped_rows = self.merge_hmimdi_metadata(
            self.srs_clean, self.hmi_keys, self.mdi_keys, tolerance=merge_tolerance
        )

        logger.info(">> Merging full-disk metadata with cutouts")
        #  Merge the HMI and MDI components of the `merged_df` with the SHARPs and SMARPs DataFrames
        # !TODO this is terrible, change this!
        # HMI-SHARPs
        merged_df_hs = self.merged_df.copy(deep=True).drop(
            columns=["magnetogram_fits_mdi", "datetime_mdi", "url_mdi"]
        )  # drop mdi columns
        merged_df_hs.columns = [col.rstrip("_hmi") if col.endswith("_hmi") else col for col in merged_df_hs.columns]
        self.hmi_sharps = self.merge_activeregionpatchs(merged_df_hs, self.sharp_keys[["datetime", "url", "record"]])
        # MDI-SMARPs
        merged_df_ms = self.merged_df.copy(deep=True).drop(
            columns=["magnetogram_fits_hmi", "datetime_hmi", "url_hmi"]
        )  # drop hmi columns
        merged_df_ms.columns = [col.rstrip("_mdi") if col.endswith("_mdi") else col for col in merged_df_ms.columns]
        self.mdi_smarps = self.merge_activeregionpatchs(merged_df_ms, self.smarp_keys[["datetime", "url", "record"]])
        # ------

        # !TODO replace the 'url' column with a file or at least append...
        # 4. save merged dataframes
        self.save_dfs(
            dataframe_list=[self.merged_df, self.hmi_sharps, self.mdi_smarps],
            filepaths=[
                Path(dv.MAG_INTERMEDIATE_HMIMDI_DATA_CSV),
                Path(dv.MAG_INTERMEDIATE_HMISHARPS_DATA_CSV),
                Path(dv.MAG_INTERMEDIATE_MDISMARPS_DATA_CSV),
            ],
            to_csv=self.save_to_csv,
            to_html=self.save_to_html,
        )

        # 4a. check if image data exists
        # !TODO implement this checking if each file that is expected exists.
        # Compile list of URLs to download
        self.urls_to_download = self.extract_urls()

        if download_fits:
            self.fetch_urls(self.urls_to_download)
            logger.info("Download completed successfully")
        else:
            logger.info(
                "To fetch the magnetograms, use the `.fetch_urls()` method with a list(str) of urls, e.g. `.urls_to_download`"
            )

    def extract_urls(self, column_prefix="url_"):
        """
        Extract unique URLs from multiple dataframes using a common column prefix.

        Parameters
        ----------
        column_prefix : str, optional
            Prefix of columns containing URLs in the dataframes, by default "url_".

        Returns
        -------
        list
            A list of unique URLs extracted from the specified dataframes.

        Notes
        -----
        This function extracts URLs from columns with names starting with the specified
        `column_prefix` in three dataframes: `self.merged_df`, `self.hmi_sharps`, and
        `self.mdi_smarps`. The extracted URLs are returned as a list of unique values.
        """

        def get_url_columns(dataframe):
            return [col for col in dataframe.columns if col.startswith(column_prefix)]

        merged_url_columns = get_url_columns(self.merged_df)
        hmi_sharps_url_columns = get_url_columns(self.hmi_sharps)
        mdi_smarps_url_columns = get_url_columns(self.mdi_smarps)

        all_urls = (
            pd.concat(
                [
                    self.merged_df[merged_url_columns],
                    self.hmi_sharps[hmi_sharps_url_columns],
                    self.mdi_smarps[mdi_smarps_url_columns],
                ]
            )
            .stack()
            .dropna()
            .unique()
        )

        return list(all_urls)

    def save_dfs(
        self,
        dataframe_list: list[pd.DataFrame],
        filepaths: list[Path],
        to_csv: bool = True,
        to_html: bool = False,
    ) -> None:
        """
        Save DataFrames to CSV and/or HTML files.

        Parameters
        ----------
        dataframes : list
            List of DataFrames to be saved.

        filepaths : list
            List of filepaths to save DataFrames.

        to_csv : bool, optional
            Whether to save as CSV. Default is True.

        to_html : bool, optional
            Whether to save as HTML. Default is True.
        """
        if not to_csv and not to_html:
            logger.info("No action taken. Both `to_csv` and `to_html` are False.")

        for df, loc in zip(dataframe_list, filepaths):
            directory_path = loc.parent
            if not directory_path.exists():
                directory_path.mkdir(parents=True)

            if to_csv:
                df.to_csv(loc)
            if to_html:
                save_df_to_html(df, str(loc))

    def fetch_metadata(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Fetch and return data from various sources.

        Returns
        -------
        tuple
            Tuple containing data from different sources.
        """
        # download the txt files and create an SRS catalog
        self.swpc.fetch_data(self.start_date, self.end_date)
        srs = self.swpc.create_catalog()

        # HMI & MDI
        # !TODO itereate over children of `BaseMagnetogram`
        hmi_keys = self.hmi.fetch_metadata(self.start_date, self.end_date, batch_frequency=12, to_csv=self.save_to_csv)
        mdi_keys = self.mdi.fetch_metadata(self.start_date, self.end_date, batch_frequency=12, to_csv=self.save_to_csv)
        sharp_keys = self.sharps.fetch_metadata(
            self.start_date, self.end_date, batch_frequency=3, to_csv=self.save_to_csv
        )
        smarp_keys = self.smarps.fetch_metadata(
            self.start_date, self.end_date, batch_frequency=3, to_csv=self.save_to_csv
        )

        # logging
        for name, dataframe in {
            "HMI Keys": hmi_keys,
            "SHARP Keys": sharp_keys,
            "MDI Keys": mdi_keys,
            "SMARP Keys": smarp_keys,
        }.items():
            logger.info(f"{name}: \n{dataframe[['T_REC','T_OBS','DATE-OBS','datetime','magnetogram_fits', 'url']]}")

        return srs, mdi_keys, hmi_keys, sharp_keys, smarp_keys

    def merge_activeregionpatchs(
        self,
        full_disk_data,
        cutout_data,
    ) -> pd.DataFrame:
        """
        Merge active region patch data.

        Parameters
        ----------
        full_disk_data : pd.DataFrame
            Data from full disk observations with columns;
            ["datetime", "url"]

        cutout_data : pd.DataFrame
            Data from active region cutouts. The pd.DataFrame must contain a "datetime" column.

        Returns
        -------
        pd.DataFrame
            Merged DataFrame of active region patch data.
        """
        # # merge srs_clean and hmi
        expected_columns = ["datetime", "url"]

        if not set(expected_columns).issubset(full_disk_data.columns):
            logger.warn(
                "Currently, the columns in full_disk_data need to be ['magnetogram_fits', 'datetime', 'url'] due to poor planning and thought."
            )
            raise NotImplementedError()

        if not set(expected_columns).issubset(cutout_data.columns):
            logger.warn(
                "Currently, the columns in cutout_data need to be ['magnetogram_fits', 'datetime', 'url'] due to poor planning and thought."
            )
            raise NotImplementedError()

        # # !TODO do a check for certain keys (no duplicates...)
        # # extract only the relevant HMI keys, and rename
        # # (should probably do this earlier on)
        # full_disk_data = full_disk_data[mag_cols]
        # full_disk_data = full_disk_data.add_suffix("_hmi")
        full_disk_data_dropna = full_disk_data.dropna().reset_index(drop=True)

        # cutout_data = cutout_data[mag_cols]
        cutout_data = cutout_data.add_suffix("_arc")
        cutout_data_dropna = cutout_data.dropna().reset_index(drop=True)

        logger.info(f"len of full_disk_data_dropna is {len(full_disk_data_dropna)}")
        logger.info(f"len of cutout_data_dropna is {len(cutout_data_dropna)}")

        # need to figure out if a left merge is what we want...
        merged_df = pd.merge(
            full_disk_data_dropna, cutout_data_dropna, left_on="datetime", right_on="datetime_arc"
        )  # no tolerance as should be exact

        logger.info(f"len of merged_df is {len(merged_df)}")
        return merged_df

    def merge_hmimdi_metadata(
        self,
        srs_keys: pd.DataFrame,
        hmi_keys: pd.DataFrame,
        mdi_keys: pd.DataFrame,
        tolerance: pd.Timedelta = pd.Timedelta("30m"),
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Merge SRS, HMI, and MDI metadata.

        This function merges NOAA SRS, Helioseismic and Magnetic Imager (HMI),
        and Michelson Doppler Imager (MDI) metadata based on datetime keys with a specified tolerance.

        Parameters
        ----------
        srs_keys : pd.DataFrame
            DataFrame containing SRS metadata.

        hmi_keys : pd.DataFrame
            DataFrame containing HMI metadata.

        mdi_keys : pd.DataFrame
            DataFrame containing MDI metadata.

        tolerance : pd.Timedelta, optional
            Time tolerance for merging operations. Default is pd.Timedelta("30m").

        Returns
        -------
        pd.DataFrame
            Merged DataFrame of SRS, HMI, and MDI metadata.

        pd.DataFrame
            DataFrame of dropped rows after merging.
        """
        srs_keys = srs_keys.copy(deep=True)
        # merge srs_clean and hmi
        mag_cols = ["magnetogram_fits", "datetime", "url"]

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

    @staticmethod
    def fetch_urls(
        urls: list[str] = None, base_directory_path: Path = Path(dv.MAG_RAW_DATA_DIR), max_retries: int = 5
    ) -> Results:
        """
        Download data from urls using parfive.

        Parameters
        ----------
        urls : list[str]
            List of URLs to download. Default is None.

        base_directory_path : str or Path, optional
            Base directory path to save downloaded files. Default is `arccnet.data_generation.utils.default_variables.MAG_RAW_DATA_DIR`

        max_retries : int, optional
            Maximum number of download retries. Default is 5.

        Returns
        -------
        results : parfive.Results or None
            parfive results object of the download operation, or None if no URLs provided or invalid format.
        """
        if urls is None or not all(isinstance(url, str) for url in (urls or [])):
            logger.warning("Invalid URLs format. Expected a list of strings.")
            return None

        if not base_directory_path.exists():
            base_directory_path.mkdir(parents=True)

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
            logger.warning(f"results.errors: {results.errors}")
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
