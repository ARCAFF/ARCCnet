import random
import multiprocessing
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sunpy.map
from tqdm import tqdm

import astropy.units as u
from astropy.coordinates import SkyCoord

import arccnet.data_generation.utils.default_variables as dv
from arccnet.data_generation.utils.data_logger import logger
from arccnet.data_generation.utils.utils import make_relative, save_compressed_map

matplotlib.use("Agg")

__all__ = ["MagnetogramProcessor", "ARExtractor", "QSExtractor"]  # , "ARDetection"]


class MagnetogramProcessor:
    """
    Process Magnetograms.

    This class provides methods to process magnetogram data using multiprocessing.
    """

    def __init__(
        self,
        csv_in_file: Path = Path(dv.MAG_INTERMEDIATE_HMIMDI_DATA_CSV),
        csv_out_file: Path = Path(dv.MAG_INTERMEDIATE_HMIMDI_PROCESSED_DATA_CSV),
        columns: list[str] = None,
        processed_data_dir: Path = Path(dv.MAG_INTERMEDIATE_DATA_DIR),
        process_data: bool = True,
        use_multiprocessing: bool = False,
    ) -> None:
        """
        Reads data paths, processes and saves the data.
        """
        logger.info("Instantiated `MagnetogramProcessor`")
        if columns is None:
            columns = ["download_path_hmi", "download_path_mdi"]
        self.processed_data_dir = processed_data_dir
        self.paths, self.loaded_csv = self._read_columns(columns=columns, csv_file=csv_in_file)

        if process_data:
            self.processed_paths = self.process_data(
                use_multiprocessing=use_multiprocessing,
                paths=self.paths,
                save_path=self.processed_data_dir,
            )

            # Map processed paths to original DataFrame using Path.name
            processed_path_mapping = {path.name: path for path in self.processed_paths}
            for column in columns:
                self.loaded_csv[f"processed_{column}"] = self.loaded_csv.apply(
                    lambda row: processed_path_mapping.get(Path(row[column]).name, np.nan)
                    if pd.notna(row[column])
                    else np.nan,
                    axis=1,
                )

            # should probably allow to csv to be a value
            self.loaded_csv.to_csv(csv_out_file, index=False)

    def _read_columns(
        self,
        columns: list[str] = ["download_path_hmi", "download_path_mdi"],
        csv_file=Path(dv.MAG_INTERMEDIATE_HMIMDI_DATA_CSV),
    ):
        """
        Read and prepare data paths from CSV file.

        Parameters
        ----------
        url_columns: list[str]
            list of column names (str).

        csv_file: Path
            location of the csv file to read.

        Returns
        -------
        paths: list[Path]
            List of data file paths.
        """

        if csv_file.exists():
            loaded_data = pd.read_csv(csv_file)
            file_list = [column for col in columns for column in loaded_data[col].dropna().unique()]
            paths = [Path(path) if isinstance(path, str) else np.nan for path in file_list]

            existing_paths = [path for path in paths if path.exists()]  # check if the raw files exist
            if len(existing_paths) < len(paths):
                missing_paths = [str(path) for path in paths if path not in existing_paths]
                raise FileNotFoundError(f"The following paths do not exist: {', '.join(missing_paths)}")
        else:
            raise FileNotFoundError(f"{csv_file} does not exist.")

        return paths, loaded_data

    def process_data(self, use_multiprocessing: bool = True, paths=None, save_path=None):
        """
        Process data using multiprocessing.
        """
        # if paths is None:
        #     paths = self.paths

        if save_path is None:
            base_directory_path = self.processed_data_dir
        else:
            base_directory_path = save_path

        if not base_directory_path.exists():
            base_directory_path.mkdir(parents=True)

        processed_paths = []  # list of processed filepaths

        logger.info(f"processing of {len(paths)} paths with multiprocessing = {use_multiprocessing}")
        if use_multiprocessing:
            # Use tqdm to create a progress bar for multiprocessing
            with multiprocessing.Pool() as pool:
                for processed_path in tqdm(
                    pool.imap_unordered(
                        self._multiprocess_and_save_data_wrapper, [(path, base_directory_path) for path in paths]
                    ),
                    total=len(paths),
                    desc="Processing",
                ):
                    processed_paths.append(processed_path)
                    # pass
        else:
            for path in tqdm(self.paths, desc="Processing"):
                processed_path = self._process_and_save_data(path, base_directory_path)
                processed_paths.append(processed_path)

        return processed_paths

    def _multiprocess_and_save_data_wrapper(self, args):
        """
        Wrapper method to process and save data using `_process_and_save_data`.

        This method takes a tuple of arguments containing the file path and the output directory,
        and then calls the `_process_and_save_data` method with the provided arguments.

        Parameters
        ----------
        args : tuple
            A tuple containing the file path and output directory.

        Returns
        -------
        Path
            A path for the processed file

        See Also:
        --------
        _process_and_save_data, _process_data
        """
        file, output_dir = args
        return self._process_and_save_data(file, output_dir)

    def _process_and_save_data(self, file: Path, output_dir: Path) -> Path:
        """
        Process data and save compressed map.

        Parameters
        ----------
        file : Path
            Data file path.

        output_dir : Path
            Directory to save processed data.

        Returns
        -------
        None
        """
        output_file = output_dir / file.name  # !TODO prefix the file.name?
        processed_data = self._process_datum(file)
        save_compressed_map(processed_data, path=output_file, overwrite=True)
        return output_file

    def _process_datum(self, file) -> sunpy.map.Map:
        """
        Process a single data file.

        Processing Steps:
            1. Load and rotate
            2. Set off-disk data to 0
            # !TODO
            3. Normalise radius to a fixed value
            4. Project to a certain location in space

        Parameters
        ----------
        file : Path
            Data file path.

        Returns
        -------
        rotated_map : sunpy.map.Map
            Processed sunpy map.
        """
        #!TODO remove 'BLANK' keyword
        # v3925 WARNING: VerifyWarning: Invalid 'BLANK' keyword in header.
        # The 'BLANK' keyword is only applicable to integer data, and will be ignored in this HDU.
        # [astropy.io.fits.hdu.image]
        # 1. Load & Rotate
        single_map = sunpy.map.Map(file)
        rotated_map = self._rotate_datum(single_map)
        # 2. set data off-disk to 0 (np.nan would be ideal, but deep learning)
        rotated_map.data[~sunpy.map.coordinate_is_on_solar_disk(sunpy.map.all_coordinates_from_map(rotated_map))] = 0.0
        # !TODO understand why this isn't working correctly with MDI (leaves a white ring around the disk)
        # 3. !TODO normalise radius to fixed value
        # 4. !TODO project to a certain location in space
        return rotated_map

    def _rotate_datum(self, amap: sunpy.map.Map) -> sunpy.map.Map:
        """
        Rotate a map according to metadata.

        Args:
        amap : sunpy.map.Map
            An input sunpy map.

        Parameters
        ----------
        rotated_map : sunpy.map.Map
            Rotated sunpy map.

        Notes
        -----
        before rotation a HMI map may have: `crota2 = 180.082565`, for example.
        """
        return amap.rotate()


class ARExtractor:
    def __init__(self) -> None:
        loaded_data = load_filename()

        dv_process_fits_path = Path(dv.MAG_PROCESSED_FITS_DIR)
        if not dv_process_fits_path.exists():
            dv_process_fits_path.mkdir(parents=True)

        dv_summary_plots_path = Path(dv.MAG_PROCESSED_SUMMARYPLOTS_DIR)
        if not dv_summary_plots_path.exists():
            dv_summary_plots_path.mkdir(parents=True)
        # Iterate through the columns and update the paths
        # !TODO deal with earlier on in the codebase

        # set empty list of cutout for hmi
        cutout_list_hmi = []
        cutout_hmi_dim = []
        bls = []
        trs = []

        loaded_subset = loaded_data[
            [
                "Latitude",
                "Longitude",
                "Number",
                "Area",
                "Z",
                "Mag Type",
                "processed_hmi",
                "datetime_hmi",
                "datetime_srs",
            ]
        ].copy()

        logger.info(loaded_subset)

        # drop rows with NaN (so drop none with HMI)
        # !TODO go through HMI and MDI separately
        loaded_subset.dropna(inplace=True)
        # group by SRS files
        grouped_data = loaded_subset.groupby("datetime_srs")

        # logger.info(f"there are len(grouped_data) {len(grouped_data)}")

        for time_srs, group in tqdm(grouped_data):
            summary_info = []

            my_hmi_map = sunpy.map.Map(group.processed_hmi.unique()[0])  # take the first hmi
            time_hmi = group.datetime_hmi.unique()[0]

            # iterate through the groups (by datetime_srs)
            for _, row in group.iterrows():
                # logger.info(srs_dt)
                # extract the lat/long and NOAA AR Number (for saving)
                numbr = row["Number"]
                # logger.info(f" >>> {numbr}")

                my_hmi_submap, top_right, bottom_left, ar_pos_pixels = extract_submaps(
                    my_hmi_map, time_hmi, row[["Latitude", "Longitude"]], xsize=dv.X_EXTENT, ysize=dv.Y_EXTENT
                )

                # append to summary info for plotting
                summary_info.append([top_right, bottom_left, numbr, my_hmi_submap.data.shape, ar_pos_pixels, time_srs])
                cutout_list_hmi.append(dv_process_fits_path / f"{time_srs}_{numbr}.fits")
                cutout_hmi_dim.append(my_hmi_submap.data.shape)

                # !TODO see
                # https://gitlab.com/frontierdevelopmentlab/living-with-our-star/super-resolution-maps-of-solar-magnetic-field/-/blob/master/source/prep.py?ref_type=heads
                save_compressed_map(my_hmi_submap, dv_process_fits_path / f"{time_srs}_{numbr}.fits", overwrite=True)

                bls.append(bottom_left)
                trs.append(top_right)

                del my_hmi_submap  # delete the submap

            self.plot(my_hmi_map, time_srs, dv_summary_plots_path, summary_info)

            del summary_info

        loaded_subset.loc[:, "hmi_cutout"] = cutout_list_hmi
        loaded_subset.loc[:, "hmi_cutout_dim"] = cutout_hmi_dim
        loaded_subset.loc[:, "bottom_left"] = bls
        loaded_subset.loc[:, "top_right"] = trs

        loaded_subset.to_csv(Path(dv.MAG_PROCESSED_DIR) / "processed.csv")

        # clean data
        dv_final_path = Path(dv.DATA_DIR_FINAL)
        if not dv_final_path.exists():
            dv_final_path.mkdir(parents=True)

        # clean data
        # 1. Ensure the data is (400, 800)
        loaded_subset_cleaned = loaded_subset[loaded_subset["hmi_cutout_dim"] == (dv.Y_EXTENT, dv.X_EXTENT)]
        # Drop NaN, Reset Index, Save to `arcutout_clean.csv`
        loaded_subset_cleaned = loaded_subset_cleaned.dropna()
        loaded_subset_cleaned = loaded_subset_cleaned.reset_index()
        loaded_subset_cleaned.to_csv(Path(dv.DATA_DIR_FINAL) / "arcutout_clean.csv")  # need to reset index

        self.data = loaded_subset_cleaned

    def plot(self, aplotmap, time, filepath, summary_arr):
        # Plotting and saving
        # !TODO move to a new function
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(projection=aplotmap)
        aplotmap.plot_settings["norm"].vmin = -1500
        aplotmap.plot_settings["norm"].vmax = 1500
        aplotmap.plot(axes=ax, cmap="hmimag")

        text_objects = []

        for tr, bl, num, shape, arc, _ in summary_arr:
            if shape == (dv.Y_EXTENT, dv.X_EXTENT):
                rectangle_cr = "red"
                rectangle_ls = "-"
            else:
                rectangle_cr = "black"
                rectangle_ls = "-."

            aplotmap.draw_quadrangle(
                bl,
                axes=ax,
                top_right=tr,
                edgecolor=rectangle_cr,
                linestyle=rectangle_ls,
                linewidth=1,
                label=num,
            )

            text = ax.text(
                arc[0],
                arc[1] + (dv.Y_EXTENT / 2) + 5,
                num,
                **{"size": "x-small", "color": "black", "ha": "center"},
            )
            text_objects.append(text)

        plt.savefig(filepath / f"{time}.png", dpi=300)
        plt.close("all")

        # remove the text objects?
        for text in text_objects:
            text.remove()


class QSExtractor:
    def __init__(self, num_random_attempts=10):
        filename = Path(dv.MAG_PROCESSED_DIR) / "processed.csv"

        if filename.exists():
            loaded_data = pd.read_csv(filename)

        dir = Path(dv.MAG_PROCESSED_QSFITS_DIR)
        if not dir.exists():
            dir.mkdir(parents=True)

        dv_summary_plots_path = Path(dv.MAG_PROCESSED_QSSUMMARYPLOTS_DIR)
        if not dv_summary_plots_path.exists():
            dv_summary_plots_path.mkdir(parents=True)

        loaded_data["datetime_srs"] = pd.to_datetime(loaded_data["datetime_srs"])
        grouped_data = loaded_data.groupby("datetime_srs")

        qs_df = pd.DataFrame(columns=["datetime_srs", "datetime_hmi", "hmi_cutout", "hmi_cutout_dim"])

        all_qs = []

        for time_srs, group in grouped_data:
            my_hmi_map = sunpy.map.Map(group.processed_hmi.unique()[0])  # take the first hmi
            time_hmi = group.datetime_hmi.unique()[0]

            vals = []
            for _, row in group.iterrows():
                # extract the lat/long and NOAA AR Number (for saving)
                lat, lng, _ = row[["Latitude", "Longitude", "Number"]]
                # logger.info(f" >>> {lat}, {lng}, {numbr}")

                ar_pos_pixels = (
                    SkyCoord(
                        lng * u.deg,
                        lat * u.deg,
                        obstime=time_hmi,
                        frame=sunpy.coordinates.frames.HeliographicStonyhurst,
                    )
                    .transform_to(my_hmi_map.coordinate_frame)
                    .to_pixel(my_hmi_map.wcs)
                )

                # all active region centres
                vals.append(ar_pos_pixels)

            qs_reg = []
            for i in range(0, num_random_attempts):
                # create random location
                rand_1 = random.uniform(-1000, 1000) * u.arcsec
                rand_2 = random.uniform(-500, 500) * u.arcsec

                # convert to pixel coordinates
                ar_pos_hgs = SkyCoord(
                    rand_1,
                    rand_2,
                    frame=my_hmi_map.coordinate_frame,
                ).to_pixel(my_hmi_map.wcs)

                # check ar_pos_hgs is far enough from other vals
                tt = list(
                    map(
                        lambda v: self.is_point_far_from_point(
                            ar_pos_hgs[0], ar_pos_hgs[1], v[0], v[1], dv.X_EXTENT * 1.2, dv.Y_EXTENT * 1.2
                        ),
                        vals,
                    )
                )

                if all(tt):  # len of tt?
                    top_right = [ar_pos_hgs[0] + (dv.X_EXTENT - 1) / 2, ar_pos_hgs[1] + (dv.Y_EXTENT - 1) / 2] * u.pix
                    bottom_left = [ar_pos_hgs[0] - (dv.X_EXTENT - 1) / 2, ar_pos_hgs[1] - (dv.Y_EXTENT - 1) / 2] * u.pix
                    my_hmi_submap = my_hmi_map.submap(bottom_left, top_right=top_right)

                    fn = (
                        Path(dv.MAG_PROCESSED_QSFITS_DIR)
                        / f"{time_srs.year}-{time_srs.month}-{time_srs.day}_QS_{i}.fits"
                    )

                    save_compressed_map(my_hmi_submap, path=fn, overwrite=True)

                    qs_temp = pd.DataFrame(
                        {
                            "datetime_hmi": group.datetime_hmi.unique()[0],
                            "datetime_srs": pd.to_datetime(group.datetime_srs.unique()[0]).date(),
                            "hmi_cutout": str(fn),
                            "hmi_cutout_dim": [my_hmi_submap.data.shape],
                        },
                        index=[0],
                    )

                    qs_df = pd.concat([qs_df, qs_temp], ignore_index=True)

                    del my_hmi_submap

                    vals.append(ar_pos_hgs)
                    qs_reg.append(ar_pos_hgs)

            all_qs += qs_reg[:]
            # logger.info(f"{len(qs_reg)} QS regions saved at {time_hmi}")

            self.plot(my_hmi_map, vals, qs_reg, time_srs)

            qs_df.to_csv(Path(dv.MAG_PROCESSED_DIR) / "qs_fits.csv")
            self.data = qs_df

    def plot(self, hmi_map, vals, qs_reg, time_srs):
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(projection=hmi_map)
        hmi_map.plot_settings["norm"].vmin = -1500
        hmi_map.plot_settings["norm"].vmax = 1500
        hmi_map.plot(axes=ax, cmap="hmimag")

        for value in vals:
            top_right = [value[0] + (dv.X_EXTENT - 1) / 2, value[1] + (dv.Y_EXTENT - 1) / 2] * u.pix
            bottom_left = [value[0] - (dv.X_EXTENT - 1) / 2, value[1] - (dv.Y_EXTENT - 1) / 2] * u.pix
            my_hmi_submap = hmi_map.submap(bottom_left, top_right=top_right)

            rectangle_cr = "red"
            if my_hmi_submap.data.shape == (dv.Y_EXTENT, dv.X_EXTENT):
                rectangle_ls = "-"
            else:
                rectangle_ls = "-."
            if value in qs_reg:
                rectangle_cr = "blue"

            hmi_map.draw_quadrangle(
                bottom_left,
                axes=ax,
                top_right=top_right,
                edgecolor=rectangle_cr,
                linestyle=rectangle_ls,
                linewidth=1,
            )

        plt.savefig(
            Path(dv.MAG_PROCESSED_QSSUMMARYPLOTS_DIR) / f"{time_srs.year}-{time_srs.month}-{time_srs.day}_QS.png",
            dpi=300,
        )
        plt.close("all")

    def is_point_far_from_point(self, x, y, x1, y1, threshold_x, threshold_y):
        # test this code
        return abs(x - x1) > threshold_x or abs(y - y1) > threshold_y


def load_filename():
    filename = Path(dv.MAG_INTERMEDIATE_HMIMDI_DATA_CSV)

    if filename.exists():
        loaded_data = pd.read_csv(filename)

    columns_to_update = ["url_hmi", "url_mdi"]
    new_columns = ["processed_hmi", "processed_mdi"]

    # !TODO replace with default_variables.py
    dv_base_path = Path(dv.MAG_INTERMEDIATE_DATA_DIR)

    # Iterate through the columns and update the paths
    # !TODO deal with earlier on in the codebase
    for old_column, new_column in zip(columns_to_update, new_columns):
        loaded_data[new_column] = loaded_data[old_column].map(
            lambda x: dv_base_path / Path(x).name if pd.notna(x) else x
        )

    return loaded_data


def extract_submaps(map, time, coords, xsize=dv.X_EXTENT, ysize=dv.Y_EXTENT) -> sunpy.map.Map:
    """

    Parameters
    ----------
    map : sunpy.map.Map

    time : datetime

    coords : tuple
        tuple consisting of (latitude, longitude)

    xsize : int
        x extent of region to extract (in pixels)

    ysize : int
        y extend of region to extract (in pixels)


    Returns
    -------
    submap : sunpy.map.Map
        sunpy map centered on coords, with size (xsize, ysize)

    """
    print(f">> {map.date}, {time}")

    lat, lng = coords

    ar_pos_hgs = SkyCoord(
        lng * u.deg,
        lat * u.deg,
        obstime=time,
        frame=sunpy.coordinates.frames.HeliographicStonyhurst,
    )

    transformed = ar_pos_hgs.transform_to(map.coordinate_frame)
    ar_pos_pixels = transformed.to_pixel(map.wcs)

    # Perform in pixel coordinates
    top_right = [ar_pos_pixels[0] + (xsize - 1) / 2, ar_pos_pixels[1] + (ysize - 1) / 2] * u.pix
    bottom_left = [
        ar_pos_pixels[0] - (xsize - 1) / 2,
        ar_pos_pixels[1] - (ysize - 1) / 2,
    ] * u.pix

    submap = map.submap(bottom_left, top_right=top_right)

    return submap, top_right, bottom_left, ar_pos_pixels


if __name__ == "__main__":
    logger.info(f"Executing {__file__} as main program")

    mag_process = True
    ar_classification = False
    # ar_detection = True

    # 1. Process full-disk magnetograms
    if mag_process:
        mp = MagnetogramProcessor()
        mp.process_data(use_multiprocessing=True)
        logger.info(">> processed data")

    # 2. Extract NOAA ARs and QS regions
    if ar_classification:
        ar_df = ARExtractor()
        qs_df = QSExtractor()
        arccnet_df = pd.concat([ar_df.data, qs_df.data], ignore_index=True).sort_values(
            by="datetime_hmi", ignore_index=True
        )
        arccnet_df = arccnet_df[
            [
                "datetime_srs",
                "Latitude",
                "Longitude",
                "Number",
                "Area",
                "Z",
                "Mag Type",
                "datetime_hmi",
                "hmi_cutout",
                "hmi_cutout_dim",
            ]
        ]
        # for now just limit to 400,800
        arccnet_df = arccnet_df[arccnet_df["hmi_cutout_dim"] == (400, 800)]
        arccnet_df["hmi_cutout"] = arccnet_df["hmi_cutout"].apply(
            lambda path: make_relative(Path("/Users/pjwright/Documents/work/ARCCnet/"), path)
        )
        arccnet_df["Number"] = arccnet_df["Number"].astype("Int32")  # convert to Int with NaN, see SWPC
        logger.info(arccnet_df)
        arccnet_df.to_csv("/Users/pjwright/Documents/work/ARCCnet/data/04_final/AR-QS_classification.csv", index=False)

    # 3. Extract SHARP regions for AR Classification
    # !TODO ideally we want these SHARP around NOAA AR # along with classification in one df.
    # https://gist.github.com/PaulJWright/f9e12454db8d23a46d8bee153c8fbd3a
