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

from arccnet import config
from arccnet.data_generation.utils.data_logger import logger
from arccnet.data_generation.utils.utils import is_point_far_from_point, save_compressed_map

matplotlib.use("Agg")

__all__ = ["MagnetogramProcessor", "RegionExtractor"]  # , "ARDetection"]


class MagnetogramProcessor:
    """
    Process Magnetograms.

    This class provides methods to process magnetogram data using multiprocessing.
    """

    def __init__(
        self,
        csv_in_file: Path = Path(config["paths"]["mag_intermediate_hmimdi_data_csv"]),
        csv_out_file: Path = Path(config["paths"]["mag_intermediate_hmimdi_processed_data_csv"]),
        columns: list[str] = None,
        processed_data_dir: Path = Path(config["paths"]["mag_intermediate_data_dir"]),
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
        csv_file=Path(config["paths"]["mag_intermediate_hmimdi_data_csv"]),
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


class RegionBox:
    """
    Parameters
    ----------
    top_right
        pixel coordinates of the top right of the bounding box

    bottom_left
        pixel coordinates of the top right of the bounding box

    shape
        shape in pixels of the region

    ar_pos_pixels
        pixel coordinates of the active region centre

    identifier
        an identifier for the region, e.g. NOAA AR Number

    filepath
        filepath of the region

    """

    def __init__(
        self,
        top_right: tuple[float, float],
        bottom_left: tuple[float, float],
        shape: tuple[int, int],
        ar_pos_pixels=tuple[int, int],
        identifier=None,
        filepath=None,
    ):
        self.top_right = top_right
        self.bottom_left = bottom_left
        self.identifier = identifier
        self.shape = shape
        self.ar_pos_pixels = ar_pos_pixels
        self.filepath = filepath


class SRSBox(RegionBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class QSBox(RegionBox):
    def __init__(self, sunpy_map: sunpy.map.Map, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if sunpy.map.coordinate_is_on_solar_disk(sunpy_map.center):
            latlon = sunpy_map.center.transform_to(sunpy.coordinates.frames.HeliographicStonyhurst)
            self.center = sunpy_map.center
            self.latitude = latlon.lat.value
            self.longitude = latlon.lon.value
        else:
            self.center = np.nan
            self.latitude = np.nan
            self.longitude = np.nan


class RegionExtractor:
    def __init__(
        self,
        dataframe=Path(config["paths"]["mag_intermediate_hmimdi_processed_data_csv"]),
        out_fnames: list[str] = None,
        datetimes: list[str] = None,
        data_cols: list[str] = None,
        cutout_sizes: list[tuple] = None,
        common_datetime_col: str = None,
        num_random_attempts: int = 10,
    ) -> None:
        # validate these columns
        self.validate([out_fnames, datetimes, data_cols, cutout_sizes], length=2)
        # load to df and make datetime
        self.df = load_df_to_datetimedf(dataframe)
        dv_summary_plots_path = Path(config["paths"]["mag_processed_qssummaryplots_dir"])

        self.dataframes = []
        combined_indices = set()

        # split the dataframe, and columns into an iterable (list)
        for datetime_col, data_col, co_size, ofname in zip(datetimes, data_cols, cutout_sizes, out_fnames):
            # Create DataFrame for datetime_col and data_col not null
            df_sset = self.df[(self.df[datetime_col].notnull() & self.df[data_col].notnull())]
            combined_indices.update(df_sset.index)
            # df_sset = df_sset.reset_index(drop=True)
            self.dataframes.append((df_sset, datetime_col, data_col, co_size, ofname))
        # check that all indices in the original dataframe are accounted for
        if not set(self.df.index) == combined_indices:
            raise ValueError("there are missing rows")

        # empty dataframes to capture active region and quiet sun data
        activeregion_arr = []
        quietsun_arr = []

        # iterate through dataframes extracting AR info, and creating QS patches
        for single_df in self.dataframes:
            df_subset, datetime_column, data_column, cutout_size, instr = single_df

            # Active region dataframe is a copy of df_subset that is to be modified.
            ar_df = df_subset.copy(deep=True)

            # Create a new QuietSun dataframe with the df_subset column names
            # ideally need to drop most of the SRS columns
            qs_df = pd.DataFrame(columns=df_subset.columns.tolist())

            # ensure columns needed are in the df
            for c in ["top_right_" + instr, "bottom_left_" + instr, "cutout_path_" + instr, "cutout_dim_" + instr]:
                if c not in ar_df.columns:
                    ar_df[c] = None
                else:
                    raise ValueError("column already exists")

            xsize, ysize = cutout_size

            # group by the common datetime column, and iterate
            grouped_by_datetime = df_subset.groupby(common_datetime_col)
            for time_srs, group in grouped_by_datetime:
                summary_info = []

                if len(group[data_column].unique()) != 1:
                    raise ValueError("group[data_column].unique() is not 1")

                my_hmi_map = sunpy.map.Map(group[data_column].unique()[0])  # take the first hmi
                time_instr = group[datetime_column].unique()[0]  # Shane asked about just querying the map this.

                # set nan values in the map to zero
                # workaround for issues seen in processing
                data = my_hmi_map.data
                on_disk_nans = np.isnan(data).sum()

                if on_disk_nans > 0:
                    logger.warning(
                        f"There are {on_disk_nans} on-disk nans in this {time_srs} {my_hmi_map.instrument} map"
                    )
                    indices = np.where(np.isnan(data))
                    data[indices] = 0.0

                # -- AR Extraction
                for idx, row in group.iterrows():
                    # NOAA AR Number
                    numbr = row["Number"]
                    #
                    top_right, bottom_left, ar_pos_pixels = extract_region_lonlat(
                        my_hmi_map,
                        time_instr,
                        row["Latitude"] * u.deg,
                        row["Longitude"] * u.deg,
                        xsize=xsize * u.pix,
                        ysize=ysize * u.pix,  # units should be dealt with earlier
                    )

                    # cutout the active region
                    my_hmi_submap = my_hmi_map.submap(bottom_left, top_right=top_right)

                    path = (
                        Path(config["paths"]["mag_processed_fits_dir"])
                        / f"{time_srs.year}-{time_srs.month}-{time_srs.day}_{numbr}_{instr}.fits"
                    )

                    save_compressed_map(my_hmi_submap, path, overwrite=True)

                    srs_box_obj = SRSBox(
                        top_right=top_right,
                        bottom_left=bottom_left,
                        shape=my_hmi_submap.data.shape,
                        ar_pos_pixels=ar_pos_pixels,
                        identifier=numbr,
                        filepath=path,
                    )
                    summary_info.append(srs_box_obj)

                    # !TODO change this up
                    ar_df.at[idx, "top_right_" + instr] = (
                        srs_box_obj.top_right[0].value,
                        srs_box_obj.top_right[1].value,
                    )
                    ar_df.at[idx, "bottom_left_" + instr] = (
                        srs_box_obj.bottom_left[0].value,
                        srs_box_obj.bottom_left[1].value,
                    )
                    ar_df.at[idx, "cutout_path_" + instr] = srs_box_obj.filepath
                    ar_df.at[idx, "cutout_dim_" + instr] = srs_box_obj.shape
                    ar_df.at[idx, "num_ondisk_nans_" + instr] = on_disk_nans

                    del my_hmi_submap

                # -- QS Extraction
                iterations = 0
                qs_df_len = 0
                while qs_df_len < num_random_attempts and iterations <= 20:
                    # there may be an existing CS algo for this,
                    # it's essentially a simplified 2D bin packing problem,

                    # generate random lng/lat and convert Helioprojective coordinates to pixel coordinates
                    qs_center_hproj = SkyCoord(
                        random.uniform(-1000, 1000) * u.arcsec,
                        random.uniform(-1000, 1000) * u.arcsec,
                        frame=my_hmi_map.coordinate_frame,
                    ).to_pixel(my_hmi_map.wcs)

                    # check ar_pos_hproj is far enough from other vals
                    candidates = list(
                        map(
                            lambda v: is_point_far_from_point(
                                qs_center_hproj[0], qs_center_hproj[1], v[0], v[1], xsize * 1.01, ysize * 1.01
                            ),
                            [box_info.ar_pos_pixels for box_info in summary_info],
                        )
                    )

                    # if far enough away from all other values
                    if all(candidates):
                        # generate the submap
                        fd_bottom_left, fd_top_right = pixel_to_bboxcoords(
                            xsize * u.pix, ysize * u.pix, qs_center_hproj * u.pix
                        )
                        qs_submap = my_hmi_map.submap(bottom_left, top_right=top_right)

                        # save to file
                        output_filename = (
                            Path(config["paths"]["mag_processed_qsfits_dir"])
                            / f"{time_srs.year}-{time_srs.month}-{time_srs.day}_QS_{qs_df_len}_{instr}.fits"
                        )

                        # create QS BBox object
                        qs_region = QSBox(
                            sunpy_map=qs_submap,
                            top_right=fd_top_right,
                            bottom_left=fd_bottom_left,
                            shape=qs_submap.data.shape,
                            ar_pos_pixels=qs_center_hproj,
                            identifier=None,
                            filepath=output_filename,
                        )

                        # only keep those with the center on disk
                        if qs_region.center is np.nan:
                            continue

                        save_compressed_map(qs_submap, path=output_filename, overwrite=True)

                        summary_info.append(qs_region)

                        # create the dataframe for a single QS region
                        qs_temp = pd.DataFrame(
                            {
                                datetime_column: time_instr,
                                "datetime_srs": time_srs,
                                "cutout_path_" + instr: str(output_filename),
                                "cutout_dim_" + instr: [qs_region.shape],
                                "Longitude": qs_region.longitude,
                                "Latitude": qs_region.latitude,
                                "download_path_hmi": row["download_path_hmi"],
                                "downloaded_successfully_hmi": row["downloaded_successfully_hmi"],
                                "download_path_mdi": row["download_path_mdi"],
                                "downloaded_successfully_mdi": row["downloaded_successfully_mdi"],
                                "processed_download_path_hmi": row["processed_download_path_hmi"],
                                "processed_download_path_mdi": row["processed_download_path_mdi"],
                                "num_ondisk_nans_" + instr: on_disk_nans,
                            },
                            index=[0],
                        )
                        qs_df = pd.concat([qs_df, qs_temp], ignore_index=True)

                        del qs_submap  # unsure if necessary; was having memories issues
                        qs_df_len += 1
                        iterations += 1

                qs_df = qs_df.sort_values("datetime_srs").reset_index(drop=True)

                # plot and save QS + AR
                self.plotting(dv_summary_plots_path, instr, time_srs, summary_info, my_hmi_map, ysize)

            activeregion_arr.append(ar_df)
            quietsun_arr.append(qs_df)

        # Merge the two active region dataframes
        # set of common columns to drop for merging
        common_columns = activeregion_arr[0].columns.intersection(activeregion_arr[1].columns)
        # outer merge to keep rows where there is no HMI/MDI pairs
        result_df = self.df.merge(
            activeregion_arr[0].drop(columns=common_columns), how="outer", left_index=True, right_index=True
        )
        result_df = result_df.merge(
            activeregion_arr[1].drop(columns=common_columns), how="outer", left_index=True, right_index=True
        )
        if result_df[common_columns].equals(self.df):
            # only return if the dataframe
            self.activeregion_classification_df = result_df.copy()
        else:
            pd.testing.assert_frame_equal(result_df[common_columns], self.df)
            logger.warn("Unable to preserve the `pd.DataFrame` through splitting and recombination.")
            # raise ValueError

        # concatenate and save the Quiet Sun `pd.DataFrame`
        final_df = pd.concat(quietsun_arr)
        final_df.sort_values("datetime_srs", inplace=True)
        final_df.reset_index(inplace=True, drop=True)
        self.quietsun_df = final_df.copy()

    def validate(self, dataframe_cols, length: int):
        """
        Validates the input data to ensure uniformity of length and maximum length constraint.

        This method checks whether all input data columns in the given list have the same length
        and if none of the elements in any input column exceed the specified length constraint.

        Parameters
        ----------
        dataframe_cols : list
            A list of input data columns, each represented as an iterable (e.g., list, tuple).

        length : int
            Maximum allowed length for elements within the input data columns.

        Raises
        ------
        ValueError
            If the input data columns do not have the same length or if any element
            in the input data columns exceeds the specified length constraint.
        """
        input_lengths = [len(input_data) for input_data in dataframe_cols]
        if len(set(input_lengths)) > 1:
            raise ValueError("All inputs must be of the same length.")

        if any(len(item) > length for item in dataframe_cols):
            raise ValueError("None of the inputs should be longer than 2.")

    def plotting(self, summary_plot_path, instr, time_srs, summary_info, my_hmi_map, ysize) -> None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(projection=my_hmi_map)
        my_hmi_map.plot_settings["norm"].vmin = -1500
        my_hmi_map.plot_settings["norm"].vmax = 1500
        my_hmi_map.plot(axes=ax, cmap="hmimag")

        text_objects = []

        for box_info in summary_info:
            if isinstance(box_info, SRSBox):
                rectangle_cr = "red"
            elif isinstance(box_info, QSBox):
                rectangle_cr = "blue"
            else:
                raise ValueError("Unsupported box type")

            # deal with boxes off the edge
            my_hmi_map.draw_quadrangle(
                box_info.bottom_left,
                axes=ax,
                top_right=box_info.top_right,
                edgecolor=rectangle_cr,
                linewidth=1,
            )

            text = ax.text(
                box_info.ar_pos_pixels[0],
                box_info.ar_pos_pixels[1] + ysize / 2 + ysize / 10,
                box_info.identifier,
                **{"size": "x-small", "color": "black", "ha": "center"},
            )

            text_objects.append(text)

        logger.info(time_srs)
        plt.savefig(
            summary_plot_path / f"{time_srs.year}-{time_srs.month}-{time_srs.day}_{instr}.png",
            dpi=300,
        )
        plt.close("all")

        for text in text_objects:
            text.remove()


def load_df_to_datetimedf(filename: Path = None):
    """
    Load a CSV file into a DataFrame and convert columns with datetime prefix to datetime objects.

    Parameters
    ----------
    filename : Path or None
        Path to the CSV file to load. Default is None

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the loaded data with datetime columns converted to datetime objects.
    """
    if filename.exists():
        loaded_data = pd.read_csv(filename)

    datetime_columns = [col for col in loaded_data.columns if col.startswith("datetime")]
    for col in datetime_columns:
        loaded_data[col] = pd.to_datetime(loaded_data[col])

    return loaded_data


@u.quantity_input
def extract_region_lonlat(sunpy_map, time, lat: u.deg, lon: u.deg, xsize: u.pix, ysize: u.pix) -> sunpy.map.Map:
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
    tuple[float], tuple[float], tuple[float]]
        locations of the top right, bottom left and active region center
    """
    ar_pos_pixels = latlon_to_map_pixels(lat, lon, time, sunpy_map)
    bottom_left, top_right = pixel_to_bboxcoords(xsize, ysize, ar_pos_pixels * u.pix)

    return top_right, bottom_left, ar_pos_pixels


@u.quantity_input
def pixel_to_bboxcoords(xsize: u.pix, ysize: u.pix, box_center: u.pix):
    """
    Given the box center, and xsize, ysize, return the bottom left and top right coordinates in pixels
    """
    # remove u.pix
    xsize = xsize.value
    ysize = ysize.value
    box_center = box_center.value

    top_right = [box_center[0] + (xsize - 1) / 2, box_center[1] + (ysize - 1) / 2] * u.pix
    bottom_left = [
        box_center[0] - (xsize - 1) / 2,
        box_center[1] - (ysize - 1) / 2,
    ] * u.pix

    return bottom_left, top_right


@u.quantity_input
def latlon_to_map_pixels(
    latitude: u.deg,
    longitude: u.deg,
    time,
    sunpy_map: sunpy.map.Map,
    frame=sunpy.coordinates.frames.HeliographicStonyhurst,
):
    """
    Given lat/lon in degrees, convert to pixel locations
    """
    ar_pos_hgs = SkyCoord(
        longitude,
        latitude,
        obstime=time,
        frame=frame,
    )
    transformed = ar_pos_hgs.transform_to(sunpy_map.coordinate_frame)
    ar_pos_pixels = transformed.to_pixel(sunpy_map.wcs)
    return ar_pos_pixels


def map_pixels_to_latlon(sunpy_map: sunpy.map.Map):
    """
    provide pixels, get out latlon
    """
    pass
