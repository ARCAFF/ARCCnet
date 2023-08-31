import random
import multiprocessing
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import pandas as pd
import sunpy.map
from tqdm import tqdm

import astropy.units as u
from astropy.coordinates import SkyCoord

import arccnet.data_generation.utils.default_variables as dv
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


class SRSBox:
    def __init__(
        self,
        top_right: tuple[float, float],
        bottom_left: tuple[float, float],
        shape: tuple[int, int],
        ar_pos_pixels=tuple[int, int],
        time: str = None,
        identifier=None,
    ):
        self.top_right = top_right
        self.bottom_left = bottom_left
        self.identifier = identifier
        self.shape = shape
        self.ar_pos_pixels = ar_pos_pixels
        self.time = time


class QSBox:
    def __init__(
        self,
        sunpy_map: sunpy.map.Map,
        top_right: tuple[float, float],
        bottom_left: tuple[float, float],
        shape: tuple[int, int],
        ar_pos_pixels: tuple[int, int],
        identifier=None,
    ):
        self.top_right = top_right
        self.bottom_left = bottom_left
        self.identifier = identifier
        self.shape = shape
        self.ar_pos_pixels = ar_pos_pixels

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
        dataframe=Path(dv.MAG_INTERMEDIATE_HMIMDI_PROCESSED_DATA_CSV),
        out_fnames: list[str] = ["mdi", "hmi"],
        datetimes: list[str] = ["datetime_mdi", "datetime_hmi"],
        data_cols: list[str] = ["processed_download_path_mdi", "processed_download_path_hmi"],
        new_cols: list[str] = ["cutout_mdi", "cutout_hmi"],
        cutout_sizes: list[tuple] = [
            (int(dv.X_EXTENT / 4), int(dv.Y_EXTENT / 4)),
            (int(dv.X_EXTENT), int(dv.Y_EXTENT)),
        ],
        num_random_attempts: int = 10,
    ) -> None:
        common_datetime_col = "datetime_srs"

        # load to df and make datetime
        df = load_df_to_datetime(dataframe)

        self.dataframes = []
        combined_indices = set()
        for datetime_col, data_col, new_col, cutout_size, ofname in zip(
            datetimes, data_cols, new_cols, cutout_sizes, out_fnames
        ):
            # Create DataFrame for datetime_col and data_col not null
            df_subset = df[(df[datetime_col].notnull() & df[data_col].notnull())]
            combined_indices.update(df_subset.index)
            df_subset = df_subset.reset_index(drop=True)
            self.dataframes.append((df_subset, datetime_col, data_col, new_col, cutout_size, ofname))

        # check that all indices are accounted for
        if not set(df.index) == combined_indices:
            raise ValueError("there are missing rows")

        # ---
        dv_summary_plots_path = Path(dv.MAG_PROCESSED_QSSUMMARYPLOTS_DIR)
        df_arr = []

        # iterate through dataframes
        for single_df in self.dataframes:
            df_subset, datetime_column, data_column, new_column, cutout_size, instr = single_df
            qs_df = df_subset.copy()
            xsize, ysize = cutout_size
            print("cuoutsize", cutout_size)
            grouped_by_datetime = df_subset.groupby(common_datetime_col)

            for time_srs, group in grouped_by_datetime:
                summary_info = []

                if len(group[data_column].unique()) != 1:
                    raise ValueError("group[data_column].unique() is not 1")

                my_hmi_map = sunpy.map.Map(group[data_column].unique()[0])  # take the first hmi
                time_hmi = group[datetime_column].unique()[0]  # I think shane asked about just querying the map

                # set nan values to zero
                data = my_hmi_map.data
                nans = np.isnan(data).sum()
                if nans > 0:
                    logger.warning(f"warning ... there are {nans} nans in this {my_hmi_map.instrument} map")
                    indices = np.where(np.isnan(data))
                    data[indices] = 0.0

                # -- AR Extraction
                columns = ["Latitude", "Longitude"]
                for _, row in group.iterrows():
                    numbr = row["Number"]

                    my_hmi_submap, top_right, bottom_left, ar_pos_pixels = extract_region_lonlat(
                        my_hmi_map,
                        time_hmi,
                        row[columns][0] * u.deg,
                        row[columns][1] * u.deg,
                        xsize=xsize * u.pix,
                        ysize=ysize * u.pix,  # units should be dealt with earlier
                    )
                    summary_info.append(
                        SRSBox(
                            top_right=top_right,
                            bottom_left=bottom_left,
                            identifier=numbr,
                            shape=my_hmi_submap.data.shape,
                            ar_pos_pixels=ar_pos_pixels,
                            time=time_srs,
                        )
                    )

                    # save
                    save_compressed_map(
                        my_hmi_submap,
                        Path(dv.MAG_PROCESSED_FITS_DIR)
                        / f"{time_srs.year}-{time_srs.month}-{time_srs.day}_{numbr}_{instr}.fits",
                        overwrite=True,
                    )
                    del my_hmi_submap

                # -- QS Extraction
                iterations = 0
                qs_df_len = 0
                while qs_df_len < num_random_attempts and iterations <= 20:
                    # there may be an existing CS algo for this,
                    # it's essentially a simplified 2D bin packing problem,

                    # generate random lng/lat and convert Helioprojective coordinates to pixel coordinates
                    ar_pos_hproj = SkyCoord(
                        random.uniform(-1000, 1000) * u.arcsec,
                        random.uniform(-1000, 1000) * u.arcsec,
                        frame=my_hmi_map.coordinate_frame,
                    ).to_pixel(my_hmi_map.wcs)

                    # check ar_pos_hproj is far enough from other vals
                    candidates = list(
                        map(
                            lambda v: is_point_far_from_point(
                                ar_pos_hproj[0], ar_pos_hproj[1], v[0], v[1], xsize * 1.01, ysize * 1.01
                            ),
                            [box_info.ar_pos_pixels for box_info in summary_info],
                        )
                    )

                    if all(candidates):
                        # generate the submap
                        qs_submap, fd_top_right, fd_bottom_left = extract_submap_pixels(
                            my_hmi_map, ar_pos_hproj, xsize, ysize
                        )

                        # create QS BBox object
                        qs_region = QSBox(
                            sunpy_map=qs_submap,
                            top_right=fd_top_right,
                            bottom_left=fd_bottom_left,
                            identifier=None,
                            shape=qs_submap.data.shape,
                            ar_pos_pixels=ar_pos_hproj,
                        )

                        # only keep those with the center on disk
                        if qs_region.center is np.nan:
                            continue

                        summary_info.append(qs_region)

                        # save to file
                        output_filename = (
                            Path(dv.MAG_PROCESSED_QSFITS_DIR)
                            / f"{time_srs.year}-{time_srs.month}-{time_srs.day}_QS_{qs_df_len}_{instr}.fits"
                        )

                        save_compressed_map(qs_submap, path=output_filename, overwrite=True)

                        qs_temp = pd.DataFrame(
                            {
                                datetime_column: time_hmi,
                                "datetime_srs": time_srs,
                                new_column: str(output_filename),
                                new_column + "_dim": [qs_region.shape],
                                "Longitude": qs_region.longitude,
                                "Latitude": qs_region.latitude,
                            },
                            index=[0],
                        )

                        qs_df = pd.concat([qs_df, qs_temp], ignore_index=True)
                        del qs_submap

                        qs_df_len += 1
                        iterations += 1

                # !TODO add lat/lon for QS, add hmi_cutout_dim to SRS
                qs_df = qs_df.sort_values("datetime_srs").reset_index(drop=True)

                self.plotting(dv_summary_plots_path, instr, time_srs, summary_info, my_hmi_map, ysize)

                del summary_info

            df_arr.append(qs_df)

    def plotting(self, dv_summary_plots_path, instr, time_srs, summary_info, my_hmi_map, ysize) -> None:
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
            dv_summary_plots_path / f"{time_srs.year}-{time_srs.month}-{time_srs.day}_{instr}.png",
            dpi=300,
        )
        plt.close("all")

        for text in text_objects:
            text.remove()


def load_df_to_datetime(filename: Path = None):
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


def extract_submap_pixels(amap, center, xsize, ysize) -> sunpy.map.Map:
    """
    ...
    """
    # Perform in pixel coordinates
    top_right = [center[0] + (xsize - 1) / 2, center[1] + (ysize - 1) / 2] * u.pix
    bottom_left = [
        center[0] - (xsize - 1) / 2,
        center[1] - (ysize - 1) / 2,
    ] * u.pix

    submap = amap.submap(bottom_left, top_right=top_right)
    return submap, top_right, bottom_left


@u.quantity_input
def latlon_to_map_pixels(
    latitude: u.deg, longitude: u.deg, time, amap: sunpy.map.Map, frame=sunpy.coordinates.frames.HeliographicStonyhurst
):
    ar_pos_hgs = SkyCoord(
        longitude,
        latitude,
        obstime=time,
        frame=frame,
    )
    transformed = ar_pos_hgs.transform_to(amap.coordinate_frame)
    ar_pos_pixels = transformed.to_pixel(amap.wcs)
    return ar_pos_pixels


def map_pixels_to_latlon(time, amap: sunpy.map.Map):
    """
    provide pixels, get out latlon"""
    pass


@u.quantity_input
def pixel_to_bboxcoords(xsize: u.pix, ysize: u.pix, box_center: u.pix):
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
def extract_region_lonlat(amap, time, lat: u.deg, lon: u.deg, xsize: u.pix, ysize: u.pix) -> sunpy.map.Map:
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
    ar_pos_pixels = latlon_to_map_pixels(lat, lon, time, amap)
    bottom_left, top_right = pixel_to_bboxcoords(xsize, ysize, ar_pos_pixels * u.pix)
    submap = amap.submap(bottom_left, top_right=top_right)

    return submap, top_right, bottom_left, ar_pos_pixels
