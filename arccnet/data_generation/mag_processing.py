import random
import multiprocessing
from pathlib import Path
from dataclasses import dataclass

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
from arccnet.data_generation.utils.utils import make_relative, save_compressed_map

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


def file_stuff():
    dv_process_fits_path = Path(dv.MAG_PROCESSED_FITS_DIR)
    if not dv_process_fits_path.exists():
        dv_process_fits_path.mkdir(parents=True)

    dv_summary_plots_path = Path(dv.MAG_PROCESSED_SUMMARYPLOTS_DIR)
    if not dv_summary_plots_path.exists():
        dv_summary_plots_path.mkdir(parents=True)

    dir = Path(dv.MAG_PROCESSED_QSFITS_DIR)
    if not dir.exists():
        dir.mkdir(parents=True)

    dv_qs_summary_plots_path = Path(dv.MAG_PROCESSED_QSSUMMARYPLOTS_DIR)
    if not dv_summary_plots_path.exists():
        dv_summary_plots_path.mkdir(parents=True)

    return dv_process_fits_path, dv_summary_plots_path, dv_qs_summary_plots_path


@dataclass
class SRSBox:
    top_right: tuple[float, float]
    bottom_left: tuple[float, float]
    identifier: int
    shape: tuple[int, int]
    ar_pos_pixels: tuple[int, int]
    time: str


@dataclass
class QSBox:
    top_right: tuple[float, float]
    bottom_left: tuple[float, float]
    identifier: None
    shape: tuple[int, int]
    ar_pos_pixels: tuple[int, int]


#     _latlon: tuple(float)

#     @property
#     def latlon(self):
#         return self._latlon

#     @latlon.setter
#     def latlon(self, value):
#         self._ar_pos_pixels = value

# qs_submap.center.transform_to(sunpy.coordinates.frames.HeliographicStonyhurst)


class RegionExtractor:
    def __init__(self) -> None:
        # Load minimal data
        num_random_attempts = 10

        df = load_minimal_srs_mag_df(Path(dv.MAG_INTERMEDIATE_DATA_CSV))

        # Create DataFrame for datetime_hmi and processed_hmi not null
        df_hmi = df[(df["datetime_hmi"].notnull() & df["processed_hmi"].notnull())]

        # Create DataFrame for datetime_mdi and processed_mdi not null
        df_mdi = df[(df["datetime_mdi"].notnull() & df["processed_mdi"].notnull())]

        # Create DataFrame for rows not in df_hmi or df_mdi
        df_not_in_either = df[~df.index.isin(df_hmi.index) & ~df.index.isin(df_mdi.index)]

        # Reset the index if needed for all DataFrames
        df_hmi.reset_index(drop=True, inplace=True)
        df_mdi.reset_index(drop=True, inplace=True)
        df_not_in_either.reset_index(drop=True, inplace=True)

        # Print the filtered DataFrames
        logger.info(f"DataFrame {df.shape}")
        logger.info(f"DataFrame for HMI: {df_hmi.shape}")
        logger.info(f"DataFrame for MDI: {df_mdi.shape}")
        logger.info(f"DataFrame not in either: {df_not_in_either.shape}")
        if df_not_in_either.shape[0] > 0:
            raise ValueError

        dv_process_fits_path, dv_summary_plots_path, _ = file_stuff()

        sizes = {
            "HMI SIDE1": (800, 400),  # ~0.5 arcsec per pixel
            "MDI": (200, 100),  # ~ 2 arcsec per pixel
        }

        instruments = {
            "MDI": {
                "df": df_mdi,
                "cols": ["datetime_mdi", "processed_mdi"],
                "newc": ["cutout_mdi", "cutout_dim_mdi"],
            },
            "HMI": {
                "df": df_hmi,
                "cols": ["datetime_hmi", "processed_hmi"],
                "newc": ["cutout_hmi", "cutout_dim_hmi"],
            },
        }

        df_arr = []
        for instr in instruments:
            grouped_data = instruments[instr]["df"].groupby("datetime_srs")
            datetime_column = instruments[instr]["cols"][0]
            data_column = instruments[instr]["cols"][1]
            newc0 = instruments[instr]["newc"][0]
            newc1 = instruments[instr]["newc"][1]

            qs_df = instruments[instr]["df"].copy()
            for time_srs, group in grouped_data:
                summary_info = []

                if len(group.processed_hmi.unique()) > 1:
                    raise ValueError()

                my_hmi_map = sunpy.map.Map(group[data_column].unique()[0])  # take the first hmi
                time_hmi = group[datetime_column].unique()[0]  # I think shane asked about just querying the map

                # set nan values to zero
                data = my_hmi_map.data
                nans = np.isnan(data).sum()
                if nans > 0:
                    logger.warning(f"warning ... there are {nans} nans in this {my_hmi_map.instrument} map")
                    indices = np.where(np.isnan(data))
                    data[indices] = 0.0

                # set instrument
                instrument = my_hmi_map.instrument
                try:
                    xsize, ysize = sizes[instrument]
                except KeyError:
                    raise ValueError(f"Size not defined for instrument: {instrument}")

                # -- AR Extraction
                columns = ["Latitude", "Longitude"]
                for _, row in group.iterrows():
                    numbr = row["Number"]

                    my_hmi_submap, top_right, bottom_left, ar_pos_pixels = extract_region_lonlat(
                        my_hmi_map, time_hmi, row[columns], xsize=xsize, ysize=ysize
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
                    save_compressed_map(
                        my_hmi_submap, dv_process_fits_path / f"{time_srs}_{numbr}_{instr}.fits", overwrite=True
                    )
                    del my_hmi_submap

                # -- QS Extraction
                for i in range(0, num_random_attempts):
                    # there may be an existing CS algo for this,
                    # it's essentially a simplified 2D bin packing problem,
                    # however doing it randomly might be preferred.
                    random_1ng = random.uniform(-1000, 1000) * u.arcsec
                    random_lat = random.uniform(-1000, 1000) * u.arcsec

                    # convert lng/lat in Helioprojective coordinates to pixel coordinates
                    ar_pos_hproj = SkyCoord(
                        random_1ng,
                        random_lat,
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
                        qs_submap, top_right, bottom_left = extract_submap_pixels(
                            my_hmi_map, ar_pos_hproj, xsize, ysize
                        )

                        # create QS BBox object
                        # make more elegant
                        qs_region = QSBox(
                            top_right=top_right,
                            bottom_left=bottom_left,
                            identifier=None,
                            shape=qs_submap.data.shape,
                            ar_pos_pixels=ar_pos_hproj,
                        )
                        summary_info.append(qs_region)

                        # save to file
                        output_filename = (
                            Path(dv.MAG_PROCESSED_QSFITS_DIR)
                            / f"{time_srs.year}-{time_srs.month}-{time_srs.day}_QS_{i}_{instr}.fits"
                        )
                        save_compressed_map(qs_submap, path=output_filename, overwrite=True)

                        # create dummy dict
                        # why does this destroy the fucking dtype of cols

                        # Fix this to be in the class
                        if sunpy.map.coordinate_is_on_solar_disk(qs_submap.center):
                            qs_coords = qs_submap.center.transform_to(sunpy.coordinates.frames.HeliographicStonyhurst)
                            qs_temp = pd.DataFrame(
                                {
                                    "datetime_hmi": time_hmi,
                                    "datetime_srs": time_srs,
                                    newc0: str(output_filename),
                                    newc1: [qs_region.shape],
                                    "Longitude": qs_coords.lon.value,
                                    "Latitude": qs_coords.lat.value,
                                },
                                index=[0],
                            )
                        else:
                            qs_temp = pd.DataFrame(
                                {
                                    "datetime_hmi": time_hmi,
                                    "datetime_srs": time_srs,
                                    newc0: str(output_filename),
                                    newc1: [qs_region.shape],
                                    "Longitude": pd.NA,
                                    "Latitude": pd.NA,
                                },
                                index=[0],
                            )

                        # honestly, we should maybe just export the full qs_df without concat, and do that later
                        qs_df = pd.concat([qs_df, qs_temp], ignore_index=True)
                        del qs_submap

                    # return qs_df containing all quiet sun regions for an instrument

                # !TODO add lat/lon for QS, add hmi_cutout_dim to SRS
                qs_df = qs_df.sort_values("datetime_srs").reset_index(drop=True)

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

                del summary_info

            df_arr.append(qs_df)


def load_minimal_srs_mag_df(filename: Path = Path(dv.MAG_INTERMEDIATE_DATA_CSV)):
    """
    load filename
    """
    if filename.exists():
        loaded_data = pd.read_csv(filename)

    columns_to_update = ["url_hmi", "url_mdi"]
    new_columns = ["processed_hmi", "processed_mdi"]

    dv_base_path = Path(dv.MAG_INTERMEDIATE_DATA_DIR)

    # Iterate through the columns and update the paths
    # !TODO deal with earlier on in the codebase
    for old_column, new_column in zip(columns_to_update, new_columns):
        loaded_data[new_column] = loaded_data[old_column].map(
            lambda x: dv_base_path / Path(x).name if pd.notna(x) else x
        )

    # only return a subset of the data. At this point it should all be matched
    loaded_data = loaded_data[
        [
            "datetime_srs",
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
            "filepath_srs",  # SRS
            "datetime_mdi",
            "processed_mdi",  # MDI
            "datetime_hmi",
            "processed_hmi",  # HMI
        ]
    ]

    loaded_data["datetime_srs"] = pd.to_datetime(loaded_data["datetime_srs"])
    loaded_data["datetime_hmi"] = pd.to_datetime(loaded_data["datetime_hmi"])
    loaded_data["datetime_mdi"] = pd.to_datetime(loaded_data["datetime_mdi"])

    return loaded_data


def extract_submap_latlon(amap, time, coords, xsize=dv.X_EXTENT, ysize=dv.Y_EXTENT) -> sunpy.map.Map:
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
    #     logger.info(f">> {map.date}, {time}")

    lat, lng = coords

    ar_pos_hgs = SkyCoord(
        lng * u.deg,
        lat * u.deg,
        obstime=time,
        frame=sunpy.coordinates.frames.HeliographicStonyhurst,
    )

    transformed = ar_pos_hgs.transform_to(map.coordinate_frame)
    ar_pos_pixels = transformed.to_pixel(map.wcs)

    submap, top_right, bottom_left = extract_submap_pixels(map, ar_pos_pixels, xsize, ysize)
    # Perform in pixel coordinates
    top_right = [ar_pos_pixels[0] + (xsize - 1) / 2, ar_pos_pixels[1] + (ysize - 1) / 2] * u.pix
    bottom_left = [
        ar_pos_pixels[0] - (xsize - 1) / 2,
        ar_pos_pixels[1] - (ysize - 1) / 2,
    ] * u.pix

    submap = amap.submap(bottom_left, top_right=top_right)

    return submap, top_right, bottom_left, ar_pos_pixels


def extract_submap_pixels(amap, center, xsize, ysize) -> sunpy.map.Map:
    """ """
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


def extract_region_lonlat(amap, time, coords, xsize, ysize) -> sunpy.map.Map:
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
    #     logger.info(f">> {amap.date}, {time}")

    lat, lng = coords

    ar_pos_pixels = latlon_to_map_pixels(lat * u.deg, lng * u.deg, time, amap)
    bottom_left, top_right = pixel_to_bboxcoords(xsize * u.pix, ysize * u.pix, ar_pos_pixels * u.pix)
    submap = amap.submap(bottom_left, top_right=top_right)

    return submap, top_right, bottom_left, ar_pos_pixels


def make_relative(base_path, path):
    return Path(path).relative_to(Path(base_path))


def is_point_far_from_point(x, y, x1, y1, threshold_x, threshold_y):
    return abs(x - x1) > abs(threshold_x) or abs(y - y1) > abs(threshold_y)


def save_compressed_map(map: sunpy.map.Map, path: Path, **kwargs) -> None:
    """
    Save a compressed map.

    If "bscale" and "bzero" exist in the metadata, remove before saving.
    See: https://github.com/sunpy/sunpy/issues/7139

    Parameters
    ----------
    map : sunpy.map.Map
        the sunpy map object to be saved

    path : Path
        the path to save the file to

    Returns
    -------
    None
    """
    if "bscale" in map.meta:
        del map.meta["bscale"]

    if "bzero" in map.meta:
        del map.meta["bzero"]

    map.save(path, hdu_type=astropy.io.fits.CompImageHDU, **kwargs)


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
        _ = RegionExtractor()
        # ar_df = ARExtractor()
        # qs_df = QSExtractor()
        # arccnet_df = pd.concat([ar_df.data, qs_df.data], ignore_index=True).sort_values(
        #     by="datetime_hmi", ignore_index=True
        # )
        # arccnet_df = arccnet_df[
        #     [
        #         "datetime_srs",
        #         "Latitude",
        #         "Longitude",
        #         "Number",
        #         "Area",
        #         "Z",
        #         "Mag Type",
        #         "datetime_hmi",
        #         "hmi_cutout",
        #         "hmi_cutout_dim",
        #     ]
        # ]
        # # for now just limit to 400,800
        # arccnet_df = arccnet_df[arccnet_df["hmi_cutout_dim"] == (400, 800)]
        # arccnet_df["hmi_cutout"] = arccnet_df["hmi_cutout"].apply(
        #     lambda path: make_relative(Path("/Users/pjwright/Documents/work/ARCCnet/"), path)
        # )
        # arccnet_df["Number"] = arccnet_df["Number"].astype("Int32")  # convert to Int with NaN, see SWPC
        # logger.info(arccnet_df)
        # arccnet_df.to_csv("/Users/pjwright/Documents/work/ARCCnet/data/04_final/AR-QS_classification.csv", index=False)

    # 3. Extract SHARP regions for AR Classification
    # !TODO ideally we want these SHARP around NOAA AR # along with classification in one df.
    # https://gist.github.com/PaulJWright/f9e12454db8d23a46d8bee153c8fbd3a
