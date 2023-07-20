from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import astropy.units as u
from astropy.coordinates import SkyCoord

import arccnet.data_generation.utils.default_variables as dv
import sunpy.map
from arccnet.data_generation.utils.data_logger import logger

__all__ = ["MagnetogramProcessor"]


class MagnetogramProcessor:
    """
    Process Magnetograms
    """

    def __init__(self) -> None:
        filename = Path(dv.MAG_INTERMEDIATE_DATA_CSV)

        if filename.exists():
            self.loaded_data = pd.read_csv(filename)
            file_list = list(self.loaded_data.url_hmi.dropna().unique()) + list(
                self.loaded_data.url_mdi.dropna().unique()
            )
            paths = []
            for url in file_list:
                filename = Path(url).name  # Extract the filename from the URL
                file_path = Path(dv.MAG_RAW_DATA_DIR) / filename  # Join the path and filename
                paths.append(file_path)
        else:
            raise FileNotFoundError(f"{filename} does not exist. Try running `DataManager` first.")

        base_directory_path = Path(dv.MAG_INTERMEDIATE_DATA_DIR)
        if not base_directory_path.exists():
            base_directory_path.mkdir(parents=True)

        _ = self._process_data(paths)

    def _process_data(self, files) -> None:
        # !TODO find a good way to deal with the paths

        for file in tqdm(files, desc="Processing data", unit="file"):
            processed_data = self._process_datum(file)
            # !TODO probably append the name with something
            processed_data.save(Path(dv.MAG_INTERMEDIATE_DATA_DIR) / file.name, overwrite=True)

    def _process_datum(self, file) -> None:
        # 1. Load & Rotate
        map = sunpy.map.Map(file)
        r_map = self._rotate_datum(map)

        # 2. set data off-disk to 0 (np.nan would be ideal, but deep learning)
        r_map.data[~sunpy.map.coordinate_is_on_solar_disk(sunpy.map.all_coordinates_from_map(r_map))] = 0

        # !TODO
        # 3. normalise radius to fixed value
        # 4. ...

        return r_map

    def _rotate_datum(self, amap: sunpy.map.Map) -> sunpy.map.Map:
        """
        rotate a list of maps according to metadata

        e.g. before rotation a HMI map may have: `crota2 = 180.082565`
        """

        if "crota2" not in amap.meta:
            logger.info(f"The MetaDict for {amap.meta['t_rec']} does not have 'crota2' key.")

        rmap = amap.rotate()

        return rmap


class ARExtractor:
    def __init__(self) -> None:
        filename = Path(dv.MAG_INTERMEDIATE_DATA_CSV)

        if filename.exists():
            self.loaded_data = pd.read_csv(filename)

        columns_to_update = ["url_hmi", "url_mdi"]
        new_columns = ["processed_hmi", "processed_mdi"]

        # !TODO replace with default_variables.py
        dv_base_path = Path("/Users/pjwright/Documents/work/ARCCnet/data/02_intermediate/mag/fits/")
        dv_process_path = Path("/Users/pjwright/Documents/work/ARCCnet/data/03_processed/mag/fits/")

        # Iterate through the columns and update the paths
        for old_column, new_column in zip(columns_to_update, new_columns):
            self.loaded_data[new_column] = self.loaded_data[old_column].map(
                lambda x: dv_base_path / Path(x).name if pd.notna(x) else x
            )

        # set empty list of cutout for hmi
        cutout_list_hmi = []
        for index, row in self.loaded_data.iterrows():
            if self.loaded_data.iloc[index]["processed_hmi"] is np.nan:
                cutout_list_hmi.append(np.nan)
            else:
                lat = self.loaded_data.iloc[index]["Latitude"]
                lng = self.loaded_data.iloc[index]["Longitude"]
                hmi = self.loaded_data.iloc[index]["processed_hmi"]
                time = self.loaded_data.iloc[index]["datetime_hmi"]

                dt = self.loaded_data.iloc[index]["datetime_srs"]
                numbr = self.loaded_data.iloc[index]["Number"]

                my_hmi_map = sunpy.map.Map(hmi)

                _cd = SkyCoord(
                    lng * u.deg,
                    lat * u.deg,
                    obstime=time,
                    frame=sunpy.coordinates.frames.HeliographicStonyhurst,
                )

                transformed = _cd.transform_to(my_hmi_map.coordinate_frame)

                tr_x = transformed.helioprojective.Tx + 200 * u.arcsec
                tr_y = transformed.helioprojective.Ty + 100 * u.arcsec

                bl_x = transformed.helioprojective.Tx - 200 * u.arcsec
                bl_y = transformed.helioprojective.Ty - 100 * u.arcsec

                top_right = SkyCoord(tr_x, tr_y, frame=my_hmi_map.coordinate_frame)
                bottom_left = SkyCoord(bl_x, bl_y, frame=my_hmi_map.coordinate_frame)
                my_hmi_submap = my_hmi_map.submap(bottom_left, top_right=top_right)

                print(dv_process_path / f"{dt}_{numbr}.fits")
                my_hmi_submap.save(dv_process_path / f"{dt}_{numbr}.fits", overwrite=True)

                cutout_list_hmi.append(dv_process_path / f"{dt}_{numbr}.fits")

        self.loaded_data["hmi_cutout"] = cutout_list_hmi
        self.loaded_data.to_csv(Path("/Users/pjwright/Documents/work/ARCCnet/data/03_processed/mag/") / "processed.csv")


if __name__ == "__main__":
    logger.info(f"Executing {__file__} as main program")
    _ = MagnetogramProcessor()
    _ = ARExtractor()
