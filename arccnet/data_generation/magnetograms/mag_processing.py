from pathlib import Path

import pandas as pd
import tqdm

import arccnet.data_generation.utils.default_variables as dv
import sunpy.map
from arccnet.data_generation.utils.data_logger import logger


class MagProcessor:
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
                filename = url.split("/")[-1]  # Extract the filename from the URL
                file_path = Path(dv.MAG_INTERMEDIATE_DATA_DIR) / filename  # Join the path and filename
                paths.append(file_path)
        else:
            raise FileNotFoundError(f"{filename} does not exist. Try running `DataManager` first.")

        _ = self.process_data(paths)

    def process_data(self, files) -> None:
        # !TODO find a good way to deal with the paths

        # 1. Load & Rotate
        maps = self.load_data(files)

        # 2. Fix radius
        r_maps = self.rotate_data(maps)

        # 3. ...

        # 4. ...

        return r_maps

    def load_data(self, files) -> None:
        """
        load all data from a list
        """
        mps = []
        for file in tqdm(files, desc="Loading data", unit="file"):
            mps.append(self._load_datum(file))

        return mps

    def _load_datum(self, file) -> sunpy.map.Map:
        """
        load single data
        """
        return sunpy.map.Map(file)

    def rotate_data(self, data_maps) -> None:
        """
        rotate a list of maps according to metadata

        e.g. before rotation a HMI map may have: `crota2 = 180.082565`
        """
        r_maps = []
        for amap in tqdm(data_maps, desc="rotating maps", unit="map"):
            if "crota2" not in amap.meta:
                logger.info(f"The MetaDict for {amap.meta['t_rec']} does not have 'crota2' key.")

            r_maps.append(self._rotate_datum(amap))

        return r_maps

    def _rotate_datum(self, smap) -> sunpy.map.Map:
        """
        rotate single data
        """
        return smap.rotate()
