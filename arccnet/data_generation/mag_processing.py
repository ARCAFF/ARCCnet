from pathlib import Path

import pandas as pd
from tqdm import tqdm

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
                file_path = Path(dv.MAG_INTERMEDIATE_DATA_DIR) / filename  # Join the path and filename
                paths.append(file_path)
        else:
            raise FileNotFoundError(f"{filename} does not exist. Try running `DataManager` first.")

        paths = paths[0:2]
        _ = self._process_data(paths)

    def _process_data(self, files) -> None:
        # !TODO find a good way to deal with the paths

        for file in tqdm(files, desc="Processing data", unit="file"):
            self._process_datum(file)

    def _process_datum(self, file) -> None:
        # !TODO find a good way to deal with the paths

        # 1. Load & Rotate
        map = self._load_datum(file)
        r_map = self._rotate_datum(map)

        # 2. Set a constant radius

        # 3. ...

        # 4. ...

        return r_map

    def _load_datum(self, file) -> sunpy.map.Map:
        """
        load single data
        """
        return sunpy.map.Map(file)

    def _rotate_datum(self, amap) -> None:
        """
        rotate a list of maps according to metadata

        e.g. before rotation a HMI map may have: `crota2 = 180.082565`
        """

        if "crota2" not in amap.meta:
            logger.info(f"The MetaDict for {amap.meta['t_rec']} does not have 'crota2' key.")

            rmap = amap.rotate()

        return rmap


if __name__ == "__main__":
    logger.info(f"Executing {__file__} as main program")
    _ = MagnetogramProcessor()
