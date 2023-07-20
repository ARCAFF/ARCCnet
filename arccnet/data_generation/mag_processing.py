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

        # !TODO ...
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


if __name__ == "__main__":
    logger.info(f"Executing {__file__} as main program")
    _ = MagnetogramProcessor()
