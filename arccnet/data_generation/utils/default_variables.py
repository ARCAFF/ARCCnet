from arccnet.catalogs.active_regions import HALE_CLASSES, MCINTOSH_CLASSES

NOAA_SRS_ID_DICT = {
    "I": "Regions with Sunspots",
    "IA": "H-alpha Plages without Spots",
    "II": "Regions Due to Return",
}

VALID_SRS_VALUES = {
    "Mag Type": HALE_CLASSES,
    "Z": MCINTOSH_CLASSES,  # https://www.cv-helios.net/cvzmval.html
    "ID": list(NOAA_SRS_ID_DICT.keys()),  # ["I"],  # , "IA", "II"]
}

SRS_FILEPATHS_IGNORED = ["19961209SRS.txt"]
