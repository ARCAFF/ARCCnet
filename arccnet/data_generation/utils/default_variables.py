import os
import subprocess
from datetime import datetime

# ---- JSOC
JSOC_DEFAULT_EMAIL = "pjwright@stanford.edu"
JSOC_DATE_FORMAT = "%Y.%m.%d_%H:%M:%S"
JSOC_BASE_URL = "http://jsoc.stanford.edu"

# -- Magnetograms
MDI_DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
MDI_SEG_COL = "data"
HMI_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"
HMI_SEG_COL = "magnetogram"

# ---- Observation Dates
DATA_START_TIME = datetime(1995, 1, 1, 0, 0, 0)
DATA_END_TIME = datetime(2022, 12, 31, 0, 0, 0)
# SRS data is given at 00:30, valid at 00:00

# -- Valid Classes
# !TODO move these to a yaml or other option.
HALE_CLASSES = [
    "Alpha",
    "Beta",
    "Beta-Gamma",
    "Gamma",
    "Beta-Delta",
    "Beta-Gamma-Delta",
    "Gamma-Delta",
]

MCINTOSH_CLASSES = [
    "Axx",
    "Bxo",
    "Bxi",
    "Hrx",
    "Cro",
    "Dro",
    "Ero",
    "Fro",
    "Cri",
    "Dri",
    "Eri",
    "Fri",
    "Hsx",
    "Cso",
    "Dso",
    "Eso",
    "Fso",
    "Csi",
    "Dsi",
    "Esi",
    "Fsi",
    "Hax",
    "Cao",
    "Dao",
    "Eao",
    "Fao",
    "Cai",
    "Dai",
    "Eai",
    "Fai",
    "Dsc",
    "Esc",
    "Fsc",
    "Dac",
    "Eac",
    "Fac",
    "Hhx",
    "Cho",
    "Dho",
    "Eho",
    "Fho",
    "Chi",
    "Dhi",
    "Ehi",
    "Fhi",
    "Hkx",
    "Cko",
    "Dko",
    "Eko",
    "Fko",
    "Cki",
    "Dki",
    "Eki",
    "Fki",
    "Dhc",
    "Ehc",
    "Fhc",
    "Dkc",
    "Ekc",
    "Fkc",
]

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

# --- Data-related

BASE_DIR = (
    subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode().strip()
)  #!TODO change to config; and assume current working dir.

DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_DIR_RAW = os.path.join(DATA_DIR, "raw")
DATA_DIR_INTERMEDIATE = os.path.join(DATA_DIR, "intermediate")
DATA_DIR_LOGS = os.path.join(DATA_DIR, "logs")

DATA_LOGFILE = os.path.join(
    DATA_DIR_LOGS,
    f"data_processing_{datetime.utcnow().strftime('%Y_%m_%d_%H%M%S')}.log",
)

NOAA_SRS_DIR = os.path.join(DATA_DIR_RAW, "noaa_srs")
NOAA_SRS_TEXT_DIR = os.path.join(NOAA_SRS_DIR, "txt")
NOAA_SRS_TEXT_EXCEPT_DIR = os.path.join(NOAA_SRS_DIR, "txt_load_error")
NOAA_SRS_RAW_DATA_CSV = os.path.join(NOAA_SRS_DIR, "raw_data.csv")
NOAA_SRS_RAW_DATA_EXCEPT_CSV = os.path.join(NOAA_SRS_DIR, "raw_data_load_error.csv")

NOAA_SRS_RAW_DATA_HTML = os.path.join(NOAA_SRS_DIR, "raw_data.html")
NOAA_SRS_RAW_DATA_EXCEPT_HTML = os.path.join(NOAA_SRS_DIR, "raw_data_load_error.html")

NOAA_SRS_INTERMEDIATE_DIR = os.path.join(DATA_DIR_INTERMEDIATE, "noaa_srs")
NOAA_SRS_INTERMEDIATE_DATA_CSV = os.path.join(NOAA_SRS_INTERMEDIATE_DIR, "clean_catalog.csv")

HMI_MAG_DIR = os.path.join(DATA_DIR_RAW, "hmi_mag")
MDI_MAG_DIR = os.path.join(DATA_DIR_RAW, "mdi_mag")

MAG_INTERMEDIATE_DIR = os.path.join(DATA_DIR_INTERMEDIATE, "mag")
MAG_INTERMEDIATE_DATA_CSV = os.path.join(MAG_INTERMEDIATE_DIR, "clean_catalog.csv")
MAG_INTERMEDIATE_DATA_DIR = os.path.join(MAG_INTERMEDIATE_DIR, "data")


if __name__ == "__main__":
    print(f"The base directory is `{BASE_DIR}`")
