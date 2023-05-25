import os
import subprocess
from datetime import datetime

# ---- JSOC
JSOC_DEFAULT_EMAIL = "pjwright@stanford.edu"
JSOC_DATE_FORMAT = "%Y.%m.%d_%H:%M:%S"
JSOC_BASE_URL = "http://jsoc.stanford.edu/"

# ----
# DATA_START_TIME = datetime(1995, 1, 1)
DATA_START_TIME = datetime(1995, 1, 1, 0, 30, 0)
DATA_END_TIME = datetime(2022, 12, 30, 0, 30, 0)

# !TODO move these to a yaml or something.
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

# -- Base path
# BASE_DIR = os.getcwd()

BASE_DIR = (
    subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode().strip()
)  #!TODO change to config; and assume current working dir.

# --- Error Messages
# NO_DATA_ERROR_MSG = 'No data available.'

# --- Data-related
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_DIR_RAW = os.path.join(DATA_DIR, "raw")
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

HMI_MAG_DIR = os.path.join(DATA_DIR_RAW, "hmi_mag")
MDI_MAG_DIR = os.path.join(DATA_DIR_RAW, "mdi_mag")

# ---- Common Files

RAW_UNABLE_TO_LOAD = "raw_data_unable_to_load.csv"  # might want csv/html etc.

if __name__ == "__main__":
    print(f"The base directory is `{BASE_DIR}`")
