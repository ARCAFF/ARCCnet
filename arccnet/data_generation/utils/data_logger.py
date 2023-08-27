import os
import logging

from arccnet.data_generation.utils.default_variables import DATA_DIR_LOGS, DATA_LOGFILE

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    # "%(asctime)s - %(name)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s"  # noqa: E501
    "%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s"
)  # can use `%(pathname)s` to get the full path

# create console handler and set level to INFO
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)

os.makedirs(DATA_DIR_LOGS, exist_ok=True)
# create file handler and set level to INFO
fh = logging.FileHandler(DATA_LOGFILE)
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
logger.addHandler(fh)
