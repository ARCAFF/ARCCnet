import os
import logging
from datetime import datetime

from arccnet import config

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    # "%(asctime)s - %(name)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s"  # noqa: E501
    "%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s"
)  # can use `%(pathname)s` to get the full path

os.makedirs(config["paths"]["data_dir_logs"], exist_ok=True)
data_logfile = f"{config['paths']['data_dir_logs']}/{datetime.utcnow().strftime('%Y_%m_%d_%H%M%S')}.log"


def get_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(data_logfile)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
