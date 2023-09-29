import datetime

from arccnet import config

__all__ = ["datetime_to_jsoc", "jsoc_to_datetime"]


def datetime_to_jsoc(datetime_obj: datetime.datetime) -> str:
    """
    JSOC string from a datetime object

    Parameters
    ----------
    datetime_obj : datetime.datetime
        a datetime object to convert to JSOC format

    Returns
    -------
    str
        string of format `dv.JSOC_DATE_FORMAT`

    """
    return datetime_obj.strftime(config["jsoc"]["jsoc_date_format"])


def jsoc_to_datetime(date_string: str) -> datetime.datetime:
    """
    Convert jsoc string to a datetime object.

    Parameters
    ----------
    date_string : str
        JSOC formatted string to convert to a datetime object

    Returns
    -------
    datetime.datetime
        a datetime object for the provided `date_string`

    """
    return datetime.datetime.strptime(date_string, config["jsoc"]["jsoc_date_format"])
