import datetime

import arccnet.data_generation.utils.default_variables as dv

__all__ = ["datetime_to_jsoc", "jsoc_to_datetime"]


def datetime_to_jsoc(datetime_obj: datetime.datetime) -> str:
    """
    jsoc string from datetime object

    Parameters
    ----------
    datetime_obj : datetime.datetime
        a datetime object to convert to JSOC format

    Returns
    -------
    str
        string of format `dv.JSOC_DATE_FORMAT`

    """
    return datetime_obj.strftime(dv.JSOC_DATE_FORMAT)


def jsoc_to_datetime(date_string: str) -> datetime.datetime:
    """
    datetime object from jsoc string

    Parameters
    ----------
    date_string : str
        a JSOC format string to convert to datetime object

    Returns
    -------
    datetime.datetime
        a datetime object for the provided `date_string`

    """
    return datetime.datetime.strptime(date_string, dv.JSOC_DATE_FORMAT)
