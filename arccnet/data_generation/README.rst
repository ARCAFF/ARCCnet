Directory Structure
===================

.. code-block:: shell

    arccnet/data_generation/
    ├── README.rst                # this file
    ├── __init__.py
    │
    ├── catalogs
    │   ├── base_catalog.py       # base class for all catalogs
    │   ├── active_region_catalogs
    │   │   ├── __init__.py
    │   │   ├── assa.py           # ASSA; https://spaceweather.rra.go.kr/assa
    │   │   ├── swpc.py           # NOAA SWPC
    │   │   └── ukmo.py           # UK Met Office
    │   └── utils.py              # catalog-related utils
    │
    ├── magnetograms
    │   ├── __init__.py
    │   ├── base_magnetogram.py   # base class for all magnetogram instruments
    │   ├── instruments
    │   │   ├── __init__.py
    │   │   ├── hmi.py            # SDO/HMI
    │   │   └── mdi.py            # SoHO/MDI
    │   └── utils.py              # magnetogram-related utils
    │
    ├── utils
    │   ├── __init__.py
    │   ├── data_logger.py        # data generation logger
    │   └── default_variables.py  # default variables used for data generation
    │
    ├── data_manager.py           # entry point for obtaining and combining data sources
    .. └── data_processor.py         # entry point to process data into training data

Example Usage
=============

.. code-block:: zsh

    python arccnet/data_generation/data_manager.py

this will instantiate the `arccnet.data_generation.data_manager.DataManager` class, with times defined
in `arccnet.data_generation.utils.default_variables`.

The logs and data are saved in the `data` directory

.. code-block:: shell

    data
    ├── logs                                 # logs for each run of the data processing pipeline
    │    └── data_processing_<year>_<month>_<day>_<hours><minutes><seconds>.log
    │
    ├── 01_raw                              # raw data
    │   ├── mag
    │   │   ├── fits                        # raw magnetogram fits files (subset from `02_intermediate/mag/clean_catalog.csv`)
    │   │   │   └── ...
    │   │   ├── hmi
    │   │   │   └── raw.csv                 # csv of raw hmi meta/fits
    │   │   └── mdi
    │   │       └── raw.csv                 # csv of raw hmi meta/fits
    │   │
    │   └── noaa_srs                        # noaa srs files
    │       ├── txt
    │       │   └── ...                     # files that were loaded correctly
    │       ├── raw_data.html               # html of compiled (raw) data
    │       ├── raw_data.csv                # csv of compiled (raw) data
    │       ├── txt_load_error
    │       │   └── ...                     # files that encountered a load error
    │       ├── raw_data_load_error.html    # html of raw data that encountered a load error
    │       └── raw_data_load_error.csv     # csv of raw data that encountered a load error
    │
    ├── 02_intermediate
    │   ├── mag
    │   │   └── clean_catalog.csv           # cleaned magnetogram catalog. Matched with SRS
    │   └── noaa_srs
    │       └── clean_catalog.csv           # cleaned srs catalog.
    │
    └── final                               # not implemented
