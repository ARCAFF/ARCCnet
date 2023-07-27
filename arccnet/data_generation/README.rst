Directory Structure
===================

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
    │   └── noaa_srs                        # noaa SRS files
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
    │       └── clean_catalog.csv           # cleaned SRS catalog.
    │
    ├── 03_processed
    │   ├── mag
    │   │   └── fits
    │   │       └── ...                     # AR Cutouts `<year>-<month>-<day>_<NOAA_ARNUM>.fits`
    │   └── processed.csv                   # CSV regarding processed data
    │
    └── final                               # not implemented


Example Usage
=============

.. code-block:: zsh

    python arccnet/data_generation/data_manager.py

this will instantiate the `arccnet.data_generation.data_manager.DataManager` class, with times defined
in `arccnet.data_generation.utils.default_variables`.

The logs and data are saved in the `data` directory

.. code-block:: zsh

    python arccnet/data_generation/mag_processing.py

this will execute data processing and active region extraction.
