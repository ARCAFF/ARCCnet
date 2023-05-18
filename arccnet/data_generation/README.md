
## Directory Structure

```bash
src/data/
├── README.md # this file
├── __init__.py
│
├── catalogs
│   ├── base_catalog.py # base class for all catalogs
│   ├── active_region_catalogs # contains modules for obtaining and processing active region classification data from different sources.
│   │   ├── __init__.py
│   │   ├── assa.py # ASSA; https://spaceweather.rra.go.kr/assa
│   │   ├── swpc.py # NOAA SWPC
│   │   └── ukmo.py # UK MetOffice
│   └── utils.py # catalog-related utils
│
├── magnetograms # contains modules for obtaining and processing magnetogram data from different instruments.
│   ├── __init__.py
│   ├── base_magnetogram.py # base class for all magnetogram instruments
│   ├── instruments # directory for specific magnetogram instruments
│   │   ├── __init__.py
│   │   ├── hmi.py # SDO/HMI
│   │   └── mdi.py # SoHO/MDI
│   └── utils.py # magnetogram-related utils
│
├── data_manager.py # entry point for obtaining and combining data sources
└── make_dataset.py

```

## Example Usage
To use the modules in this directory, you can import them in your code as follows...

(coming soon.)
