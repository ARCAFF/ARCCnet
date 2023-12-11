---
file_format: mystnb

kernelspec:
  display_name: Python 3
  language: python
  name: python3

jupytext:
  text_representation:
    extension: .md
    format_name: myst

---
```{code-cell} python3
:tags: [hide-cell, remove-input, remove-output]

# set working directory to the base of the repo
%cd ../..

from myst_nb import glue
from datetime import datetime
from pathlib import Path
from astropy.table import QTable, join
import numpy as np
import pandas as pd
import sunpy
import sunpy.map
from arccnet import config
from arccnet.visualisation.data import (
    plot_hmi_mdi_availability,
    plot_col_scatter,
    plot_col_scatter_single,
    plot_maps,
    plot_map,
    plot_maps_regions,
)

import matplotlib.pyplot as plt

from arccnet import __version__
glue("arccnet_version", __version__, display=False)

# Load data -------------------------------------------------------------------
NOAA_AR_IDENTIFIER = 'I'
glue("NOAA_AR_IDENTIFIER", NOAA_AR_IDENTIFIER, display=False)

# 1. SRS
srs_query_results = QTable.read(Path(config["paths"]["data_dir_intermediate"]) / "noaa_srs" / "srs_results.parq") # search results
srs_raw_catalog = QTable.read(Path(config["paths"]["data_dir_intermediate"]) / "noaa_srs" / "srs_raw_catalog.parq") # search results
srs_clean_catalog = QTable.read(Path(config["paths"]["data_dir_final"]) / "srs_clean_catalog.parq") # final cleaned catalog

# srs query: start, end, length
glue("srs_query_start", srs_query_results['start_time'][0].strftime("%Y-%m-%d"), display=False)
glue("srs_query_end", srs_query_results['start_time'][-1].strftime("%Y-%m-%d"), display=False)
glue("len_srs_query_results", len(srs_query_results), display=False)

# srs exist (may not load)
srs_files_exist = srs_raw_catalog[srs_raw_catalog['filename'] != ''] # files exist
glue("num_unique_srs_exist", len(np.unique(srs_files_exist['filename'])), display=False) # unique files

# srs exist (and loaded)
srs_files_exist_loaded = srs_files_exist[srs_files_exist['loaded_successfully'] == True] # files that exist and loaded correctly
# glue("srs_exist_loaded_start", srs_files_exist_loaded['time'][0].strftime("%Y-%m-%d %H:%M:%S"), display=False) # start time
# glue("srs_exist_loaded_end", srs_files_exist_loaded['time'][-1].strftime("%Y-%m-%d %H:%M:%S"), display=False) # end time
glue("srs_exist_loaded_start", pd.to_datetime(srs_files_exist_loaded.to_pandas()['time']).round("D").unique()[0].strftime("%Y-%m-%d"), display=False) # rounded to nearest dat
glue("srs_exist_loaded_end", pd.to_datetime(srs_files_exist_loaded.to_pandas()['time']).round("D").unique()[-1].strftime("%Y-%m-%d"), display=False) # rounded to nearest day
glue("len_srs_exist_loaded", len(srs_files_exist_loaded), display=False) # number of regions that exist
glue("num_unique_srs_exist_loaded", len(np.unique(srs_files_exist_loaded['filename'])), display=False) # number of unique files
glue("srs_files_exist_loaded_num_noaa_ar_identifier", len(srs_files_exist_loaded[srs_files_exist_loaded['id'] == NOAA_AR_IDENTIFIER]), display=False) # ... are of the NOAA type

# unique ID type in the SRS table
glue("srs_files_exist_loaded_unique_srs_id", set(srs_files_exist_loaded[~srs_files_exist_loaded['id'].mask]['id']), display=False)

# clean catalog (only ID == I currently)
glue("len_srs_clean_catalog", len(srs_clean_catalog), display=False)
glue("num_unique_dates_srs_clean_catalog", len(np.unique(srs_clean_catalog['time'])), display=False)

# 2. HMI
hmi_table = QTable.read(Path(config["paths"]["data_root"]) / "02_intermediate" / "mag" / "hmi_results.parq").to_pandas()
hmi_processed = QTable.read(Path(config["paths"]["data_root"]) / "03_processed" / "mag" / "hmi_processed.parq")
sharps_downloads = QTable.read(Path(config["paths"]["data_root"]) / "02_intermediate" / "mag" / "sharps_downloads.parq") # sharps downloads
# ... extracted values
# glue("hmi_query_start", hmi_processed['target_time'][0].strftime("%Y-%m-%d %H:%M:%S"), display=False)
# glue("hmi_query_end", hmi_processed['target_time'][-1].strftime("%Y-%m-%d %H:%M:%S"), display=False)
glue("hmi_query_start", hmi_processed['target_time'][0].strftime("%Y-%m-%d"), display=False) # just provide date
glue("hmi_query_end", hmi_processed['target_time'][-1].strftime("%Y-%m-%d"), display=False) # just provide date
hmi_data_exist = hmi_processed[~hmi_processed['datetime'].mask]
glue("len_hmi_data_exist", len(hmi_data_exist), display=False)
glue("num_unique_hmi_data_exist", len(np.unique(hmi_data_exist['datetime'])), display=False)
# glue("hmi_data_exist_start", hmi_data_exist['datetime'][0].strftime("%Y-%m-%d %H:%M:%S"), display=False)
# glue("hmi_data_exist_end", hmi_data_exist['datetime'][-1].strftime("%Y-%m-%d %H:%M:%S"), display=False)
glue("hmi_data_exist_start", pd.to_datetime(hmi_data_exist.to_pandas()['datetime']).round("D").unique()[0].strftime("%Y-%m-%d"), display=False) # rounded to nearest day
glue("hmi_data_exist_end", pd.to_datetime(hmi_data_exist.to_pandas()['datetime']).round("D").unique()[-1].strftime("%Y-%m-%d"), display=False) # rounded to nearest day

sharps_data_exist = sharps_downloads[~sharps_downloads['datetime'].mask]
glue("len_sharps_data_exist", len(sharps_data_exist), display=False)
glue("num_unique_sharps_data_exist", len(np.unique(sharps_data_exist['datetime'])), display=False)
hmi_sharps_merged_file = QTable.read(Path(config["paths"]["data_root"]) / "03_processed" / "mag" / "hmi_sharps_merged.parq")
glue("len_hmi_sharps_merged", len(hmi_sharps_merged_file), display=False)
glue("num_unique_hmi_sharps_merged", len(np.unique(hmi_sharps_merged_file['datetime'])), display=False)

# 3. MDI
mdi_table = QTable.read(Path(config["paths"]["data_root"]) / "02_intermediate" / "mag" / "mdi_results.parq").to_pandas()
mdi_processed = QTable.read(Path(config["paths"]["data_root"]) / "03_processed" / "mag" / "mdi_processed.parq")
smarps_downloads = QTable.read(Path(config["paths"]["data_root"]) / "02_intermediate" / "mag" / "smarps_downloads.parq") # smarps downloads
# ... extracted values
glue("mdi_query_start", mdi_processed['target_time'][0].strftime("%Y-%m-%d"), display=False) # just show date
glue("mdi_query_end", mdi_processed['target_time'][-1].strftime("%Y-%m-%d"), display=False) # just show date
mdi_data_exist = mdi_processed[~mdi_processed['datetime'].mask]
glue("len_mdi_data_exist", len(mdi_data_exist), display=False)
glue("num_unique_mdi_data_exist", len(np.unique(mdi_data_exist['datetime'])), display=False)
# glue("mdi_data_exist_start", mdi_data_exist['datetime'][0].strftime("%Y-%m-%d %H:%M:%S"), display=False)
# glue("mdi_data_exist_end", mdi_data_exist['datetime'][-1].strftime("%Y-%m-%d %H:%M:%S"), display=False)
glue("mdi_data_exist_start", pd.to_datetime(mdi_data_exist.to_pandas()['datetime']).round("D").unique()[0].strftime("%Y-%m-%d"), display=False) # rounded to nearest day
glue("mdi_data_exist_end", pd.to_datetime(mdi_data_exist.to_pandas()['datetime']).round("D").unique()[-1].strftime("%Y-%m-%d"), display=False) # rounded to nearest day

smarps_data_exist = smarps_downloads[~smarps_downloads['datetime'].mask]
glue("len_smarps_data_exist", len(smarps_data_exist), display=False)
glue("num_unique_smarps_data_exist", len(np.unique(smarps_data_exist['datetime'])), display=False)
mdi_smarps_merged_file = QTable.read(Path(config["paths"]["data_root"]) / "03_processed" / "mag" / "mdi_smarps_merged.parq")
glue("len_mdi_smarps_merged", len(mdi_smarps_merged_file), display=False)
glue("num_unique_mdi_smarps_merged", len(np.unique(mdi_smarps_merged_file['datetime'])), display=False)

# 4. SRS-Magnetogram
srs_hmi = QTable.read(Path(config["paths"]["data_root"]) / "03_processed" / "mag" / "srs_hmi_merged.parq")
srs_mdi = QTable.read(Path(config["paths"]["data_root"]) / "03_processed" / "mag" / "srs_mdi_merged.parq")

glue("len_srs_hmi", len(srs_hmi), display=False)
srs_hmi_ppi_exists = np.unique(srs_hmi[~srs_hmi['processed_path_image'].mask]['time'])
glue("len_srs_hmi_processed_path", len(srs_hmi_ppi_exists), display=False)
glue("num_unique_dates_srs_hmi_processed_path", len(np.unique(srs_hmi_ppi_exists)), display=False)

glue("len_srs_mdi", len(srs_mdi), display=False)
srs_mdi_ppi_exists = np.unique(srs_hmi[~srs_hmi['processed_path_image'].mask]['time'])
glue("len_srs_mdi_processed_path", len(srs_mdi_ppi_exists), display=False)
glue("num_unique_dates_srs_mdi_processed_path", len(np.unique(srs_mdi_ppi_exists)), display=False)


# 5. DATASETS
# ... Region Detection
region_detection_table = QTable.read(Path(config["paths"]["data_root"]) / "04_final" / "mag" / "region_detection" / "region_detection.parq")
# ... Region Extraction
region_classification_table = QTable.read(Path(config["paths"]["data_root"]) / "04_final" / "mag" / "region_extraction" / "region_classification.parq")
rct_hmi = region_classification_table[~region_classification_table['path_image_cutout_hmi'].mask]
rct_mdi = region_classification_table[~region_classification_table['path_image_cutout_mdi'].mask]
glue("len_hmi_region_extraction_ar", len(rct_hmi[rct_hmi['region_type'] == 'AR']), display=False)
glue("len_hmi_region_extraction_qs", len(rct_hmi[rct_hmi['region_type'] == 'QS']), display=False)
glue("len_mdi_region_extraction_ar", len(rct_mdi[rct_mdi['region_type'] == 'AR']), display=False)
glue("len_mdi_region_extraction_qs", len(rct_mdi[rct_mdi['region_type'] == 'QS']), display=False)

# -----------------------------------------------------------------------------



# -- Data Availability --------------------------------------------------------
#
plotting_start_date = datetime(1995, 1, 1)
plotting_end_date = datetime.now()

# plotting
# 1. magnetogram availability
# 2. cdelt for MDI/HMI
# 3. dsun for MDI/HMI
mag_availability = plot_hmi_mdi_availability(hmi_table, mdi_table, plotting_start_date, plotting_end_date)
zoom_start_date = datetime(2009, 1, 1)
zoom_end_date = datetime(2012, 6, 1)
mag_availability_zoom = plot_hmi_mdi_availability(hmi_table, mdi_table, zoom_start_date, zoom_end_date)

cdelt = plot_col_scatter([mdi_table, hmi_table], column="CDELT1", colors=["red", "blue"]) # plate scale
dsun = plot_col_scatter_single([mdi_table, hmi_table], column="DSUN_OBS", colors=["red", "blue"]) # instrument-sun distance

glue("mag_cdelt", cdelt[0], display=False)
glue("mag_dsun", dsun[0], display=False)
glue("mag_availability", mag_availability[0], display=False)
glue("mag_availability_zoom", mag_availability_zoom[0], display=False)
glue("plotting_start_date", plotting_start_date.strftime("%d-%h-%Y"), display=False)
glue("plotting_end_date", plotting_end_date.strftime("%d-%h-%Y"), display=False)
glue("plotting_start_date_zoom", zoom_start_date.strftime("%d-%h-%Y"), display=False)
glue("plotting_end_date_zoom", zoom_end_date.strftime("%d-%h-%Y"), display=False)

# -- Extract a cotemporal observation --
# Group the table by the "time" column and iterate through the groups to find
# the first group with both 'mdi' and 'hmi' in the "instrument" column
for group in region_detection_table.group_by('target_time').groups:
    if 'MDI' in group['instrument'] and 'HMI' in group['instrument']:
        cotemporal_obs_date_time = np.unique(group['target_time'])[0]
        break

cotemporal_obs_date = cotemporal_obs_date_time.to_datetime().strftime("%Y-%m-%d")

# pre-processed maps
hmi = sunpy.map.Map(hmi_processed[hmi_processed['target_time'] == cotemporal_obs_date_time]['path'].data.data[0])
mdi = sunpy.map.Map(mdi_processed[mdi_processed['target_time'] == cotemporal_obs_date_time]['path'].data.data[0])

# post-processed maps
processed_hmi_file = sunpy.map.Map(hmi_processed[hmi_processed['target_time'] == cotemporal_obs_date_time]['processed_path'].data.data[0])
processed_mdi_file = sunpy.map.Map(mdi_processed[mdi_processed['target_time'] == cotemporal_obs_date_time]['processed_path'].data.data[0])

# plotting
mag, _ = plot_maps(mdi, hmi)
mag_processed, _ = plot_maps(processed_mdi_file, processed_hmi_file)
# -----------------------------------------------------------------------------



# -- AR Classification --------------------------------------------------------

# subset based on the cotemporal observations
rct_subset = region_classification_table[region_classification_table['time'] == cotemporal_obs_date_time]

# rename for the plotting function
map_one = sunpy.map.Map(rct_subset['processed_path_image_hmi'][0])
regions_one = rct_subset['region_type', 'top_right_cutout_hmi', 'bottom_left_cutout_hmi', 'number', 'magnetic_class', 'carrington_longitude', 'area', 'mcintosh_class', 'number_of_sunspots']
regions_one.rename_columns(
    ['top_right_cutout_hmi', 'bottom_left_cutout_hmi'],
    ['top_right_cutout', 'bottom_left_cutout'],
)
regions_one = regions_one[regions_one["region_type"] == "AR"]

# rename for the plotting function
map_two = sunpy.map.Map(rct_subset['processed_path_image_mdi'][0])
regions_two = rct_subset['region_type', 'top_right_cutout_mdi', 'bottom_left_cutout_mdi', 'number', 'magnetic_class', 'carrington_longitude', 'area', 'mcintosh_class', 'number_of_sunspots']
regions_two.rename_columns(
    ['top_right_cutout_mdi', 'bottom_left_cutout_mdi'],
    ['top_right_cutout', 'bottom_left_cutout'],
)
regions_two = regions_two[regions_two["region_type"] == "AR"]

# extract cutout with largest area
# !TODO fix this for when there isn't the same region.
# I suspect this won't break because of the area criteria
regions_one_sorted = regions_one.copy()
regions_one_sorted.sort('area')
hmi_ar = regions_one_sorted[-1:]
hmi_ar_num = hmi_ar['number']
mdi_ar = regions_two[regions_two['number'] == hmi_ar_num]
hmi_ar_dict = regions_one_sorted['number', 'magnetic_class', 'carrington_longitude', 'area', 'mcintosh_class', 'number_of_sunspots'].to_pandas().to_dict(orient='records')[0]
hmi_ar_pdseries = pd.Series(hmi_ar_dict)

# -- setting the values to be the zeroth element (and plotting the associated cutouts)
glue("regions_hmi_mcintosh", hmi_ar['mcintosh_class'][0], display=False)
glue("regions_hmi_mag_class", hmi_ar['magnetic_class'][0], display=False)
glue("regions_sunspots", hmi_ar['number_of_sunspots'][0], display=False)

# plotting
map_cutouts, _ = plot_maps_regions(map_two, regions_two, map_one, regions_one, **{
    'edgecolor': 'black',
    'linestyle': '--',
})

smap_hmi = map_one.submap(top_right=hmi_ar['top_right_cutout'][0], bottom_left=hmi_ar['bottom_left_cutout'][0])
smap_mdi = map_two.submap(top_right=mdi_ar['top_right_cutout'][0], bottom_left=mdi_ar['bottom_left_cutout'][0])
mag_cutouts, _ = plot_maps(smap_mdi, smap_hmi, figsize=(10,2))

glue("mag_co", plot_map(smap_hmi)[0], display=False)
glue("mag_co_dict", pd.DataFrame({'Key': hmi_ar_pdseries.index, 'Value': hmi_ar_pdseries.values}), display=False)
glue("mag_co_html", ''.join([f"<b>{key}</b>: {value}<br>" for key, value in hmi_ar_dict.items()]), display=False)

# -----------------------------------------------------------------------------



# -- AR Detection -------------------------------------------------------------

# subset based on the cotemporal observations
rdt_subset = region_detection_table[region_detection_table['target_time'] == cotemporal_obs_date_time]

hmi_region_table = rdt_subset[rdt_subset['instrument'] == 'HMI']
hmi_map = sunpy.map.Map(hmi_region_table["processed_path"][0])
mdi_region_table = rdt_subset[rdt_subset['instrument'] == 'MDI']
mdi_map = sunpy.map.Map(mdi_region_table["processed_path"][0])

map_cutouts_two, _ = plot_maps_regions(mdi_map, mdi_region_table, hmi_map, hmi_region_table, **{
    'edgecolor': 'black',
    'linestyle': '--',
})
# -----------------------------------------------------------------------------

glue("two_plots", mag, display=False)
glue("two_plots_processed", mag_processed, display=False)
glue("two_plots_cutouts", map_cutouts, display=False)
# glue("two_plots_cutouts_mdi", mag_cutouts_mdi, display=False)
glue("two_plots_cutouts_hmi", mag_cutouts, display=False)
glue("two_plots_cutouts_two", map_cutouts_two, display=False)
# glue("rct_subset", rct_subset, display=True)
# glue("rdt_subset", rdt_subset, display=True)
glue("hmi_plot", hmi, display=False)
glue("mdi_plot", mdi, display=False)
glue("obs_date", cotemporal_obs_date, display=False)

# -----------------------------------------------------------------------------

```

# Magnetograms

```{glossary}

SoHO
  Solar and Heliospheric Observatory

MDI
  Michelson Doppler Imager

SDO
  Solar Dynamics Observatory

HMI
  Helioseismic and Magnetic Imager

SHARPs
  Space-weather HMI Active Region Patches

SMARPs
  Space-weather MDI Active Region Patches

JSOC
  Joint Science Operations Center
```

## Introduction

Two primary goals in the ARCAFF project are Active Region Detection and Classification. To complement the NOAA SRS data described earlier,
we retrieve line-of-sight magnetograms once-per-day, from 1995 - 2022, synchronized with the validity of NOAA SRS reports at 00:00 UTC (issued at 00:30 UTC).

This document will discuss the magnetogram data, from full-disk SoHO/MDI, SDO/HMI to SMARP/SHARP cutouts, and the processing required to generate ML-ready datasets.

:::{important}
This report was generated based upon version ({glue}`arccnet_version`) of the `arccnet` dataset.
:::

## Data Sources and Observations

The observations from SoHO/MDI (1995 - 2011; {cite:t}`Scherrer1995,Domingo1995`) and SDO/HMI (2010 - present {cite:p}`Scherrer2012,Pesnell2012`) are retrieved from the Joint Science Operations Center (JSOC) at Stanford University for 1996 - 2022 inclusive, leaving 2023 as unseen data.

The availability of images in our dataset is shown for this period in Figure {numref}`fig:mag_availability`, where between 2010 and 2011, there are co-temporal observations of the line-of-sight component of the magnetic field.

```{glue:figure} mag_availability
:alt: "HMI-MDI Availability"
:name: "fig:mag_availability"
HMI-MDI coverage diagram from {glue}`plotting_start_date` to {glue}`plotting_end_date`.
```

```{glue:figure} mag_availability_zoom
:alt: "HMI-MDI Availability"
:name: "fig:mag_availability_zoom"
MDI and HMI coverage diagram from {glue}`plotting_start_date_zoom` to {glue}`plotting_end_date_zoom`.
```

### Co-temporal observations

For the problems of active region classification and detection, the observed distributions of active region classes across a given solar cycle (and therefore instrument) are not uniform, and the number of observed active regions varies across solar cycles themselves.

Datasets that combine observations from multiple observatories allow us to understand dynamics across solar cycles and play a cruicial role in increasing the number of available samples for training machine learning models. However, while improvements to instrumentation can fuel scientific advancements, for studies over the typical observatory lifespan, the varying spatial resolutions, cadences, and systematics (to name a few) make their direct comparisons challenging.

The expansion of the SHARP series {cite:p}`Bobra2014` to SoHO/MDI (SMARPs; {cite:t}`Bobra2021`) (see {ref}`sec:cutouts`) has tried to negate this with a tuned detection algorithm to provide similar active region cutouts (and associated parameters) across two solar cycles. Other authors have incorporated advancements in the state-of-the-art for image translation to cross-calibrate data, however, out-of-the-box, these models generally prefer perceptual similarity. Importantly, progress has been made towards physically-driven approaches for instrument cross-calibration/super-resolution (e.g. Munoz-Jaramillo et al 2023 (in revision)) that takes into account knowledge of the underlying physics.

Initially as part of `arccnet`, we will utilise each instrument as a separate source, before expanding to cross-calibration techniques that homogenise the data. Examples of co-temporal data (for {glue}`obs_date`) are shown below with SunPy map objects in Figure {numref}`fig:mag_cotemporal`.

```{glue:figure} two_plots
:alt: "Cotemporal HMI-MDI"
:name: "fig:mag_cotemporal"
MDI (left) and HMI (right) observation of the Sun's magnetic field at {glue}`obs_date`.
```

#### Instrumental/Orbital Effects on Data

While there are noticeable visual differences (e.g. resolution and noise properties), there are a number of subtle differences between these instruments that can be observed in the metadata, and should be accounted for. Both instruments are located at different positions in space, and at different distances from the Sun, which vary as the Earth orbits around the Sun.

```{glue:figure} mag_cdelt
:alt: "HMI-MDI CDELT1"
:name: "fig:mag_cdelt"
`CDELT1` (image scale in the x-direction [arcsec/pixel]) from {glue}`plotting_start_date` to {glue}`plotting_end_date` for SDO/HMI (top, blue) and SoHO/MDI (bottom, red).
```

```{glue:figure} mag_dsun
:alt: HMI-MDI DSUN
:name: fig:mag_dsun
`DSUN_OBS` (distance from instrument to sun-centre [metres]) from {glue}`plotting_start_date` to {glue}`plotting_end_date` for SDO/HMI (blue) and SoHO/MDI (red).
```

To demonstrate some of these instrumental and orbital differences Figure {numref}`fig:mag_cdelt` shows the image scale in the x-axis of the image plane as a function of time for SDO/HMI and SoHO/MDI, while Figure {numref}`fig:mag_dsun` demonstrates how the radius of the Sun varies during normal observations. {numref}`fig:mag_cdelt`.

While these can be corrected through data preparation and processing, including the reprojection of images to be as-observed from a set location along the Sun-Earth line, complex relationships mean it may be necessary to use machine learning models (such as the cross-calibration approach mentioned previously) to prepare data.

(sec:cutouts)=
### Cutouts (SHARP/SMARP)

The SHARP/SMARP series are the "Spaceweather HMI/MDI Active Region patches". Using SHARP as an example, a SHARP is a series that contains:

1. various spaceweather quantities calculated from vector magnetogram data and stored as FITS header keywords
2. 31 data segments, including each component of the vector magnetic field, the line-of-sight magnetic field,
continuum intensity, doppler velocity, error maps and bitmaps.

Similarly to NOAA AR numbers, the SHARP(/SMARP) have their own region identifiers HARP(/TARP).
Below is an example of the output of the active-region automatic detection algorithm for the 13th Jan 2013.
Figure {numref}`fig:bobra_sharp` provides a visual representation of the detection algorithm, with HARP regions and NOAA ARs shown.

```{figure} images/bobra2014_sharps_fig1.png
:alt: Full-disk image of the sun with HARP regions (Bobra et al 2014)
:name: fig:bobra_sharp
The results of the active-region automatic detection algorithm applied to the data on 13 January 2013 at 00:48 TAI. NOAA active-region numbers are labeled in blue near the Equator, next to arrows indicating the hemisphere; the HARP number is indicated inside the rectangular bounding box at the upper right. Note that HARP 2360 (lower right, in green) includes two NOAA active regions, 11650 and 11655. The colored patches show coherent magnetic structures that comprise the HARP. White pixels have a line-of-sight field strength above a line-of-sight magnetic-field threshold {cite:p}`Turmon2014AAS`. Blue + symbols indicate coordinates that correspond to the reported center of a NOAA active region. The temporal life of a definitive HARP starts when it rotates onto the visible disk or two days before the magnetic feature is first identified in the photosphere. As such, empty boxes, e.g. HARP 2398 (on the left), represent patches of photosphere that will contain a coherent magnetic structure at a future time. This figure is reproduced from {cite:t}`Bobra2014`.
```

Of the 31 data segments, the bitmap segment identifies the pixels with five unique values because they describe two different characteristics, as shown below:

| Value | Keyword   | Definition                                       |
|-------|-----------|-------------------------------------------------|
| 0     | `OFFDISK`   | The pixel is located somewhere off the solar disk.           |
| 1     | `QUIET`     | The pixel is associated with weak line-of-sight magnetic field. |
| 2     | `ACTIVE`    | The pixel is associated with strong line-of-sight magnetic field.|
| 32    | `ON_PATCH`  | The pixel is inside the patch.                       |

where all possible permutations are as follows:

| Value | Definition                                                         |
|-------|-------------------------------------------------------------------|
| 0     | The pixel is located somewhere off the solar disk.               |
| 1     | The pixel is located outside the patch and associated with weak line-of-sight magnetic field. |
| 2     | The pixel is located outside the patch and associated with strong line-of-sight magnetic field. |
| 33    | The pixel is located inside the patch and associated with weak line-of-sight magnetic field. |
| 34    | The pixel is located inside the patch and associated with strong line-of-sight magnetic field. |

The bitmap segment for SMARPs is similar, but more complicated. See <https://github.com/mbobra/SMARPs/blob/main/example_gallery/Compare_SMARP_and_SHARP_bitmaps.ipynb>.

:::{seealso}
For more discussion on the generation of SHARP/SMARP data, see <https://github.com/mbobra/SHARPs>, <https://github.com/mbobra/SMARPs>.
:::

## Data Processing

The obtained data is of various processing levels, taken from individual instruments that have varying noise properties, plate scales, and systematics, among others.
For example, the *Line-of-sight magnetograms* from SDO/HMI and SoHO/MDI are obtained as "level 1.5", and while not addressed in version  SDO/AIA data is provided at level 1,
and requires various pre-processing steps to reach the same pre-processing level.

As such, all data obtained as part of this project needs to be carefully processed to an agreed upon level, with further corrections and transformations applied to process these data for machine learning applications.

### Full-disk HMI/MDI

In version {glue}`arccnet_version` of the dataset, a preliminary data processing routine is applied to full-disk MDI and HMI (as shown below) to include

1. Rotation to Solar North
2. Removal (and zero) off-disk data


Compared to Figure {numref}`fig:mag_cotemporal`, Figure {numref}`fig:hmi_cotemporalmagprocess` shows these corrections applied, however, the removal of off-disk data in MDI is incomplete, leaving NaN values on the limb (white).


```{glue:figure} two_plots_processed
:alt: "Processed Cotemporal HMI-MDI"
:name: "fig:hmi_cotemporalmagprocess"
MDI (left) and HMI (right) observation of the Sun's magnetic field at {glue}`obs_date`.
```

As we progress towards v1.0, the processing pipeline will also be expanded to include additional corrections such as

* Reprojection of images to a fixed point at 1 AU along the Sun-Earth line.
* Filtering according to the hexadecimal `QUALITY` flag.

Currently the correct choice of reprojection is still under consideration, and the handling of conversion of hexadecimal to float needs to be clarified.

#### Further Corrections

In future versions, additional data processing steps may include

1. Instrument inter-calibration (and super-resolution)
    * Due to optically distortion in MDI, even with reprojection to a common coordinate frame, perfect alignment of images is not possible between these instruments
2. Addition of Coronal Information
    * Inclusion and alignment of EUV images (such as those from SDO/AIA)
    * Generation of Differential Emission Measure (DEM) maps {cite:p}`2012A&A...539A.146H,2015ApJ...807..143C`
3. Magnetic Field Extrapolation

### Cutouts

In this version, we have obtained the bitmap segments as a means to extract regions-of-interest from the full-disk images, and the only necessary correction is to account for the rotation of the instrument to Solar North, as was performed with the full-disk images (the SHARP/SMARP regions are extracted from the full-disk image in the detector frame).

(sec:dataset_generation)=
## Dataset Generation

The NOAA SRS archive was queried daily from {glue}`srs_query_start` to {glue}`srs_query_end` (spanning a total of {glue}`len_srs_query_results` days). This query returned {glue}`num_unique_srs_exist` NOAA SRS files, where {glue}`num_unique_srs_exist_loaded` were valid and successfully parsed. From these {glue}`num_unique_srs_exist_loaded` SRS text files, we extracted a total of {glue}`len_srs_exist_loaded` regions (with IDs {glue}`srs_files_exist_loaded_unique_srs_id`), spanning {glue}`srs_exist_loaded_start` to {glue}`srs_exist_loaded_end`, inclusive. Out of these {glue}`len_srs_exist_loaded` regions, {glue}`srs_files_exist_loaded_num_noaa_ar_identifier` are of interest as they have an identifier equal to {glue}`NOAA_AR_IDENTIFIER`. This table is further filtered based on criteria described in {doc}`./arcnet-dataset.md`, resulting in a reduced number of usable regions, totaling {glue}`len_srs_clean_catalog`, obtained over {glue}`num_unique_dates_srs_clean_catalog` days.

Additionally, we queried JSOC for SDO/HMI and SoHO/MDI data over the period {glue}`hmi_query_start` to {glue}`hmi_query_end`. For HMI, we obtained {glue}`len_hmi_data_exist` observations spanning {glue}`hmi_data_exist_start` to {glue}`hmi_data_exist_end`, and for MDI there were {glue}`len_mdi_data_exist` observations (spanning {glue}`mdi_data_exist_start` to {glue}`mdi_data_exist_end`).

To generate datasets that relate the NOAA SRS regions and full-disk images, we performed individual merges of the NOAA SRS and SDO/HMI and SoHO/MDI catalogs, resulting in the creation of four primary tables: SRS-HMI, SRS-MDI, and HMI-SHARPS, MDI-SMARPS.

**SRS-HMI & SRS-MDI**

As mentioned, the NOAA SRS catalog contains {glue}`len_srs_clean_catalog` rows, while the HMI table contains {glue}`len_hmi_data_exist` rows (and MDI, {glue}`len_mdi_data_exist` rows). We independently merged the NOAA SRS table with both the MDI and HMI tables using their respective time columns, resulting in an SRS-HMI table of length {glue}`len_srs_hmi` ({glue}`len_srs_mdi` for SRS-MDI). Among these rows, there are {glue}`len_srs_hmi_processed_path` rows ({glue}`num_unique_dates_srs_hmi_processed_path` dates) where SRS and HMI both have data ({glue}`len_srs_mdi_processed_path` for SRS-MDI; {glue}`num_unique_dates_srs_mdi_processed_path` dates).

**HMI-SHARPS & MDI-SMARPS**

For each Full-disk and cutout magnetogram pair (e.g. HMI-SHARPs, MDI-SMARPs) the respective HMI/MDI and SHARP/SMARP tables are merged on the `datetime` column, limiting the dataset to those full-disk images that have on-disk regions.

We start with {glue}`len_hmi_data_exist` rows for HMI ({glue}`len_mdi_data_exist` for MDI), corresponding to {glue:}`num_unique_hmi_data_exist` ({glue:}`num_unique_mdi_data_exist`) dates.
These are merged with the SHARP(/SMARP) tables, that contain {glue}`len_sharps_data_exist` ({glue}`len_smarps_data_exist`) rows, and correspond to {glue:}`num_unique_sharps_data_exist` ({glue:}`num_unique_smarps_data_exist`) dates.
After merging, this results in {glue}`len_hmi_sharps_merged` rows in the HMI-SHARPs table, ({glue}`len_mdi_smarps_merged` in the MDI-SMARPs table), corresponding to {glue}`num_unique_hmi_sharps_merged` ({glue}`num_unique_mdi_smarps_merged`) unique dates.

(sec:arclassdataset)=
### Active Region Classification Dataset

One of the primary goals of the ARCAFF project is develop an automated solution to classify active regions based on existing schema (E.g. Mcintosh classes, or modified Zurich).
By developing an automated solution, the aim is to automatically generate active region classifications in real-time, and utilise these as part of an end-to-end machine learning pipeline,
or even used independelty, e.g. {cite:t}`2002SoPhGallagher,2012ApJBloomfield,2016SoPhMcCloskey,2018JSWSCMcCloskey` (see <http://solarmonitor.org>) to provide flaring forecast probabilities.

To tackle this as a supervised learning problem, we require a dataset consisting of:

1. **input**: cutouts centered on labelled NOAA ARs
2. **output**: corresponding classification labels

In version {glue}`arccnet_version`, the active region locations (and classification label output) have been extracted from the daily SRS files. The locations, combined with a pre-defined region size were used to crop cutouts from the full-disk magnetograms.

An example active region from this dataset, extracted from SDO/HMI (with {glue}`regions_sunspots` sunspots) is shown in Figure {numref}`fig:hmi_mag_co`, along with the corresponding Mcintosh/Mag classification labels: {glue}`regions_hmi_mcintosh`, {glue}`regions_hmi_mag_class`.

```{glue:figure} mag_co
:alt: "HMI Magnetogram Cutout"
:name: "fig:hmi_mag_co"
HMI SHARP cutout (above), with corresponding classification labels (below) on {glue}`obs_date`
```
```{code-cell} python3
:tags: [remove-input]
pd.DataFrame({'Key': hmi_ar_pdseries.index, 'Value': hmi_ar_pdseries.values})
```

#### Dataset creation

In practice, we started with the SRS-HMI table ({glue}`len_srs_hmi_processed_path` rows) the active regions are extracted, and SRS-HMI and SRS-MDI tables are merged on the `number` and `time` columns (for the `AR` `region_type` the `number` is an NOAA AR Number, extracted from the SRS text files, and for `QS`, this is an integer $n ∈ [0, N)$, where $N$ was the number of QS regions requested during data generation.  The resulting HMI AR Classification dataset contains a total of {glue}`len_hmi_region_extraction_ar` ARs, and {glue}`len_hmi_region_extraction_qs` QS regions. Similarly, for MDI, there are {glue}`len_mdi_region_extraction_ar` ARs {glue}`len_mdi_region_extraction_qs` QS regions. The ARs at {glue}`obs_date` are shown below for both MDI and HMI.

:::{important}
In this version ({glue}`arccnet_version`), we have defined a quiet sun region as any region that is not a NOAA AR.
:::

:::{warning}
While no further filtering is currently implemented, this may be necessary based on the number of NaN values found on disk (the columns `sum_ondisk_nans_mdi`, `sum_ondisk_nans_hmi`), and based upon the size of the cutout `dim_image_cutout_hmi`, `dim_image_cutout_mdi`
:::

```{glue:figure} two_plots_cutouts
:alt: "AR Cutouts from cotemporal HMI-MDI"
:name: "fig:hmi:cotemporalmagprocess"
MDI (left) and HMI (right) observation of the Sun's magnetic field at {glue}`obs_date`, showing NOAA AR cutouts.
```

An example active region (with {glue}`regions_sunspots` sunspots) is shown below as observed by SoHO/MDI and SDO/HMI. The Mcintosh/Mag Classes are {glue}`regions_hmi_mcintosh`, {glue}`regions_hmi_mag_class`.
<!-- this is the largest by area -->

```{glue:figure} two_plots_cutouts_hmi
:alt: "AR Cutouts from HMI"
:name: "fig:hmi:cotemporalmagprocess_hmi"
Active region cutout with Mcintosh/Mag Classes: {glue}`regions_hmi_mcintosh`, {glue}`regions_hmi_mag_class`
```

To demonstrate how these figures were made, the region classification table at {glue}`obs_date` is shown below. To access data from this table for a selected `time`, you can utilise the `magnetic_class` and `mcintosh_class` columns, which offer the relevant classification classes (when `region_type` is an Active Region, denoted as "AR"). Additionally, for a specific `region_type`, the `processed_path_image_hmi` and `path_image_cutout_hmi` provide the paths to the HMI image and cutout. Alternatively, the `top_right_cutout_hmi` and `bottom_left_cutout_hmi` values specify the coordinates of the top-right and bottom-left corners of the cutout within the processed image.

```{code-cell} python3
:tags: [remove-input]
rct_subset
```

#### Quicklook files

For simplicity, the region classification table provides the columns `quicklook_path_mdi` and `quicklook_path_hmi`, to provide access to quicklook plots. In these plots, the red bounding boxes correspond to `AR` `region_type`, and blue, the `QS` `region_type`, and each region is labelled with their corresponding `number`.

```{code-cell} python3
:tags: [remove-input, remove-output]

from PIL import Image

# just doing Image.open will end up having the docs trying to save and raise: OSError: cannot write mode RGBA as JPEG

with Image.open(rct_subset['quicklook_path_hmi'][0]) as im:
    glue("quicklook_png_hmi", im, display=False)

with Image.open(rct_subset['quicklook_path_mdi'][0]) as im:
    glue("quicklook_png_mdi", im, display=False)
```

|    MDI quicklook   |   HMI quicklook   |
|:-------------------------------------:|:--------------------:|
| {glue:}`quicklook_png_mdi`                       | {glue:}`quicklook_png_hmi`     |


(sec:ardetdataset)=
### Active Region Detection Dataset

To classify active regions, they first need to be detected. By developing an automated solution to active region detection, the aim is to isolated active regions in real-time, which can then be classified, and ultimately utilised as part of an end-to-end machine learning pipeline.

To tackle this as a supervised learning problem, we require a dataset consisting of:

1. **input**: full-disk images (that contain active regions)
2. **output**: bounding boxes surrounding those active regions

As described earlier, the Spaceweather HMI(MDI) Active Region Patches (SHARP/SMARP) provide data segments such as magnetograms and bitmaps for each active region patch. These patches are provided HARP and TARP numbers, similar to NOAA AR numbers, however there is no existing mapping between the two.
To generate the active region detection dataset, we utilise the bitmap segment to extract the bounding box around each patch.

The table below, provides an example of one SHARP bitmap segments against the respective LOS HMI magnetogram cutout.

```{code-cell} python3
:tags: [hide-cell, remove-input, remove-output]

n = -2
# extract the sumpy map
sunpy_map = sunpy.map.Map(hmi_region_table['processed_path'][n])
# submap over the cutout region
smap_hmi = sunpy_map.submap(top_right=hmi_region_table['top_right_cutout'][n], bottom_left=hmi_region_table['bottom_left_cutout'][n]).rotate()
# load the equivalent bitmap
hmi_arc = sunpy.map.Map(hmi_region_table['path_arc'][n])

glue("smap_hmi_plot", plot_map(smap_hmi)[0], display=False)
```

```{code-cell} python3
:tags: [hide-cell, remove-input, remove-output]

import sunpy.map
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.colors import BoundaryNorm
from astropy import units as u

# Create a color map
colors = ['#000000', '#E7EAE9', '#FFFFFF', '#547239', '#FFFFFF']
cmap = ListedColormap(colors)

# Define the boundaries for the color map
bounds = [-0.5, 0.5, 1.5, 2.5, 33.5, 34.5]
norm = BoundaryNorm(bounds, cmap.N)

# Create a custom legend
legend_labels = ['Off-disk', 'Weak field (Outside HARP)', 'Strong field',
                 'Weak field (Inside HARP)']

legend_elements = [Patch(facecolor=cmap(norm(bound)), label=label) for label, bound in zip(legend_labels, bounds)]

# Plot the SunPy map using the specified color map
fig = plt.figure(figsize=(5, 4))

# this is because sharp stuff is done in reference frame and we rotate the Sun
rounded_angle = round(hmi_arc.meta['crota2'] / 90) * 90
hmi_arc_rotated = hmi_arc.rotate(rounded_angle*u.deg)

ax = plt.subplot(projection=hmi_arc_rotated)
im = hmi_arc_rotated.plot(cmap=cmap, norm=norm)

# Add the legend
ax.legend(handles=legend_elements)
ax.set_title("SHARP (HMI) bitmap")

glue("hmi_arc_rot_plot_full", fig, display=False)

```

|    SHARP bitmap   |   HMI magnetogram cutout   |
|:-------------------------------------:|:--------------------:|
| {glue:}`hmi_arc_rot_plot_full`                       | {glue:}`smap_hmi_plot`     |

:::{note}
SHARP/SMARP regions are processed in the instrument frame, where the valid values for SHARPs bitmap regions are ${0, 1, 2, 33, 34}$.
As part of the processing of full-disk MDI/HMI images, these images are rotated to Solar North, through the angle described in the `crota2` metadata keyword.
The interpolation during rotation of these bitmap segments can introduce intermediate values outside the valid labels.

As such, for the visualisation shown above, the rotation for these bitmap segments is defined as:

```
rounded_angle = round(sunpy_map.meta['crota2'] / 90) * 90
rotated_map = sunpy_map.rotate(rounded_angle*u.deg)
```

and may not perfectly align.

:::


#### Dataset creation

To create this dataset we start with the HMI-SHARPS/MDI-SMARPs (see {ref}`sec:dataset_generation`) which merges the HMI full-disk images with the SHARP cutout images (and equivalent for MDI & SMARPs).
With this merged table, each cutout can be opened to extract the bounding box coordinates.

An example set of co-temporal observations from MDI and HMI, with regions extracted from SHARPs/SMARPs, is shown in {numref}`fig:mag_region_detection`.
Figure {numref}`fig:mag_region_detection` shows MDI (left) and HMI (right) with the respective SMARP and SHARP bounds extracted from the bitmap segments.

```{glue:figure} two_plots_cutouts_two
:alt: "AR Cutouts from cotemporal HMI-MDI"
:name: "fig:mag_region_detection"
MDI-HMI observation of the Sun's magnetic field at {glue}`obs_date`, showing NOAA AR cutouts.
```

:::{important}
In this version ({glue}`arccnet_version`) of this dataset, all regions from SMARP/SHARPs are provided. With care, it would be possible to merge the hand-labelled NOAA ARs with the SHARP/SMARP regions based on latitude/longitude to automatically extract SHARP/SMARP regions, however it is very likely many one-to-many relationships exist between these two active region datasets.
:::

The Region Detection table contains MDI and HMI data for full-disk as well as active region cutouts ("arc").
Individual instrument access is enabled through the `instrument` column for a `target_time`.
The `processed_path` column offers the path to the full-disk image, while the `path_arc` column provides the location of the active region classification image.
Alternatively, the  `top_right_cutout` and `bottom_left_cutout` provide bounds of the `path_arc` image, in full-disk data (`processed_path`).

To demonstrate how these figures were made, the region detection table is shown below for {glue:}`obs_date`.
Here, the `target_time` is shared between two instruments and denotes the time for which the full-disk and cutout data was requested.
Each instrument, denoted by the `instrument` column contains its own respective `datetime` reference that corresponds to the full-disk observation time.
The `processed_path` column provides the full-disk image for a given `target_time` and `instrument`, while the cutout can be accessed through `path_arc`, or alternatively the bounding box can be directly from the full-disk `processed_path` image using the `top_right_cutout`/`bottom_left_cutout`

Finally, the `record_TARPNUM_arc` and `record_HARPNUM_arc` provide the TARP/HARP identifiers for each active region (similar to NOAA AR Numbers).

```{code-cell} python3
:tags: [remove-input]
rdt_subset
```

:::{note}
Columns with the suffix `_arc` reference the equivalent for the active region cutouts.
:::

##### Simplifying the data: HARP/NOAA

```{code-cell} python3
:tags: [hide-cell, remove-input, remove-output]

from arccnet.catalogs.utils import retrieve_harp_noaa_mapping

classification_file = Path(config["paths"]["data_root"]) / "04_final" / "mag" / "region_extraction" / "region_classification.parq"
arclass = QTable.read(classification_file)
ar = arclass[arclass['region_type'] == 'AR']
ar = ar[~ar['quicklook_path_hmi'].mask]
ar['NOAA'] = ar['number']
ar['target_time'] = ar['time']

merged_filtered_path = Path(config["paths"]["data_root"]) / "04_final" / "mag" / "region_detection" / "region_detection_noaa-harp.parq"
merged_filtered_data = QTable.read(merged_filtered_path)
merged_filtered = merged_filtered_data[~merged_filtered_data['filtered']]
```

```{code-cell} python3
:tags: [hide-cell, remove-input, remove-output]

group = merged_filtered[merged_filtered['target_time'] == cotemporal_obs_date_time]

sunpy_map = sunpy.map.Map(group['processed_path'][0])
regions = group

fig = plt.figure(figsize=(10, 4))

# Assign different projections to each subplot
ax = fig.add_subplot(1, 1, 1, projection=sunpy_map)

# Set the colormap limits for both maps
vmin, vmax = -1499, 1499
sunpy_map.plot_settings["norm"].vmin = vmin
sunpy_map.plot_settings["norm"].vmax = vmax

# Plot HMI and MDI maps on the respective subplots
sunpy_map.plot(axes=ax, cmap="hmimag")

# Loop through region_table and draw quadrangles for both maps
for row in regions:
    print(row["bottom_left_cutout"])
    sunpy_map.draw_quadrangle(row["bottom_left_cutout"], axes=ax, top_right=row["top_right_cutout"])

glue("harp_noaa_hmi", fig, display=False)
```

To obtain bounding boxes that are exclusive to a single NOAA active regions, we utilise the SHARP regions, and the HARP-to-NOAA mapping that is continually updated at <http://jsoc.stanford.edu/doc/data/hmi/harpnum_to_noaa/all_harps_with_noaa_ars.txt> (further work has been performed by {cite:t}`2020NatSD...7..227A` and references therein). This text file describes the HARP-to-NOAA mapping, where each line consists of a single HARP region, and one-to-many NOAA active region numbers. With this text file, and generate a table where one row contains a single HARP-to-NOAA mapping. An additional column is added to indicate how many NOAA regions are associated with a particular HARP. An example subset of this table is shown below where there are {glue:}`len_rhnm` HARP-NOAA combinations.

```{code-cell} python3
:tags: [remove-input]
rhnm = retrieve_harp_noaa_mapping()

glue("rhnm", rhnm[230:235], display=True)
glue("len_rhnm", len(rhnm), display=False)
```

Merging this with the previous table, and further with the active region classification table, results in the following table (after a number of columns are dropped):

```{code-cell} python3
:tags: [remove-input]
merged_filtered_data

glue("merged_filtered_data", merged_filtered_data, display=True)
glue("len_merged_filtered_data", len(merged_filtered_data), display=False)
glue("len_merged_filtered", len(merged_filtered), display=False)
```

:::{important}
As there is often a one-to-many relation between HARP and NOAA, we may wish to subselect only those full-disk observations that contain a set of unique HARP-to-NOAA mappings, as a single bounding box would not describe a single active region. Programmatically, that is to say:

```{code-block} python3
# Identify dates to drop
joined_table["filtered"] = False
grouped_table = joined_table.group_by("processed_path")
for date in grouped_table.groups:
    if any(date["NOAANUM"] > 1):
        date["filtered"] = True
```

The full table shown above (including regions and dates that contain multiple ) includes {glue:}`len_merged_filtered_data` rows, with {glue:}`len_merged_filtered` rows that correspond to HMI full-disk observations that only includes dates where the regions have a one-to-one mapping (e.g. the data is filtered on the `filtered` column). An example of this is shown below, where the full-disk HMI image is shown with a HARP cutout. In comparison to {numref}`fig:mag_region_detection`, all HARP regions not associated with NOAA active regions are not included.


```{glue:figure} harp_noaa_hmi
:alt: "..."
:name: "fig:hmi:harp_noaa_hmi"
HMI full-disk image with HARP cutout associated with NOAA active region.
```
:::

The data can be visualised for both McIntosh and Magnetic class, as both the full dataset, and the filtered subset:

```{code-cell} python3
:tags: [remove-input, remove-output]

from arccnet.catalogs.active_regions import HALE_CLASSES, MCINTOSH_CLASSES

df_names = pd.DataFrame({'mcintosh_class': MCINTOSH_CLASSES})

# !TODO investigate this. There are a small number of missing values in the `merged_filtered` compared
# to the `ar` table, which may be due to the HARP-NOAA relationship
# df = ar['target_time','mcintosh_class'].to_pandas()
# dists_df = df['mcintosh_class'].value_counts(normalize=False)#
df = merged_filtered_data['datetime','mcintosh_class','filtered'].to_pandas()
dists_df = df['mcintosh_class'].value_counts(normalize=False)

md_df = merged_filtered_data['datetime','mcintosh_class','filtered'].to_pandas()
md_df = md_df[~md_df['filtered']]
md_df = md_df['mcintosh_class'].value_counts(normalize=False)

merged_df = pd.merge(dists_df, md_df, left_index=True, right_index=True, how='outer', suffixes=['_original', '_subset'])
merged_df = pd.merge(df_names, merged_df, left_on='mcintosh_class', right_index=True, how='outer')

# Fill NaN values with 0, assuming NaN means no count for that class in a particular dataframe
merged_df = merged_df.fillna(0)
merged_df.set_index('mcintosh_class', inplace=True)
merged_df.sort_values('count_original', inplace=True, ascending=False)

fig, ax = plt.subplots(figsize=(12, 4))
merged_df.plot(kind='bar', ax=ax)
ax.set_yscale('log')

glue("mcintosh_plot", fig, display=False)
```

```{code-cell} python3
:tags: [remove-input, remove-output]
# Mt Wilson Magnetic classification

df_names = pd.DataFrame({'magnetic_class': HALE_CLASSES})

# !TODO investigate this. There are a small number of missing values in the `merged_filtered` compared
# df = ar['target_time','magnetic_class'].to_pandas()
# dists_df = df['magnetic_class'].value_counts(normalize=False)
df = merged_filtered_data['datetime','magnetic_class','filtered'].to_pandas()
dists_df = df['magnetic_class'].value_counts(normalize=False)

md_df = merged_filtered_data['datetime','magnetic_class','filtered'].to_pandas()
md_df = md_df[~md_df['filtered']]
md_df = md_df['magnetic_class'].value_counts(normalize=False)

merged_df2 = pd.merge(dists_df, md_df, left_index=True, right_index=True, how='outer', suffixes=['_original', '_subset'])
merged_df2 = pd.merge(df_names, merged_df2, left_on='magnetic_class', right_index=True, how='outer')

# Fill NaN values with 0, assuming NaN means no count for that class in a particular dataframe
merged_df2 = merged_df2.fillna(0)
merged_df2.set_index('magnetic_class', inplace=True)
merged_df2.sort_values('count_original', inplace=True, ascending=False)

fig, ax = plt.subplots(figsize=(12, 4))
merged_df2.plot(kind='bar', ax=ax)
ax.set_yscale('log')
ax.tick_params(axis='x', rotation=45)

glue("magnetic_plot", fig, display=False)
```

```{glue:figure} magnetic_plot
:alt: "..."
:name: "fig:magnetic_plot"
Original Hale classes (blue), and number of classes post-filtering (orange).
```

```{glue:figure} mcintosh_plot
:alt: "..."
:name: "fig:mcintosh_plot"
Original McIntosh classes (blue), and number of classes post-filtering (orange). The filtering of full-disk images to include those with only one-to-one mappings between HARP bounding boxes and NOAA active regions reduces some classes to zero.
```

```{code-cell} python3
:tags: [remove-input, remove-output]

# Assuming 'merged_df' has the structure: mcintosh_class (as index), count_original, count_subset

# Define filtering criteria using a dictionary with regex patterns
filter_criteria_z = {
    'A': '^A',
    'B': '^B',
    'C': '^C',
    'D': '^D',
    'E': '^E',
    'F': '^F',
    'H': '^H'
}

filter_criteria_p = {
    'x': '^.x',
    'r': '^.r',
    's': '^.s',
    'a': '^.a',
    'h': '^.h',
    'k': '^.k'
}

filter_criteria_c = {
    'x': '^..x',
    'o': '^..o',
    'i': '^..i',
    'c': '^..c'
}

def create_filtered_dataframe(df, filter_criteria, title):
    filtered_counts_original = {}
    filtered_counts_subset = {}

    for label, pattern in filter_criteria.items():
        filtered_counts_original[label] = df[df.index.str.contains(pattern)]['count_original'].sum()
        filtered_counts_subset[label] = df[df.index.str.contains(pattern)]['count_subset'].sum()

    filtered_df = pd.DataFrame({
        title: list(filtered_counts_original.keys()),
        'count_original': list(filtered_counts_original.values()),
        'count_subset': list(filtered_counts_subset.values())
    })

    filtered_df.set_index(title, inplace=True)  # Set the desired index

    return filtered_df

# Create DataFrames for 'Z', 'p', and 'c'
df_z = create_filtered_dataframe(merged_df, filter_criteria_z, 'Z-values')
df_p = create_filtered_dataframe(merged_df, filter_criteria_p, 'p-values')
df_c = create_filtered_dataframe(merged_df, filter_criteria_c, 'c-values')

# Plot the DataFrames with shared y-axis
fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)

dfs = [df_z, df_p, df_c]

for i, df in enumerate(dfs):
    # Plot for each DataFrame
    width = 0.4
    df.plot(kind='bar', ax=axes[i])

    # Customize plot settings
    axes[i].set_yscale('log')
    axes[i].set_ylabel('Count')
    axes[i].set_ylim(10,40000)
    axes[i].legend()
    axes[i].tick_params(axis='x', rotation=0)

# Show the plots
plt.tight_layout()
# plt.show()

glue("zpc_plot", fig, display=False)
```

As discussed on <https://www.sidc.be/educational/classification.php>, the three-component McIntosh classification {cite:p}`mcintosh1990` is based on the general form 'Zpc', where 'Z' is the modified Zurich Class, 'p' describes the penumbra of the principal spot, and 'c' describes the distribution of spots in the interior of the group, we can plot the frequency of each component separately

```{glue:figure} zpc_plot
:alt: "..."
:name: "fig:zpc_plot"
Original Mcintosh classes pre- (blue) and post-filtering (orange) for each component of the three-component classification. Compared to plotting these classes individually, this provides a larger sample for each letter.
```

The NOAA ARs have been merge the hand-labelled NOAA ARs (the SRS table) and the SHARP/SMARP tables. As there is a one-to-many relationship between HARP and NOAA, we utilised the mapping provided by Stanford, and filter the full-disk images that contain HARP regions with one-to-one mappings. Accessing only the rows with `filtered == True` removes the many-to-one problem, where a single HARP box can contain upto six NOAA active regions, but significantly reduces the amount of labelled data, and due to the nature, specifically reduces the number of complex regions, as shown in the previous plots.

## Summary

In version {glue:}`arccnet_version` of this dataset, we described the input data, and the process to generate initial versions of the Active Region Classification and Active Region Detection datasets.

### Active Region Classification

For AR Classification, Active Regions were obtained by extracting latitude/longitude information from daily NOAA SRS files, and merged with full-disk MDI/HMI observations that were requested at the validity-time of the NOAA SRS files.
Utilising the latitude/longitude of each NOAA AR from the SRS files, the active regions were extracted (`region_type == 'AR'` in the final table) alongside a set of random quiet sun regions that describe locations on the Sun that are not NOAA ARs (`region_type == 'QS'`).

As shown in the {ref}`sec:arclassdataset` section, each active region cutout has associated magnetic/mcintosh classes and other metadata extracted from the the daily SRS text files.
These provide an input and classification labels for a supervised learning approach to active region classification

### Active Region Detection

For AR Detection, Regions were obtained for MDI and HMI by querying the SMARP/SHARP series. These series provide active regions that have been automatically detected, and are labelled with TARP/HARP Numbers, however, not all regions are associated with NOAA ARs.

As shown in the {ref}`sec:ardetdataset` section, each SHARP/SMARP region is shown on the full-disk images with an example an example HMI bitmap segment also shown.
For version {glue:}`arccnet_version`, the Region Detection table provides the full-disk image, and the bounds of the bounding boxes to train a supervised region detection model.

To isolate only the NOAA ARs, we merge the hand-labelled NOAA ARs (the SRS table) and the SHARP/SMARP tables. As there is a one-to-many relationship between HARP and NOAA, we utilise the mapping provided by Stanford, and keep only the full-disk images that contain HARP regions with one-to-one mappings. This removes the many-to-one problem, where a single HARP box can contain upto six NOAA active regions, but significantly reduces the amount of labelled data, and due to the nature, specifically reduces the number of complex regions.

## Bibliography

```{bibliography}
:filter: docname in docnames
```
