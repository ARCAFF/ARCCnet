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

from myst_nb import glue
from datetime import datetime
from pathlib import Path
from astropy.table import QTable
import numpy as np
import sunpy
import sunpy.map
from arccnet import config
from arccnet.visualisation.data import (
    plot_hmi_mdi_availability,
    plot_col_scatter,
    plot_col_scatter_single,
    plot_maps,
    plot_maps_regions,
)


# Load HMI and MDI data
hmi_table = QTable.read(Path(config["paths"]["data_root"]) / "02_intermediate" / "mag" / "hmi_results.parq").to_pandas()
mdi_table = QTable.read(Path(config["paths"]["data_root"]) / "02_intermediate" / "mag" / "mdi_results.parq").to_pandas()

# set the date range regardless of what has been requested.
start_date = datetime(1995, 1, 1)
end_date = datetime.now()
mag_availability = plot_hmi_mdi_availability(hmi_table, mdi_table, start_date, end_date)

glue("hmi_mdi_availability", mag_availability[0], display=False)
glue("start_date", start_date.strftime("%d-%h-%Y"), display=False)
glue("end_date", end_date.strftime("%d-%h-%Y"), display=False)

cdelt = plot_col_scatter([mdi_table, hmi_table], column="CDELT1", colors=["red", "blue"])
glue("hmi_mdi_cdelt", cdelt[0], display=False)

dsun = plot_col_scatter_single([mdi_table, hmi_table], column="DSUN_OBS", colors=["red", "blue"])
glue("hmi_mdi_dsun", dsun[0], display=False)

# create co-temporal observations
region_table = Path(config["paths"]["data_root"]) / "04_final" / "mag" / "region_detection" / "region_detection.parq"
region_detection_table = QTable.read(region_table)

# Group the table by the "time" column
# Iterate through the groups to find the first group with both 'mdi' and 'hmi' in the "instrument" column
for group in region_detection_table.group_by('target_time').groups:
    if 'MDI' in group['instrument'] and 'HMI' in group['instrument']:
        obs_date_time = np.unique(group['target_time'])[0]
        break

obs_date = obs_date_time.to_datetime().strftime("%Y-%m-%d")

mdi = sunpy.map.Map(region_detection_table[(region_detection_table['target_time'] == obs_date_time) & (region_detection_table['instrument'] == 'MDI')]['path'][0])
hmi = sunpy.map.Map(region_detection_table[(region_detection_table['target_time'] == obs_date_time) & (region_detection_table['instrument'] == 'HMI')]['path'][0])
mag, _ = plot_maps(mdi, hmi)

hmi_processed = QTable.read(Path(config["paths"]["data_root"]) / "03_processed" / "mag" / "hmi_processed.parq")
mdi_processed = QTable.read(Path(config["paths"]["data_root"]) / "03_processed" / "mag" / "mdi_processed.parq")
processed_hmi_file = sunpy.map.Map(hmi_processed[hmi_processed['target_time'] == obs_date_time]['processed_path'].data.data[0])
processed_mdi_file = sunpy.map.Map(mdi_processed[mdi_processed['target_time'] == obs_date_time]['processed_path'].data.data[0])
mag_processed, _ = plot_maps(processed_mdi_file, processed_hmi_file)

# -- AR Classification
region_classification_table = QTable.read(Path(config["paths"]["data_root"]) / "04_final" / "mag" / "region_extraction" / "region_classification.parq")
rct_subset = region_classification_table[region_classification_table['time'] == obs_date_time]

map_one = sunpy.map.Map(rct_subset['processed_path_image_hmi'][0])
regions_one = rct_subset['region_type', 'top_right_cutout_hmi', 'bottom_left_cutout_hmi', 'number', 'magnetic_class', 'carrington_longitude', 'area', 'mcintosh_class']
regions_one.rename_columns(
    ['top_right_cutout_hmi', 'bottom_left_cutout_hmi'],
    ['top_right_cutout', 'bottom_left_cutout'],
)
regions_one = regions_one[regions_one["region_type"] == "AR"]
regions_hmi_mcintosh = regions_one['mcintosh_class'][0]
regions_hmi_mag_class = regions_one['magnetic_class'][0]

map_two = sunpy.map.Map(rct_subset['processed_path_image_mdi'][0])
regions_two = rct_subset['region_type', 'top_right_cutout_mdi', 'bottom_left_cutout_mdi', 'number', 'magnetic_class', 'carrington_longitude', 'area', 'mcintosh_class']
regions_two.rename_columns(
    ['top_right_cutout_mdi', 'bottom_left_cutout_mdi'],
    ['top_right_cutout', 'bottom_left_cutout'],
)
regions_two = regions_two[regions_two["region_type"] == "AR"]
regions_mdi_mcintosh = regions_two['mcintosh_class'][0]
regions_mdi_mag_class = regions_two['magnetic_class'][0]

# -- setting the values to be the zeroth element (and plotting the associated cutouts)
glue("regions_hmi_mcintosh", regions_hmi_mcintosh, display=False)
glue("regions_hmi_mag_class", regions_hmi_mag_class, display=False)
glue("regions_mdi_mcintosh", regions_mdi_mcintosh, display=False)
glue("regions_mdi_mag_class", regions_mdi_mag_class, display=False)

import matplotlib.pyplot as plt
# mag_cutouts_mdi = plt.figure(figsize=(7.5, 3))
# smap_mdi = map_two.submap(top_right=regions_two['top_right_cutout'][0], bottom_left=regions_two['bottom_left_cutout'][0])
# smap_mdi.plot(cmap="hmimag")

mag_cutouts_hmi = plt.figure(figsize=(7.5, 3))
smap_hmi = map_one.submap(top_right=regions_one['top_right_cutout'][0], bottom_left=regions_one['bottom_left_cutout'][0])
smap_hmi.plot(cmap="hmimag")

map_cutouts, _ = plot_maps_regions(map_two, regions_two, map_one, regions_one, **{
    'edgecolor': 'black',
    'linestyle': '--',
})

# -- AR Detection
rdt_subset = region_detection_table[region_detection_table['target_time'] == obs_date_time]

hmi_region_table = rdt_subset[rdt_subset['instrument'] == 'HMI']
hmi_map = sunpy.map.Map(hmi_region_table["processed_path"][0])
mdi_region_table = rdt_subset[rdt_subset['instrument'] == 'MDI']
mdi_map = sunpy.map.Map(mdi_region_table["processed_path"][0])

map_cutouts_two, _ = plot_maps_regions(mdi_map, mdi_region_table, hmi_map, hmi_region_table, **{
    'edgecolor': 'black',
    'linestyle': '--',
})

glue("two_plots", mag, display=False)
glue("two_plots_processed", mag_processed, display=False)
glue("two_plots_cutouts", map_cutouts, display=False)
# glue("two_plots_cutouts_mdi", mag_cutouts_mdi, display=False)
glue("two_plots_cutouts_hmi", mag_cutouts_hmi, display=False)
glue("two_plots_cutouts_two", map_cutouts_two, display=False)
glue("rct_subset_ar", rct_subset[rct_subset["region_type"] == "AR"], display=True)
glue("rdt_subset", rdt_subset, display=True)
glue("hmi_plot", hmi, display=False)
glue("mdi_plot", mdi, display=False)
glue("obs_date", obs_date, display=False)
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

To train active region classification and detection models, we also retrieve line-of-sight magnetograms once-per-day, from 1995 - 2022, synchronized with the validity of NOAA SRS reports at 00:00 UTC (issued at 00:30 UTC).

## Data Sources and Observations

The observations from SoHO/MDI (1995 - 2011; {cite:t}`Scherrer1995,Domingo1995`) and SDO/HMI (2010 - present {cite:p}`Scherrer2012,Pesnell2012`) are retrieved from the Joint Science Operations Center (JSOC) at Stanford University for 1996 - 2022 inclusive, leaving 2023 as unseen data.

The availability of images in our dataset is shown for this period in Figure {numref}`fig:mag:availability`, where between 2010 and 2011, there are co-temporal observations of the line-of-sight field.

```{glue:figure} hmi_mdi_availability
:alt: "HMI-MDI Availability"
:name: "fig:mag:availability"
HMI-MDI coverage diagram from {glue}`start_date` to {glue}`end_date`.
```

### Co-temporal observations

For the problems of active region classification and detection, the observed distributions of active region classes across a given solar cycle (and therefore instrument) are not uniform, and the number of observed active regions varies across solar cycles themselves.

Datasets that combine observations from multiple observatories allow us to understand dynamics across solar cycles and play a cruicial role in increasing the number of available samples for training machine learning models. However, while improvements to instrumentation can fuel scientific advancements, for studies over the typical observatory lifespan, the varying spatial resolutions, cadences, and systematics (to name a few) make their direct comparisons challenging.

The expansion of the SHARP series {cite:p}`Bobra2014` to SoHO/MDI (SMARPs; {cite:t}`Bobra2021`) has tried to negate this with a tuned detection algorithm to provide similar active region cutouts (and associated parameters) across two solar cycles. Other authors have incorporated advancements in the state-of-the-art for image translation to cross-calibrate data, however, out-of-the-box, these models generally prefer perceptual similarity. Importantly, progress has been made towards physically-driven approaches for instrument cross-calibration/super-resolution (e.g. Munoz-Jaramillo et al 2023 (in revision)) that takes into account knowledge of the underlying physics.

Initially, we will utilise each instrument individually, before expanding to cross-calibration techniques. Examples of co-temporal data (for {glue}`obs_date`) is shown below with SunPy map objects in Figure {numref}`fig:mdi:cotemporalmag`.

```{glue:figure} two_plots
:alt: "Cotemporal HMI-MDI"
:name: "fig:hmi:cotemporalmag"
MDI-HMI observation of the Sun's magnetic field at {glue}`obs_date`.
```

#### Instrumental/Orbital Effects on Data

While there are noticeable visual differences (e.g. resolution and noise properties), there are a number of subtle differences between these instruments that can be observed in the metadata, and should be accounted for. Both instruments are located at different positions in space, and at different distances from the Sun, which vary as the Earth orbits around the Sun.

To demonstrate some of these instrumental and orbital differences Figure {numref}`hmi_mdifig:mag:cdelt` shows the image scale in the x-axis of the image plane as a function of time for SDO/HMI and SoHO/MDI, while Figure {numref}`fig:mag:dsun` demonstrates how the radius of the Sun varies during normal observations.

```{glue:figure} hmi_mdi_cdelt
:alt: "HMI-MDI CDELT1"
:name: "fig:mag:cdelt"
CDELT1 (image scale in the x-direction [arcsec/pixel]) from {glue}`start_date` to {glue}`end_date` for SDO/HMI (top, blue) and SoHO/MDI (bottom, red).
```

```{glue:figure} hmi_mdi_dsun
:alt: "HMI-MDI DSUN"
:name: "fig:mag:dsun"
DSUN_OBS (distance from instrument to sun-centre [metres]) from {glue}`start_date` to {glue}`end_date` for SDO/HMI (blue) and SoHO/MDI (red).
```

While these can be corrected through data preparation and processing, including the reprojection of images to be as-observed from a set location along the Sun-Earth line, complex relationships mean it may be necessary to use machine learning models (such as the cross-calibration approach mentioned previously) to prepare data.

## Data Processing

### Full-disk HMI/MDI

For this v0.1 of the dataset a preliminary data processing routine is applied to full-disk HMI and MDI (as shown below) to include

1. Rotation to Solar North
2. Removal (and zero) off-disk data


```{glue:figure} two_plots_processed
:alt: "Processed Cotemporal HMI-MDI"
:name: "fig:hmi:cotemporalmagprocess"
MDI-HMI observation of the Sun's magnetic field at {glue}`obs_date`.
```

Compared to Figure {numref}`fig:mdi:cotemporalmag`, Figure {numref}`fig:hmi:cotemporalmagprocess` shows these corrections applied, however, the removal of off-disk data in MDI is incomplete, leaving NaN values on the limb (white). As we progress towards v1.0, the processing pipeline will also be expanded to include additional corrections such as

* Reprojection of images to a fixed point at 1 AU along the Sun-Earth line
* Filtering according to the hexadecimal `QUALITY` flag.

Currently the correct choice of reprojection is still under consideration, and the handling of conversion of hexadecimal to float needs to be clarified. Additional data processing steps may include

1. Instrument inter-calibration (and super-resolution) -- Due to optically distortion in MDI, even with reprojection to a common coordinate frame, perfect alignment of images is not possible between these instruments.
2. Addition of Coronal Information e.g.

* Inclusion and alignment of EUV images
* Generation of Differential Emission Measure (DEM) maps {cite:p}`2012A&A...539A.146H,2015ApJ...807..143C`

3. Magnetic Field Extrapolation

### HMI/MDI Cutouts (SHARP/SMARP)

As SHARP/SMARP are already at level 1.8, these images only need correcting for rotation. For more discussion on the generation of SHARP/SMARP data, see <https://github.com/mbobra/SHARPs>, <https://github.com/mbobra/SMARPs>.

Currently we obtain bitmap images as a preliminary method to extract regions around NOAA ARs.

## Datasets

### Active Region Classification Dataset

The AR Classification dataset contains cutouts associated with NOAA ARs, and "quiet sun" regions. Currently, in this version, a "quiet sun" region is any region that is not a NOAA AR. The ARs at {glue}`obs_date` are shown below

```{glue:figure} two_plots_cutouts
:alt: "AR Cutouts from cotemporal HMI-MDI"
:name: "fig:hmi:cotemporalmagprocess"
MDI-HMI observation of the Sun's magnetic field at {glue}`obs_date`, showing NOAA AR cutouts.
```

where the cutout from HMI is shown with Mcintosh/Hale Classes: {glue}`regions_hmi_mcintosh`, {glue}`regions_hmi_mag_class`.

<!-- ```{glue:figure} two_plots_cutouts_mdi
:alt: "AR Cutouts from MDI"
:name: "fig:hmi:cotemporalmagprocessmdi"
Active region cutout with Mcintosh/Hale Classes: {glue}`regions_mdi_mcintosh`, {glue}`regions_mdi_mag_class
``` -->

```{glue:figure} two_plots_cutouts_hmi
:alt: "AR Cutouts from HMI"
:name: "fig:hmi:cotemporalmagprocesshmi"
Active region cutout with Mcintosh/Hale Classes: {glue}`regions_hmi_mcintosh`, {glue}`regions_hmi_mag_class`
```

The Rection Classification Table for ARs only (`region_type` = "AR") at {glue}`obs_date` is as follows

```{glue:} rct_subset_ar
```

To access the data from this table, for a chosen `time`, the `magnetic_class` and `mcintosh_class` columns provide the appropriate classification classes (if `region_type` is an Active Region, "AR"). Additionally, for any given `region_type`, the `processed_path_image_hmi` and `path_image_cutout_hmi` provide the HMI image and cutout paths, with the `top_right_cutout_hmi` and `bottom_left_cutout_hmi` values providing the top-right, and bottom-left coordinates of the cutout in the processed image.

### Region Detection Dataset

The Region Detection dataset has cotemporal observations, with regions extracted from SHARPs/SMARPs, as shown below

```{glue:figure} two_plots_cutouts_two
:alt: "AR Cutouts from cotemporal HMI-MDI"
:name: "fig:hmi:cotemporalmagprocess"
MDI-HMI observation of the Sun's magnetic field at {glue}`obs_date`, showing NOAA AR cutouts.
```

where the Region Detection Table shows the HMI and MDI data for the full-disk, and active region cutout ("arc") data:

```{glue:} rdt_subset
```

To access the data, for a given `target_time`, data can be split through the `instrument` column, where the `processed_path` provides the path of the full-disk image, `path_arc` provides the location of the active region classification image, or alternatively, the bounds (that correspond to the `processed_path`) can be requested through `top_right_cutout` and `bottom_left_cutout`.

## Bibliography

```{bibliography}
```
