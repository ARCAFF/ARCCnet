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
import sunpy
import sunpy.map
from arccnet import config
from arccnet.visualisation.data import (
    plot_hmi_mdi_availability,
    plot_col_scatter,
    plot_col_scatter_single,
    plot_maps,
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
# !TODO tidy this up when we have processed data by picking a random date.
srs_hmi_mdi_merged_file = Path(config["paths"]["data_root"]) / "04_final" / "mag" / "srs_hmi_mdi_merged.parq"
t = QTable.read(srs_hmi_mdi_merged_file)
first_row = t[~t['path_mdi'].mask & ~t['path_hmi'].mask][0]
hmi = sunpy.map.Map(first_row['path_hmi'])
mdi = sunpy.map.Map(first_row['path_mdi'])
obs_date = first_row['target_time_hmi'].to_datetime().strftime("%Y-%m-%d")

mag, _ = plot_maps(mdi, hmi)
glue("two_plots", mag, display=False)
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

For this v0.1 of the dataset a preliminary data processing routine is applied to full-disk HMI and MDI to include

1. Rotation to Solar North
2. Removal (and zero) off-disk data

as shown below [placeholder for processing figure]

As we progress towards v1.0, the processing pipeline will be expanded to include additional corrections e.g.

* Reprojection of images to a fixed point at 1 AU along the Sun-Earth line
* Filtering according to the hexadecimal \texttt{QUALITY} flag.

where currently the correct choice of reprojection is still under consideration, and the handling of conversion from hexadecimal to float needs to be clarified. Additional data processing steps may include

1. Instrument inter-calibration (and super-resolution) -- Due to optically distortion in MDI, even with reprojection to a common coordinate frame, perfect alignment of images is not possible between these instruments.
2. Addition of Coronal Information e.g.

* Inclusion and alignment of EUV images
* Generation of Differential Emission Measure (DEM) maps {cite:p}`2012A&A...539A.146H,2015ApJ...807..143C`

3. Magnetic Field Extrapolation

### HMI/MDI Cutouts (SHARP/SMARP)

As SHARP/SMARP are already at level 1.8, these images only need correcting for rotation. For more discussion on the generation of SHARP/SMARP data, see <https://github.com/mbobra/SHARPs>, <https://github.com/mbobra/SMARPs>.

Currently we obtain bitmap images as a preliminary method to extract regions around NOAA ARs.

## Bibliography

```{bibliography}
```
