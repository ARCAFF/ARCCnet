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
from astropy.table import QTable
from arccnet.visualisation.data import plot_hmi_mdi_availability, plot_col_scatter, plot_col_scatter_single

# Load HMI and MDI data
hmi_table = QTable.read('/Users/pjwright/Documents/work/backup/refactor/ARCCnet/data/02_intermediate/mag/hmi_results.parq').to_pandas()
mdi_table = QTable.read('/Users/pjwright/Documents/work/backup/refactor/ARCCnet/data/02_intermediate/mag/mdi_results.parq').to_pandas()
start_date = datetime(1995,1,1)
end_date = datetime.now()
mag_availability = plot_hmi_mdi_availability(hmi_table, mdi_table, start_date, end_date)

glue("hmi_mdi_availability", mag_availability[0], display=False)
glue("start_date", start_date.strftime('%d-%h-%Y'), display=False)
glue("end_date", end_date.strftime('%d-%h-%Y'), display=False)

cdelt = plot_col_scatter([mdi_table,hmi_table], column='CDELT1', colors=['red','blue'])
glue("hmi_mdi_cdelt", cdelt[0], display=False)

dsun = plot_col_scatter_single([mdi_table,hmi_table], column='DSUN_OBS', colors=['red','blue'])
glue("hmi_mdi_dsun", dsun[0], display=False)

```

# Active Region Classification

## Introduction

In addition to the AR classifications, we retrieve line-of-sight magnetogram data from Michelson Doppler Imager on Solar and Heliospheric Observatory (SoHO/MDI, 1996 - 2011 {cite:t}`Scherrer1995,Domingo1995`) and the Helioseismic and Magnetic Imager on-board the Solar Dynamics Observatory (SDO/HMI, 2010 - present, {cite:t}`Scherrer2012,Pesnell2012`) once-per-day, synchronized with the validity of SRS reports at 00:00 UTC (issued at 00:30 UTC).

```{glue:figure} hmi_mdi_availability
:alt: "HMI-MDI Availability"
:name: "fig:mag:availability"
HMI-MDI coverage diagram from {glue}`start_date` to {glue}`end_date`.
```

## Data Sources and Observations

Plot number of active regions as a function of time.

### Co-temporal observations

To analyze the magnetic field data effectively, we need to understand the overlap and differences in observations between the MDI and HMI instruments.

Ultimately, both instruments have different resolution, cadences, and systematics, making their direct comparison difficult. The combination of SHARP/SMARP has tried to negate this, but cross-calibration of magnetograms is required for a homogeneous dataset (e.g. Munoz-Jaramillo et al (in revision).)

While an initial version of active region classification can accept both instruments, a homogeneous dataset unlocks a vast amount of AR data.

#### Data vs Time

Create a visual representation (plot) illustrating how the availability of data from both HMI and MDI evolves over time. This will help us identify periods of overlap and any gaps in data.

#### HMI vs MDI (with active region cutouts)

Include a comparative analysis of the magnetic field data obtained from HMI and MDI during their co-temporal periods. Visualizations such as side-by-side images or data comparisons can provide valuable insights into the similarities and differences between the two instruments.

#### Instrumental Effects on Data

```{glue:figure} hmi_mdi_cdelt
:alt: "HMI-MDI CDELT1"
:name: "fig:mag:cdelt"
CDELT1 (image scale in the x direction [arsec/pixel]) from {glue}`start_date` to {glue}`end_date` for SDO/HMI (top) and SoHO/MDI (bottom).
```

```{glue:figure} hmi_mdi_dsun
:alt: "HMI-MDI DSUN"
:name: "fig:mag:dsun"
DSUN_OBS (distance from instrument to sun-centre) from {glue}`start_date` to {glue}`end_date` for SDO/HMI (top) and SoHO/MDI (bottom).
```

Discuss any instrumental factors or phenomena that might affect the quality of the data, such as changes in solar radius, which can impact the observed magnetic field measurements.

### Solar Cycle Behavior and Active Regions

Explain how active regions on the solar disk are identified and tracked. Additionally, create a plot using NOAA SRS-HMI-MDI merged data to visualize the number of active regions on the solar disk over time. This plot will help us understand the solar cycle behavior and its relation to active regions.

## Data Processing

Detail the data processing steps, including how we handle data from HMI, MDI, and any cutouts (e.g., SHARP/SMARP {cite:p}`Bobra2014,Bobra2021` ). Describe the purpose and scope of the deliverable in this section.

### HMI Data Processing

Explain the specific data processing steps for HMI data, including any preprocessing, calibration, or data cleaning procedures.

### MDI Data Processing

Outline the data processing steps for MDI data, highlighting any differences or specific considerations compared to HMI data processing.

### Cutouts (e.g., SHARP/SMARP)

If relevant, provide information on how cutout data (e.g., SHARP or SMARP data) is processed and integrated into the analysis.

## Source Data

## Summary

## Bibliography

```{bibliography}
```
