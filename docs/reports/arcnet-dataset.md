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

# Active Region Classification

## Introduction

Sunspot groups or {term}`ARs` classifications were some of the earliest properties used in forecast solar flares, and they are still in use today.
Initially {term}`ARs` were classified using the Zurich system {cite:p}`waldmeier1938`, after Hale's discovered of magnetic fields in sunspots this was followed by the addition of the Hale or Mt Wislon {cite:p}`hale1919,kunzel1965` classification and finally a modified version of Zurich scheme known as the McIntosh classification {cite:p}`mcintosh1990`.
Hale or Mt Wilson and McIntosh classification are still produced daily by space weather forecasting centers such as UKMet and SWPC.
Human operators use solar observations and guides to produce the classification and as such they are subject to variable biases and error.

The first task of {term}`ARCAFF` project is to train neural networks to classify {term}`ARs` in order to do this the first task of any machine learning task is the creation of the training and evaluation dataset in this case the "AR Localisation and Classification ML Datasets".
The task has been broken down into:

1. Active Region Cutout Classification -- given an {term}`AR` cutout produce classifications,
2. Active Region Detection -- given a full disk image produce a bounding boxes and classifications for each {term}`AR`.

To assemble these dataset data on {term}`AR` classification and detected bounding boxes mush be combined with magnetogram observations.

### Purpose and Scope of the deliverable

The purpose of this deliverable is to describe the input data and processes use to create the "AR Localisation and Classification ML Datasets" and also describe the datasets themselves.

### References

### Acronyms

```{glossary}

ARCAFF
 Active Region Classification and Flare Forecasting

AR
 Active Region

ARs
 Active Regions

ML
 Machine Learning

NOAA
 National Oceanic and Atmospheric Administration

SRS
 Solar Region Summary

SWPC
 Space Weather Prediction Center
```

## Source Data

### AR Classifications

The primary source of {term}`AR` classifications are {term}`NOAA`'s {term}`SWPC` {term}`SRS` reports which are jointly prepared by the U.S. Dept. of Commerce and {term}`NOAA`.
{numref}`code:srs-report` shows and example of an {term}`SRS` report the key information is contained in Section I and is:

* 'Nubr' - The NOAA number,
* 'Location' - Sunspot group location, in heliographic degrees latitude and degrees east or west from central meridian, rotated to 00:00 UTC,
* 'Z' - Modified Zurich classification of the group,
* 'Mag Type' - The Hale/Mt Wilson or Magnetic classification of the group.

```{code-block} text
:name: code:srs-report
:caption: Example of an SRS report

:Product: 0428SRS.txt
:Issued: 2012 Apr 28 0030 UTC
# Prepared jointly by the U.S. Dept. of Commerce, NOAA,
# Space Weather Prediction Center and the U.S. Air Force.
#
Joint USAF/NOAA Solar Region Summary
SRS Number 119 Issued at 0030Z on 28 Apr 2012
Report compiled from data received at SWO on 27 Apr
I.  Regions with Sunspots.  Locations Valid at 27/2400Z
Nmbr Location  Lo  Area  Z   LL   NN Mag Type
1459 S19W88   091  0030 Cao  04   04 Beta
1465 S17W53   056  0180 Dai  05   11 Beta-Gamma-Delta
1466 N11W38   041  0090 Cso  07   08 Beta
1467 N11E32   331  0050 Dso  03   04 Beta
1468 N09W30   033  0080 Dao  07   06 Beta
1469 S24E28   335  0050 Dso  10   06 Beta
IA. H-alpha Plages without Spots.  Locations Valid at 27/2400Z Apr
Nmbr  Location  Lo
1461  N10W75   078
II. Regions Due to Return 28 Apr to 30 Apr
Nmbr Lat    Lo
None
```

```{code-cell} python3
:tags: [hide-cell, remove-input, remove-output]

# set working directory to the base of the repo
%cd ../..

from myst_nb import glue
import datetime
from pathlib import Path
from arccnet.catalogs.active_regions.swpc import ClassificationCatalog
from arccnet.visualisation.data import plot_srs_coverage, plot_srs_map, plot_filtered_srs_trace
from arccnet import config

start_date = config["general"]["start_date"]
end_date = config["general"]["end_date"]
pcat = ClassificationCatalog.read(Path(config["paths"]["data_dir_processed"]) / "noaa_srs" / "srs_processed_catalog.parq")
pcat_df = pcat.to_pandas()
srs_coverage_fig, srs_coverage_ax = plot_srs_coverage(pcat)
glue("srs_coverage_fig", srs_coverage_fig, display=False)
glue("start_date", str(start_date), display=False)
glue("end_date", str(end_date), display=False)
glue("srs_expected_no", len(pcat_df.time.unique()))
glue("srs_missing_no", pcat_df.url.isnull().sum())
glue("srs_error_no",  len(pcat_df[pcat_df.loaded_successfully == False].path.unique()) - 1)
glue("srs_good_no", len(pcat_df[pcat_df.loaded_successfully == True].url.unique()))


srs_map_fig, srs_map_ax = plot_srs_map(pcat)
glue("srs_map_fig", srs_map_fig, display=False)

srs_trace_fig, srs_trace_ax = plot_filtered_srs_trace(pcat, numbers={13173: 'Ok', 7946: 'Bad Lon Rate', 8090: 'Bad Lat Rate', 8238: 'Bad Lon, Lat Rates'})
glue("srs_trace_fig", srs_trace_fig, display=False)


pcat_df = pcat.to_pandas()
all_ars_mask = pcat_df.id == "I"
ars = pcat_df[all_ars_mask]

glue("srs_ars_tot_no", len(ars.number))
glue("srs_ars_tot_unique_no", len(ars.number.unique()))
glue("srs_ars_bad_no", len(ars[ars.filtered == True].number))
glue("srs_ars_bad_unique_no", len(ars[ars.filtered == True].number.unique()))
glue("srs_ars_good_no", len(ars[ars.filtered == False].number))
glue("srs_ars_good_unique_no", len(ars[ars.filtered == False].number.unique()))
```

{term}`SRS` data is queried, downloaded using Sunpy's unified search and retrieval tool `Fido` and is parsed using the `read_srs` method, which has been updated as part of this work to handle files pre-2000.
{numref}`fig:srs:coverage` shows the results of this processing in the form of a coverage plot of the {term}`SRS` data, of an expected {glue}`srs_expected_no` reports, {glue}`srs_missing_no` were missing,  {glue}`srs_error_no` could not be parsed, leaving {glue}`srs_good_no` reports for further processing.
Each {term}`SRS` report can contain many {term}`ARs` and an {term}`AR` may appear in many consecutive daily reports ( up ~13 days due to solar rotation).
The {glue}`srs_good_no` reports contain {glue}`srs_ars_tot_no` classifications across {glue}`srs_ars_tot_unique_no` unique active {term}`ARs`.
{numref}`fig:srs:map` is a heat map of all the individual {term}`ARs` locations, it demonstrates an interesting affect a lack of {term}`ARs` on the solar East limb (left of figure) compared to the West limb (right of figure) with some {term}`ARs` even appearing on the back side of the sun.
This is due to two factors 1) the difficulty in observing AR close to the limb and 2) the fact the AR locations are obtained at various times during the day and these location are subsequently differentially rotated to match the reporting time of 00:00 UT.
The daily {term}`SRS` reports can be combined to form a series for each {term}`AR` {numref}`fig:srs:trace` shows a number of such series and the path the {term}`ARs` trace across the disk, also revealing some issues.
Solar features such as {term}`ARs` should move across the disk at approximately the solar sidereal rotation rate of ~14 deg/day with respect to longitude with little variation in latitude ~ 0 deg/day.
The rate of change of longitude or latitude have been separately filtered to +- 7.5 deg/day of the expected rates resulting in {glue}`srs_ars_bad_no` bad positions across {glue}`srs_ars_bad_unique_no` {term}`ARs` to be filtered, leaving {glue}`srs_ars_good_no` classifications across {glue}`srs_ars_good_unique_no` {term}`ARs`.

```{glue:figure} srs_coverage_fig
:alt: "SRS Coverage"
:name: "fig:srs:coverage"
SRS coverage from {glue}`start_date` to {glue}`end_date`.
```

```{glue:figure} srs_map_fig
:alt: "SRS Map"
:name: "fig:srs:map"
Visualisation of all parsed SRS AR data from {glue}`start_date` to {glue}`end_date` as a heatmap, the quantisation of the positions caused by the latitidue and longitude being store in integers in the {term}`SRS` reports.
```

```{glue:figure} srs_trace_fig
:alt: "Traces of a number of SRS region pregresssion"
:name: "fig:srs:trace"
Traces of the progression of a number of AR across the solar disk. The traces for ARs 7946, 8090 and 8238 show issues with the reported data and will be removed from the dataset.
```

## Summary

## Bibliography

```{bibliography}
```
