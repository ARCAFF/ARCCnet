---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# ARCAFF Cutout Dataset EDA

```{code-cell} ipython3
from astropy.io import fits
from astropy.time import Time
from IPython.display import display, Image
from scipy.stats import kurtosis, skew, mode

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from arccnet.models.cutout_classification import utilities_cutout as ut

%matplotlib inline
```

```{code-cell} ipython3
# specify forlder location
folder_path = '/ARCAFF/'
numpy_dir = folder_path + 'cutout_dataset/numpy/'
saved_arrays_path = 'saved_arrays/'
info_ars = pd.read_parquet(folder_path+'/cutout-magnetic-catalog-v20231027.parq', engine='pyarrow')
```

```{code-cell} ipython3
# Convert Julian dates to datetime objects
info_ars['time'] = info_ars['time.jd1'] + info_ars['time.jd2']
times = Time(info_ars['time'], format='jd')
dates = pd.to_datetime(times.iso)  # Convert to datetime objects
info_ars['dates'] = dates
```

```{code-cell} ipython3
# remove problematic magnetograms from the dataset

tuples_list = [
    ('quicklook/20001130_000028_MDI.png', 9242),
    ('quicklook/19960903_000030_MDI.png', 7986),
    ('quicklook/19960904_000030_MDI.png', 7986),
    ('quicklook/19960908_000030_MDI.png', 7988),
]

problematic_cases = []
for quicklook_path, number in tuples_list:
    print(folder_path + quicklook_path)
    row = (info_ars['quicklook_path_mdi'] == quicklook_path) 
    filtered_df = info_ars[row]
    problematic_cases.append(filtered_df)
problematic_df = pd.concat(problematic_cases)

print(len(problematic_df))

info_ars = info_ars.drop(problematic_df.index).reset_index()
images_with_bars = np.load(saved_arrays_path + 'cleaned_df_images_with_bars.npy')
```


```{code-cell} ipython3
# find images with have horizontal bars

find_images_with_bars = False

if find_images_with_bars:

    images_with_bars = []
    images_without_bars = []

    for idx, row in info_ars.iterrows():

        path = 'path_image_cutout_hmi' if row['path_image_cutout_mdi'] == '0.0' else 'path_image_cutout_mdi'
    
        fits_file_path = folder_path + row[path]
        path_parts = os.path.split(fits_file_path)
        npy_file_name = path_parts[1].replace('.fits', '.npy')
        data = np.load(numpy_dir + npy_file_name)
        has_bars, bar_count = ut.count_and_check_bars(data)
        
        if bar_count > 5:
            images_with_bars.append(idx)
        else:
            images_without_bars.append(idx)
```

```{code-cell} ipython3
# convert and save fits to numpy arrays for faster I/O

convert_fits_to_np = False

if convert_fits_to_np:
    
    numpy_dir = folder_path + 'cutout_dataset/numpy/'

    for idx, row in info_ars.iterrows():

        path = 'path_image_cutout_hmi' if row['path_image_cutout_mdi'] == '0.0' else 'path_image_cutout_mdi'

        fits_file_path = folder_path + row[path]
        path_parts = os.path.split(fits_file_path)
        npy_file_name = path_parts[1].replace('.fits', '.npy')
        if not os.path.exists(numpy_dir + npy_file_name):
            with fits.open(fits_file_path) as img_fits:
                data = np.array(img_fits[1].data, dtype=float)
                np.save(numpy_dir + npy_file_name, data)
```

```{code-cell} ipython3
# create dataframes
AR_df = info_ars[info_ars['region_type'] == 'AR']

AR_df_MDI = AR_df[AR_df['quicklook_path_hmi'] == '0.0']
AR_df_HMI = AR_df[AR_df['quicklook_path_mdi'] == '0.0']
```

## Classes Counts

```{code-cell} ipython3
classes = ['0.0', 'Alpha', 'Beta', 'Beta-Gamma', 'Beta-Gamma-Delta', 'Beta-Delta', 'Gamma', 'Gamma-Delta']

def find_class_counts(df, classes):
    counts = df['magnetic_class'].value_counts()
    return counts.reindex(classes, fill_value=0)

classes_counts = find_class_counts(info_ars, classes)
classes_counts_MDI = find_class_counts(AR_df_MDI, classes)
classes_counts_HMI = find_class_counts(AR_df_HMI, classes)
```

```{code-cell} ipython3
print(classes_counts)
print(f"\nTotal n° of cutouts: {np.sum(classes_counts)}")
```

```{code-cell} ipython3
labels = classes_counts.index.tolist() # Get class labels from the index of the value_counts Series
values = classes_counts.values         # Get counts for each class
total = np.sum(values)                

greek_labels = ['QS', r'$\alpha$', r'$\beta$', r'$\beta-\gamma$', r'$\beta-\gamma-\delta$', r'$\beta-\delta$',  r'$\gamma-\delta$',  r'$\gamma$']

with plt.style.context('seaborn-v0_8-darkgrid'):
    plt.figure(figsize=(12, 6))
    bars = plt.bar(greek_labels, values, edgecolor='black')

    for bar in bars:
        yval = bar.get_height()
        percentage = f"{yval/total*100:.2f}%"
        plt.text(bar.get_x() + bar.get_width()/2, yval + 200, f"{yval} ({percentage})", ha='center', va='bottom', fontsize=11)

    plt.xticks(rotation=0, ha='center', fontsize=11)
    plt.yticks(fontsize=12)
    plt.ylabel('Number of Cutouts', fontsize=14)
    plt.title("Cutout Dataset", fontsize=16)

    plt.show()
```

```{code-cell} ipython3
bin_labels = [r'$\alpha$', r'$\beta$', r'$\beta-X$']
bin_values = [values[1], values[2], values[3]+values[4]+values[5]]
values_MDI = classes_counts_MDI.values  # Get counts for each class from MDI
values_HMI = classes_counts_HMI.values  # Get counts for each class from HMI
bin_values_MDI = [values_MDI[1], values_MDI[2], values_MDI[3] + values_MDI[4] + values_MDI[5]]
bin_values_HMI = [values_HMI[1], values_HMI[2], values_HMI[3] + values_HMI[4] + values_HMI[5]]

total_MDI = sum(bin_values_MDI)
total_HMI = sum(bin_values_HMI)
total = total_MDI + total_HMI  # Total for percentages

with plt.style.context('seaborn-v0_8-darkgrid'):
    plt.figure(figsize=(8, 6))
    bars_MDI = plt.bar(bin_labels, bin_values_MDI, label='MDI', edgecolor='black', color='tab:blue')
    bars_HMI = plt.bar(bin_labels, bin_values_HMI, bottom=bin_values_MDI, label='HMI', edgecolor='black', color='tab:orange')

    for mdi, hmi in zip(bars_MDI, bars_HMI):
        yval_mdi = mdi.get_height()
        yval_hmi = hmi.get_height()
        percentage_mdi = f"{yval_mdi/total*100:.2f}%"
        percentage_hmi = f"{yval_hmi/total*100:.2f}%"
        plt.text(mdi.get_x() + mdi.get_width()/2, yval_mdi / 2, f"{yval_mdi} ({percentage_mdi})", ha='center', va='center', fontsize=11, color='white')
        plt.text(hmi.get_x() + hmi.get_width()/2, yval_mdi + yval_hmi / 2, f"{yval_hmi} ({percentage_hmi})", ha='center', va='center', fontsize=11, color='white')

    plt.xticks(rotation=0, ha='center', fontsize=11)
    plt.yticks(fontsize=12)
    plt.ylabel('Number of Cutouts', fontsize=14)
    plt.title("Cutout Dataset: MDI vs HMI", fontsize=16)
    plt.legend()


    plt.show()
```

```{code-cell} ipython3
bin_labels = ['QS', 'AR']
bin_values = [values[0], np.sum(values[1:])]

with plt.style.context('seaborn-v0_8-darkgrid'):
    plt.figure(figsize=(8, 6))
    bars = plt.bar(bin_labels, bin_values, edgecolor='black')

    for bar in bars:
        yval = bar.get_height()
        percentage = f"{yval/total*100:.2f}%"
        plt.text(bar.get_x() + bar.get_width()/2, yval + 200, f"{yval} ({percentage})", ha='center', va='bottom', fontsize=11)

    plt.xticks(rotation=0, ha='center', fontsize=11)
    plt.yticks(fontsize=12)
    plt.ylabel('Number of Cutouts', fontsize=14)
    plt.title("Cutout Dataset", fontsize=16)

    plt.show()
```

## Image Bars 

```{code-cell} ipython3
info_ars['has_bars'] = 0
info_ars.loc[images_with_bars, 'has_bars'] = 1
info_ars['year'] = info_ars['dates'].dt.year
info_ars['month'] = info_ars['dates'].dt.month
monthly_data = info_ars.groupby(['year', 'month'])['has_bars'].sum().reset_index()

heatmap_data = monthly_data.pivot(index='year', columns='month', values='has_bars').fillna(0) 
heatmap_data.iloc[0,:3] = np.nan

plt.figure(figsize=(8, 12))
sns.heatmap(heatmap_data, annot=True, cmap="Blues", fmt=".0f")
plt.title('Number of Bars per Month Across Years')
plt.ylabel('Month')
plt.xlabel('Year')
plt.xticks()
plt.show()
```

```{code-cell} ipython3
print(f"Cutouts with bars: {len(images_with_bars)}")
print(f"Cutouts without bars: {len(info_ars)-len(images_with_bars)}")
print(f"Percentage of cutouts with bars: {100*len(images_with_bars)/( len(info_ars)):.2f}%")
```

```{code-cell} ipython3
# visualize images with bars
row = info_ars.iloc[images_with_bars].iloc[15]
with fits.open(folder_path + row['path_image_cutout_mdi']) as img_fits:
    data = np.array(img_fits[1].data, dtype=float)
    plt.imshow(data, origin = 'lower')
    plt.colorbar()
    plt.title(row['magnetic_class'])
    plt.show()
```

## Location of ARs on the Sun

```{code-cell} ipython3
latV = np.deg2rad(np.where(AR_df['processed_path_image_hmi'] == '0.0', AR_df['latitude_mdi'], AR_df['latitude_hmi']))
lonV = np.deg2rad(np.where(AR_df['processed_path_image_hmi'] == '0.0', AR_df['longitude_mdi'], AR_df['longitude_hmi']))

yV = np.cos(latV)*np.sin(lonV)
zV = np.sin(latV)
```

```{code-cell} ipython3
condition = yV**2 + zV**2 > 0.95

rear_latV = latV[condition]
rear_lonV = lonV[condition]
rear_yV = yV[condition]
rear_zV = zV[condition]

front_latV = latV[~condition] 
front_lonV = lonV[~condition]
front_yV = yV[~condition]
front_zV = zV[~condition]
```

```{code-cell} ipython3
print(f"Rear ARs: {len(rear_latV)}")
print(f"Front ARs: {len(front_latV)}")
print(f"Percentage of rear ARs: {100*len(rear_latV)/( len(rear_latV) + len(front_latV)):.2f}%")
```

```{code-cell} ipython3
# ARs' location on the solar disc
circle = plt.Circle((0, 0), 1, edgecolor='gray', facecolor='none')
fig, ax = plt.subplots(figsize=(10, 10))
ax.add_artist(circle)

num_meridians = 12
num_parallels = 12
num_points = 300

# Angles for the meridians and parallels
phis = np.linspace(0, 2 * np.pi, num_meridians, endpoint=False) 
lats = np.linspace(-np.pi/2, np.pi/2, num_parallels)

# Angles from south to north pole
theta = np.linspace(-np.pi/2, np.pi/2, num_points)

# Plot each meridian
for phi in phis:
    y = np.cos(theta) * np.sin(phi)
    z = np.sin(theta)
    ax.plot(y, z, 'k-', linewidth = 0.2) 

# Plot each parallel
for lat in lats:
    radius = np.cos(lat)  # This defines the radius of the latitude circle in the y-z plane
    y = radius * np.sin(theta)
    z = np.sin(lat) * np.ones(num_points)
    ax.plot(y, z, 'k-', linewidth = 0.2)

ax.scatter(rear_yV, rear_zV, s=1, alpha = 0.2, color = 'r', label='Rear')
ax.scatter(front_yV, front_zV, s=1, alpha = 0.2, color = 'b', label='Front')

ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.set_aspect('equal')
ax.axis('off')
ax.legend(fontsize = 12)

plt.show()
```

## n° of ARs vs time

```{code-cell} ipython3
time_counts_MDI = AR_df_MDI['dates'].value_counts().sort_index()
time_counts_HMI = AR_df_HMI['dates'].value_counts().sort_index()

plt.figure(figsize=(12, 6))
plt.scatter(time_counts_MDI.index, time_counts_MDI.values, color = 'b', alpha = 0.3, label='MDI', s=2)
plt.scatter(time_counts_HMI.index, time_counts_HMI.values, color ='r', alpha = 0.3, label='HMI', s=2)
plt.xlabel('date', fontsize = 11)
plt.ylabel('n° of ARs', fontsize = 11)
plt.legend(fontsize = 11 )
plt.show()
```

## Statistical analysis

```{code-cell} ipython3
def compute_statistics(df):
    """
    Returns a dictionary containing arrays of computed statistics:
        'min': Minimum values.
        'max': Maximum values.
        'mean': Average values.
        'std': Standard deviations, measuring spread around the mean.
        'var': Variances, indicating the degree of spread in the dataset.
        'median': Median values.
        'iqr': Interquartile ranges, difference between 75th and 25th percentiles.
        'skewness': Skewness values, indicating the asymmetry of the data distribution.
        'kurtosis': Kurtosis values, indicating the tailedness of the data distribution.
        'range': Range values, difference between maximum and minimum values.
    """
    
    # Initialize arrays to store the results
    results = {
        'min': np.zeros(len(df)),
        'max': np.zeros(len(df)),
        'mean': np.zeros(len(df)),
        'std': np.zeros(len(df)),
        'var': np.zeros(len(df)),
        'median': np.zeros(len(df)),
        'iqr': np.zeros(len(df)),
        'skewness': np.zeros(len(df)),
        'kurtosis': np.zeros(len(df)),
        'range': np.zeros(len(df))
    }
    
    # Iterate over the DataFrame rows
    for idx, row in df.iterrows():
        path = 'path_image_cutout_hmi' if row['path_image_cutout_mdi'] == '0.0' else 'path_image_cutout_mdi'
        
        fits_file_path = folder_path + row[path]
        path_parts = os.path.split(fits_file_path)
        npy_file_name = path_parts[1].replace('.fits', '.npy')
        
        data = np.load(numpy_dir + npy_file_name).flatten()
        
        for key, func in [
            ('min', np.min), ('max', np.max), ('mean', np.mean), 
            ('std', np.std), ('var', np.var), ('median', np.median), 
            ('skewness', skew), ('kurtosis', kurtosis), ('range', np.ptp)
        ]:
            results[key][idx] = func(data)
        
        q75, q25 = np.percentile(data, [75, 25])
        results['iqr'][idx] = q75 - q25
    
    return results
```

```{code-cell} ipython3
title_mapping = {
    'min': 'Min Values',
    'max': 'Max Values',
    'mean': 'Mean Values',
    'std': 'Standard Deviations',
    'var': 'Variances',
    'median': 'Median Values',
    'iqr': 'Interquartile Range (IQR) Values',
    'skewness': 'Skewness Values',
    'kurtosis': 'Kurtosis Values',
    'range': 'Range Values'
}
```

```{code-cell} ipython3
def find_outliers(data):
    """
    Identifies outliers in a dataset using the interquartile range (IQR) method.

    Parameters:
    - data (array): Input data, which can be a list, array, or Pandas Series.

    Returns:
    - array: Indices of outliers in the input data array.

    If the input data contains non-numeric values, they are converted to NaN (Not a Number).
    The function then calculates the lower and upper bounds for outliers based on the IQR method.
    Data points falling below the lower bound or above the upper bound are considered outliers.
    The function returns the indices of these outlier values in the original input data array.
    """
    
    data_numeric = np.array(data, dtype=float)  # Converts non-convertible values to NaN
    nan_indices = np.where(np.isnan(data_numeric))[0] # Find indices where values are NaN
    valid_data = data_numeric[~np.isnan(data_numeric)] # Filter out NaNs to get valid numeric data

    if valid_data.size == 0: # Check if valid_data has any elements left
        return nan_indices

    # Calculating Q1 and Q3 from the valid numeric data
    Q1 = np.percentile(valid_data, 25)
    Q3 = np.percentile(valid_data, 75)
    
    IQR = Q3 - Q1 

    # Defining bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identifying outlier indices in the valid numeric data
    outlier_bool = (valid_data < lower_bound) | (valid_data > upper_bound)
    outlier_indices = np.where(outlier_bool)[0]

    # Convert indices in valid_data back to original indices (excluding NaNs)
    valid_indices = np.where(~np.isnan(data_numeric))[0]
    original_indices = valid_indices[outlier_indices]

    # Combine the indices of NaN and numeric outliers
    all_outlier_indices = np.concatenate((original_indices, nan_indices))

    return all_outlier_indices
```

### Compute Statistics

```{code-cell} ipython3
df_bars = info_ars.iloc[images_with_bars]
df_nobars = info_ars.drop(images_with_bars)

df_alpha = df_nobars[df_nobars['magnetic_class']=='Alpha'].reset_index(drop=False)
df_beta = df_nobars[df_nobars['magnetic_class']=='Beta'].reset_index(drop=False)
df_betax = pd.concat([
    df_nobars[df_nobars['magnetic_class'] == 'Beta-Gamma'],
    df_nobars[df_nobars['magnetic_class'] == 'Beta-Delta'],
    df_nobars[df_nobars['magnetic_class'] == 'Beta-Gamma-Delta']
]).reset_index(drop=False)
```

```{code-cell} ipython3
compute_stats_bool = False 
if compute_stats_bool:
    results_alpha = compute_statistics(df_alpha)
    results_beta = compute_statistics(df_beta)
    results_betax = compute_statistics(df_betax)
    np.savez(saved_arrays_path + 'results_alpha.npz', **results_alpha)
    np.savez(saved_arrays_path + 'results_beta.npz', **results_beta)
    np.savez(saved_arrays_path + 'results_betax.npz', **results_betax)
else:
    with np.load(saved_arrays_path + 'results_alpha.npz') as loaded_data:
        results_alpha = {key: loaded_data[key] for key in loaded_data}
    with np.load(saved_arrays_path + 'results_beta.npz') as loaded_data:
        results_beta = {key: loaded_data[key] for key in loaded_data}
    with np.load(saved_arrays_path + 'results_betax.npz') as loaded_data:
        results_betax = {key: loaded_data[key] for key in loaded_data}
   
```

```{code-cell} ipython3
df_alpha_results = pd.DataFrame(results_alpha)
df_beta_results = pd.DataFrame(results_beta)
df_betax_results = pd.DataFrame(results_betax)
df_alpha_results['Group'] = 'Alpha'
df_beta_results['Group'] = 'Beta'
df_betax_results['Group'] = 'Beta-X'
df_results_combined = pd.concat([df_alpha_results, df_beta_results, df_betax_results], ignore_index=True)
combined_describe = df_results_combined.groupby('Group').describe()
```

### Visualizations

```{code-cell} ipython3
columns_to_display = 8
num_chunks = (combined_describe.shape[1] + columns_to_display - 1) // columns_to_display
format_func = lambda x: '{:.2f}'.format(x)
for i in range(num_chunks):
    start_col = i * columns_to_display
    end_col = start_col + columns_to_display
    display(combined_describe.iloc[:, start_col:end_col].style.format(format_func))
```

```{code-cell} ipython3
def plot_histograms(key):
    # Calculate weights for each dataset so that the sum of the weights is 1
    weights_alpha = np.ones_like(results_alpha[key]) / len(results_alpha[key])
    weights_beta = np.ones_like(results_beta[key]) / len(results_beta[key])
    weights_betax = np.ones_like(results_betax[key]) / len(results_betax[key])
    # Plot histograms
    plt.hist(results_alpha[key], weights=weights_alpha, alpha=0.35, label='Alpha', density=True)
    plt.hist(results_beta[key], weights=weights_beta, alpha=0.35, label='Beta', density=True)
    plt.hist(results_betax[key], weights=weights_betax, alpha=0.35, label='Beta-X', density=True)
    plt.legend()
    plt.title('Normalized Histograms of ' + title_mapping[key])
    plt.xlabel(key.title() + ' Value')
    plt.ylabel('Relative Frequency')
    plt.show()

def plot_boxplots(data, labels, title):
    plt.figure(figsize=(10, 6))
    plt.boxplot(data, labels=labels)
    plt.title(title)
    plt.ylabel('Value')
    plt.show()
```

### Histograms

```{code-cell} ipython3
for key in title_mapping:
    plot_histograms(key)
```

### Violin Plots

```{code-cell} ipython3
# Combine the data into a single DataFrame
def results_to_df(results, label):
    df = pd.DataFrame(results)
    df['Group'] = label
    return df

combined_df = pd.concat([
    results_to_df(results_alpha, 'Alpha'), 
    results_to_df(results_beta, 'Beta'), 
    results_to_df(results_betax, 'Beta-x')])

for stat in title_mapping:
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Group', y=stat, data=combined_df, color = 'orange')
    plt.title(title_mapping[stat])
    plt.xlabel('Magnetic Class')
    plt.ylabel(f'{stat.capitalize()}')
    plt.show()
```

### Box Plots

```{code-cell} ipython3
for key in title_mapping:
    plot_boxplots(
        [
            results_alpha[key][~np.isnan(results_alpha[key])],
            results_beta[key][~np.isnan(results_beta[key])],
            results_betax[key][~np.isnan(results_betax[key])]
        ],
        ['Alpha', 'Beta', 'Beta-x'],
        title_mapping[key]  # Use the formal title from the mapping
    )
```

### Outliers

```{code-cell} ipython3
table = []
keys = set(results_alpha.keys()) | set(results_beta.keys()) | set(results_betax.keys())  # Union of all keys

for key in sorted(keys): 
    alpha_count = len(find_outliers(results_alpha.get(key, [])))
    beta_count = len(find_outliers(results_beta.get(key, [])))
    betax_count = len(find_outliers(results_betax.get(key, [])))
    table.append([key, alpha_count, beta_count, betax_count])

outliers_df = pd.DataFrame(table, columns=['Statistic', 'Alpha', 'Beta', 'Beta-x'])

plt.figure(figsize=(4, 6)) 
sns.heatmap(outliers_df.set_index('Statistic'), annot=True, cmap='viridis', fmt="d") 
plt.title('Outliers')
plt.show()
```

```{code-cell} ipython3
# print the alpha image with median iqr

df_alpha['iqr'] = results_alpha['iqr']
df_alpha['iqr_difference'] = (df_alpha['iqr'] - df_alpha['iqr'].median()).abs()
median_iqr_path = df_alpha[df_alpha['iqr_difference'] == df_alpha['iqr_difference'].min()]['path_image_cutout_mdi']

with fits.open(folder_path + median_iqr_path.iloc[0]) as img_fits:
    data = np.array(img_fits[1].data, dtype=float)
    plt.imshow(data)
    plt.colorbar()
    plt.show()

# print 5 iqr alpha outliers

for path in df_alpha.iloc[find_outliers(results_alpha['iqr'])]['path_image_cutout_mdi'][:5]:
    print(folder_path + path)
    with fits.open(folder_path + path) as img_fits:
        data = np.array(img_fits[1].data, dtype=float)
        plt.imshow(data)
        plt.colorbar()
        plt.show()
```

```{code-cell} ipython3
# print the beta image with median iqr

df_beta['iqr'] = results_beta['iqr']
df_beta['iqr_difference'] = (df_beta['iqr'] - df_beta['iqr'].median()).abs()
median_iqr_path = df_beta[df_beta['iqr_difference'] == df_beta['iqr_difference'].min()]['path_image_cutout_mdi']

with fits.open(folder_path + median_iqr_path.iloc[0]) as img_fits:
    data = np.array(img_fits[1].data, dtype=float)
    plt.imshow(data)
    plt.colorbar()
    plt.show()
```

```{code-cell} ipython3
# print 5 iqr beta outliers

for path in df_beta.iloc[find_outliers(df_beta['iqr'])]['path_image_cutout_mdi'][:5]:
    print(folder_path + path)
    with fits.open(folder_path + path) as img_fits:
        data = np.array(img_fits[1].data, dtype=float)
        plt.imshow(data)
        plt.colorbar()
        plt.show()
```

```{code-cell} ipython3
# print the beta image with median iqr

df_betax['iqr'] = results_betax['iqr']
df_betax['iqr_difference'] = (df_betax['iqr'] - df_betax['iqr'].median()).abs()
median_iqr_path = df_betax[df_betax['iqr_difference'] == df_betax['iqr_difference'].min()]['path_image_cutout_mdi']

with fits.open(folder_path + median_iqr_path.iloc[0]) as img_fits:
    data = np.array(img_fits[1].data, dtype=float)
    plt.imshow(data)
    plt.colorbar()
    plt.title(df_betax.iloc[median_iqr_path.index.tolist()[0]]['magnetic_class'])
    plt.show()
```

```{code-cell} ipython3
# print 5 iqr beta outliers

for idx, row in df_betax.iloc[find_outliers(df_betax['iqr'])][:5].iterrows():
    path = row['path_image_cutout_mdi']
    print(folder_path + path)
    with fits.open(folder_path + path) as img_fits:
        data = np.array(img_fits[1].data, dtype=float)
        plt.imshow(data)
        plt.colorbar()
        plt.title(row['magnetic_class'])
        plt.show()
```
