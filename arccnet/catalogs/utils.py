import requests

from astropy.table import Column, QTable

__all__ = ["retrieve_harp_noaa_mapping", "remove_columns_with_suffix"]


def retrieve_harp_noaa_mapping():
    # URL of the file to download
    url = "http://jsoc.stanford.edu/doc/data/hmi/harpnum_to_noaa/all_harps_with_noaa_ars.txt"

    # Download the file
    response = requests.get(url)
    data = response.text

    # Split the data into lines and extract two columns
    lines = data.split("\n")[1:]  # Exclude the first line (header)
    split_record_HARPNUM = []
    split_record_NOAANUM = []
    split_NOAA = []

    for line in lines:
        if line:
            harp, noaa = line.split()
            commas = noaa.split(",")

            # Create a new row for each NOAA value
            for noaa_value in commas:
                split_record_HARPNUM.append(int(harp))
                split_NOAA.append(int(noaa_value))
                split_record_NOAANUM.append(len(commas))

    # Create a QTable with the split values
    table = QTable(
        [
            Column(split_record_HARPNUM, name="record_HARPNUM_arc"),
            Column(split_NOAA, name="NOAA"),
            Column(split_record_NOAANUM, name="NOAANUM"),
        ]
    )

    return table


def remove_columns_with_suffix(table, suffix):
    modified_table = table.copy()
    columns_to_remove = [col for col in modified_table.colnames if col.endswith(suffix)]
    modified_table.remove_columns(columns_to_remove)
    return modified_table
