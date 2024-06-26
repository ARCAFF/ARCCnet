import requests

from astropy.table import Column, QTable

__all__ = ["retrieve_noaa_mapping", "remove_columns_with_suffix"]


def retrieve_noaa_mapping(url: str, identifier_col_name: str) -> QTable:
    """
    Retrieve the NOAA mapping from a specified URL and return it as a QTable.

    Parameters:
    url (str): The URL of the file to download.
    identifier_col_name (str): The name of the column for the HARP or TARP identifiers.

    Returns:
    QTable: A table containing the HARP/TARP identifiers and their corresponding NOAA values.
    """
    # Download the file
    response = requests.get(url)
    data = response.text

    # Split the data into lines and extract two columns
    lines = data.split("\n")[1:]  # Exclude the first line (header)
    split_record_identifier = []
    split_record_NOAANUM = []
    split_NOAA = []

    for line in lines:
        if line:
            identifier, noaa = line.split()
            commas = noaa.split(",")

            # Create a new row for each NOAA value
            for noaa_value in commas:
                split_record_identifier.append(int(identifier))
                split_NOAA.append(int(noaa_value))
                split_record_NOAANUM.append(len(commas))

    # Create a QTable with the split values
    table = QTable(
        [
            Column(split_record_identifier, name=identifier_col_name),
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
