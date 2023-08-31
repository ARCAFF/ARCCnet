import datetime
import tempfile
from pathlib import Path

import pandas as pd

from arccnet.data_generation.magnetograms.base_magnetogram import BaseMagnetogram


class MockMagnetogram(BaseMagnetogram):
    """
    overwrite abstract methods
    """

    def generate_drms_query(self, start_time: datetime.datetime, end_time: datetime.datetime, frequency="1d") -> str:
        return f"Mock query: {start_time} - {end_time} @ {frequency}"

    # Implement other required abstract methods here
    def _get_matching_info_from_record(self, records):
        pass

    def series_name(self):
        pass

    def date_format(self):
        pass

    def segment_column_name(self):
        pass

    @property
    def metadata_save_location(self):
        # Create and return a temporary file path for testing
        pass


def test_base_magnetogram_generate_drms_query():
    # Test the generate_drms_query method of BaseMagnetogram
    mock_magnetogram = MockMagnetogram()
    query = mock_magnetogram.generate_drms_query(datetime.datetime(2023, 1, 1), datetime.datetime(2023, 1, 5))
    assert query == "Mock query: 2023-01-01 00:00:00 - 2023-01-05 00:00:00 @ 1d"


def test_save_metadata_to_csv_with_temporary_file():
    mock_instance = MockMagnetogram()
    keys_data = {"key1": [1, 2, 3], "key2": [4, 5, 6]}
    keys_df = pd.DataFrame(keys_data)

    # Call the method being tested
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = Path(temp_dir) / "temp_metadata.csv"
        mock_instance._save_metadata_to_csv(keys_df, filepath=temp_file_path, index=False)
        # Verify the file is created
        assert temp_file_path.exists()

        # test that the df are equal
        pd.testing.assert_frame_equal(keys_df, pd.read_csv(temp_file_path))
