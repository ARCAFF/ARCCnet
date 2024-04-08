import datetime

import pytest

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


@pytest.mark.remote_data
def test_base_magnetogram_generate_drms_query():
    # Test the generate_drms_query method of BaseMagnetogram
    mock_magnetogram = MockMagnetogram()
    query = mock_magnetogram.generate_drms_query(datetime.datetime(2023, 1, 1), datetime.datetime(2023, 1, 5))
    assert query == "Mock query: 2023-01-01 00:00:00 - 2023-01-05 00:00:00 @ 1d"
