import datetime

from arccnet.data_generation.magnetograms.base_magnetogram import BaseMagnetogram


def test_base_magnetogram_generate_drms_query():
    # Test the generate_drms_query method of BaseMagnetogram
    class MockMagnetogram(BaseMagnetogram):
        def generate_drms_query(
            self, start_time: datetime.datetime, end_time: datetime.datetime, frequency="1d"
        ) -> str:
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

        def metadata_save_location(self):
            pass

    mock_magnetogram = MockMagnetogram()
    query = mock_magnetogram.generate_drms_query(datetime.datetime(2023, 1, 1), datetime.datetime(2023, 1, 5))
    assert query == "Mock query: 2023-01-01 00:00:00 - 2023-01-05 00:00:00 @ 1d"
