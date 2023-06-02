# import datetime
#
# import pytest
#
# from arccnet.data_generation.magnetograms.utils import datetime_to_jsoc, jsoc_to_datetime
#
#
# def test_datetime_to_jsoc():
#     # Test a specific datetime object
#     datetime_obj = datetime.datetime(2023, 5, 26, 10, 30, 0)
#     expected_result = "2023.05.26_10:30:00"
#     assert datetime_to_jsoc(datetime_obj) == expected_result
#
#     # Test with a different datetime object
#     datetime_obj = datetime.datetime(2023, 1, 1, 0, 0, 0)
#     expected_result = "2023.01.01_00:00:00"
#     assert datetime_to_jsoc(datetime_obj) == expected_result
#
#
# def test_datetime_to_jsoc_invalid_input():
#     # Test with an invalid JSOC date string
#     invalid_datetime_obj = "20230526_103000"
#     with pytest.raises(ValueError):
#         jsoc_to_datetime(invalid_datetime_obj)
#
#     invalid_datetime_obj = 20230526
#     with pytest.raises(TypeError):
#         jsoc_to_datetime(invalid_datetime_obj)
#
#
# # arccnet.data_generation.magnetograms.utils.jsoc_to_datetime
#
#
# def test_jsoc_to_datetime():
#     # Test a specific JSOC date string
#     date_string = "2023.05.26_10:30:00"
#     expected_result = datetime.datetime(2023, 5, 26, 10, 30, 0)
#     assert jsoc_to_datetime(date_string) == expected_result
#
#     # Test with a different JSOC date string
#     date_string = "2023.01.01_00:00:00"
#     expected_result = datetime.datetime(2023, 1, 1, 0, 0, 0)
#     assert jsoc_to_datetime(date_string) == expected_result
#
#
# def test_jsoc_to_datetime_invalid_input():
#     # Test with an invalid JSOC date string
#     invalid_date_string = "20230526_103000"
#     with pytest.raises(ValueError):
#         jsoc_to_datetime(invalid_date_string)
