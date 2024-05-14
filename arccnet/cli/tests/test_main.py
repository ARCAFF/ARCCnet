from datetime import datetime

import pytest

from arccnet.cli.main import combine_args, parser


def test_parser():
    with pytest.raises(SystemExit):
        options, rest = parser([])

    with pytest.raises(SystemExit):
        options, rest = parser(["catalog"])

    with pytest.raises(SystemExit):
        options, rest = parser(["catalog", "generate"])

    options, rest = parser(["catalog", "generate", "flares"])
    assert options["catalog"] == "generate"

    options, rest = parser(["catalog", "generate", "flares", "--start-date", "2021-06-01T12:45:58"])
    assert options["catalog"] == "generate"
    assert options["general.start_date"] == datetime(2021, 6, 1, 12, 45, 58)


def test_combine_args():
    options = combine_args(["catalog", "generate", "flares", "--start-date", "2025-01-01"])
    assert options["general"]["start_date"] == datetime(2025, 1, 1)
