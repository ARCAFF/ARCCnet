from datetime import datetime

import pytest

from arccnet.cli.main import main, parser


def test_parser():
    with pytest.raises(SystemExit):
        options = parser([])

    with pytest.raises(SystemExit):
        options = parser(["datasets"])

    with pytest.raises(SystemExit):
        options = parser(["datasets", "generate"])

    options = parser(["datasets", "generate", "all"])
    assert options["dataset"] == "all"

    options = parser(["datasets", "generate", "all", "--start-date", "2021-06-01T12:45:58"])
    assert options["dataset"] == "all"
    assert options["general.start_date"] == datetime(2021, 6, 1, 12, 45, 58)


def test_main():
    options = main(["datasets", "generate", "all", "--start-date", "2025-01-01"])
    assert options["general"]["start_date"] == datetime(2025, 1, 1)
