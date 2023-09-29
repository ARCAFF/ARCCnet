# isort: skip_file
"""
``ARCCnet``
===========

Active Region Classification Network

Part of the ARCAFF project

* Homepage: https://arcaff.eu
* Documentation:
"""
from arccnet.utils.config import load_config, print_config  # noqa

# from arccnet.util.logger import _init_log  # noqa
from .version import __version__  # noqa

config = load_config()
# log = _init_log(config=config)

__all__ = [
    "config",
    "print_config",
    "__version__",
]
