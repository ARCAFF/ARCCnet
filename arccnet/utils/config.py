import configparser
from pathlib import Path

from platformdirs import PlatformDirs

import arccnet

dirs = PlatformDirs("ARCCnet", "ARCCAF")

#: User configuration directory
CONFIG_DIR = Path(dirs.user_config_dir)

__all__ = ["load_config", "print_config", "CONFIG_DIR"]


def load_config():
    r"""
    Read the "arccnet" configuration file.

    If one does not exist in the user's home directory, then read in the defaults from "arccnet/utils/arccnet".
    """
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())

    # Get locations of SunPy configuration files to be loaded
    config_files = _find_config_files()

    # Read in configuration files
    config.read(config_files)

    config.set("paths", "data_root", str(Path(dirs.user_data_dir) / "arccnet"))

    return config


def _find_config_files():
    """
    Finds locations of ARCCnet configuration files.
    """
    config_files = []
    config_filename = "arccnetrc"

    # find default configuration file
    module_dir = Path(arccnet.__file__).parent
    config_files.append(str(module_dir / "utils" / "arccnetrc"))

    # if a site configuration file exists, add that to list of files to read
    # so that any values set there will override ones specified in the default
    # config file
    config_path = Path(dirs.site_config_dir)
    if config_path.joinpath(config_filename).exists():
        config_files.append(str(config_path.joinpath(config_filename)))

    # if a user configuration file exists, add that to list of files to read
    # so that any values set there will override ones specified in the default
    # config file
    config_path = Path(dirs.user_config_dir)
    if config_path.joinpath(config_filename).exists():
        config_files.append(str(config_path.joinpath(config_filename)))

    return config_files


def print_config():
    """
    Print current configuration options.
    """
    print("FILES USED:")
    for file_ in _find_config_files():
        print("  " + file_)

    print("\nCONFIGURATION:")
    for section in arccnet.config.sections():
        print(f"  [{section}]")
        for option in arccnet.config.options(section):
            print(f"  {option} = {arccnet.config.get(section, option)}")
        print("")
