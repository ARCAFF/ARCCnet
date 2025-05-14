""" """

import os
import configparser
from pathlib import Path
from datetime import datetime

from platformdirs import PlatformDirs

import arccnet

dirs = PlatformDirs("ARCCnet", "ARCAFF")

#: User configuration directory
CONFIG_DIR = Path(dirs.user_config_dir)

#: User data directory
DATA_DIR = Path(dirs.user_data_dir)

__all__ = ["load_config", "print_config", "CONFIG_DIR"]


class EnvAndExtendedInterpolation(configparser.ExtendedInterpolation):
    """
    Custom interpolation class that first replaces environment variables,
    then handles config option substitutions.
    """

    def before_get(self, parser, section, option, value, defaults):
        # Replace environment variables in the value
        value = os.path.expandvars(value)
        # Now perform the usual ExtendedInterpolation
        return super().before_get(parser, section, option, value, defaults)


def _find_config_files():
    """
    Finds locations of ARCCnet configuration files.

    Checks for default, site, and user configurations.
    """
    config_files = []
    config_filename = "arccnetrc"

    # Find default configuration file
    module_dir = Path(arccnet.__file__).parent
    default_config = module_dir / "utils" / "arccnetrc"
    if default_config.exists():
        config_files.append(str(default_config))

    # if a site configuration file exists, add that to list of files to read
    # so that any values set there will override ones specified in the default
    # config file
    site_config = Path(dirs.site_config_dir) / config_filename
    if site_config.exists():
        config_files.append(str(site_config))

    # if a user configuration file exists, add that to list of files to read
    # so that any values set there will override ones specified in the default
    # config file
    user_config = CONFIG_DIR / config_filename
    if user_config.exists():
        config_files.append(str(user_config))

    # Find configuration file in the user's home directory
    home_config = Path.home() / config_filename
    if home_config.exists():
        config_files.append(str(home_config))

    return config_files


def load_config():
    r"""
    Read the "arccnet" configuration file.

    If one does not exist in the user's home directory, then read in the defaults from "arccnet/utils/arccnetrc".
    """
    converters = {"_date": datetime.fromisoformat}
    config = configparser.ConfigParser(interpolation=EnvAndExtendedInterpolation(), converters=converters)

    # Get locations of ARCCnet configuration files to be loaded
    config_files = _find_config_files()

    # Read in configuration files
    config.read(config_files)

    # Set data_root if not defined
    if config.get("paths", "data_root", fallback=None) is None:
        config.set("paths", "data_root", str(DATA_DIR / "arccnet"))
    # Set data_root if not defined
    if config.get("paths", "ARCAFF_DATA_FOLDER", fallback=None) is None:
        config.set("paths", "ARCAFF_DATA_FOLDER", str(DATA_DIR / "arccnet"))

    return config


def print_config():
    """
    Print current configuration options.
    """
    config = load_config()

    print("FILES USED:")
    for file_ in _find_config_files():
        print("  " + file_)

    print("\nCONFIGURATION:")
    for section in config.sections():
        print(f"  [{section}]")
        for option in config.options(section):
            print(f"  {option} = {config.get(section, option)}")
        print("")
