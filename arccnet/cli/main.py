import sys
import argparse
import configparser
from io import StringIO
from pathlib import Path
from datetime import datetime
from collections import ChainMap, defaultdict
from collections.abc import Mapping

from arccnet import load_config
from arccnet.pipeline.main import process_flares
from arccnet.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["NestedChainMap", "parser", "combine_args", "main"]


class NestedChainMap(ChainMap):
    r"""
    A nested or recursive chainmap.

    The first occurrence takes precedence over others.

    Examples
    --------
    >>> d1 = {'a': {'b':1}}
    >>> d2 = {'a': {'b': 0, 'c': 0}}
    >>> ncm1 = NestedChainMap(d1, d2)
    >>> ncm1['a']['c'] == 0
    True
    >>> ncm1['a']['b'] == 1
    True
    >>> ncm2 = NestedChainMap(d2, d1)
    >>> ncm2['a']['c'] == 0
    True
    >>> ncm2['a']['b'] == 0
    True
    """

    def __getitem__(self, key):
        submaps = [mapping for mapping in self.maps if key in mapping]
        if not submaps:
            return self.__missing__(key)
        if isinstance(submaps[0][key], Mapping):
            return NestedChainMap(*(submap[key] for submap in submaps))
        return super().__getitem__(key)


def parser(args=None):
    root_parser = argparse.ArgumentParser(prog="arccnet", description="")
    root_parser.add_argument("--config-file", type=Path)
    root_parser.add_argument("--data-root", type=Path, dest="paths.data_root")
    commands = root_parser.add_subparsers(title="Commands", help="Commands", required=True)

    catalog_parser = commands.add_parser("catalog", help="Catalog generation and download")
    catlog_commands = catalog_parser.add_subparsers(
        required=True,
        dest="catalog",
    )

    # Create
    dataset_generation = catlog_commands.add_parser(
        "generate", help="Create generate datasets by downloading and processing raw data and metadata"
    )
    dataset_generation.add_argument("dataset", choices=["ars", "flares"], help="Type of dataset to create.")
    dataset_generation.add_argument(
        "--start-date", type=datetime.fromisoformat, help="Start date for data (ISO format)", dest="general.start_date"
    )
    dataset_generation.add_argument(
        "--end-date", type=datetime.fromisoformat, help="Start date for data (ISO format)", dest="general.end_date"
    )
    dataset_generation.add_argument(
        "--jsoc_email", type=str, help="JSOC registered email address to use in exports", dest="jsoc.jsoc_email"
    )

    # Download
    dataset_downlaod = catlog_commands.add_parser("download", help="Download preprocessed datasets")
    dataset_downlaod.add_argument(
        "dataset", choices=["cutout_classification", "region_detection", "all"], help="Type of dataset to create"
    )

    train_parser = commands.add_parser("train", help="Train models on datasets")
    train_parser.add_subparsers(dest="train")

    eval_parser = commands.add_parser("eval", help="Evaluate models on given data")
    eval_parser.add_subparsers(dest="eval")

    options, rest = root_parser.parse_known_args(args)

    options_dict = vars(options)
    if "train" in options_dict or "eval" in options_dict:
        raise NotImplementedError("Please wait 'train' and 'eval' commands have not been implemented")

    return options_dict, rest


def catalog_commands(options):
    if options["catalog"] == "generate":
        if options["dataset"] == "flares":
            process_flares(options)


def combine_args(args=None):
    r"""
    Combines command line arguments, user config and default values into single config.

    Order of precedence CLI -> CONFIG -> DEFAULT.

    Parameters
    ----------
    args

    Returns
    -------

    """
    cli_options, rest = parser(args or sys.argv[1:])

    # Drop None and convert to nested dict based on key e.g. `paths.data_root = a` -> `['paths']['data_root'] = a`
    # to match configparser format
    nested_cli_options = defaultdict(dict)
    for key, value in cli_options.items():
        if value is not None:
            if "." in key:
                section, name = key.split(".")
                nested_cli_options[section][name] = value
            else:
                nested_cli_options[key] = value

    # check for cli config file and load
    cli_config = nested_cli_options.pop("config_file", dict())
    if cli_config:
        config_file = Path(cli_config)
        if not config_file.exists():
            raise FileNotFoundError(f"The given configuration file does not exist {str(config_file)}")

        config_reader = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        cli_config = config_reader.read(config_file)

    config = load_config()
    combined = NestedChainMap(nested_cli_options, cli_config, config)

    config_str = StringIO()
    for section, conf in combined.items():
        if isinstance(conf, Mapping):
            print(section, file=config_str)
            for key, value in conf.items():
                print(f"\t{key}:\t{value}", file=config_str)
        else:
            print(f"\t{section}:\t{conf}", file=config_str)
    logger.info("\n" + config_str.getvalue())
    return combined


def main(args=None):
    combined = combine_args(args)
    if "catalog" in combined:
        catalog_commands(combined)
