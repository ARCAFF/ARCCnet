import os
import sys
import argparse
import configparser
from io import StringIO
from pathlib import Path
from datetime import datetime
from collections import ChainMap, defaultdict
from collections.abc import Mapping

from arccnet import load_config
from arccnet.models.cutouts import config as config_module
from arccnet.models.cutouts.inference import predict
from arccnet.models.cutouts.train import run_training
from arccnet.pipeline.main import process_ar_catalogs, process_ars, process_flares
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
    commands = root_parser.add_subparsers(title="Commands", help="Commands", required=True, dest="command")

    catalog_parser = commands.add_parser("catalog", help="Catalog generation and download")
    catlog_commands = catalog_parser.add_subparsers(
        required=True,
        dest="catalog",
    )

    # Create
    dataset_generation = catlog_commands.add_parser(
        "generate", help="Create generate datasets by downloading and processing raw data and metadata"
    )
    dataset_generation.add_argument(
        "dataset", choices=["ar_catalog", "ars", "flares"], help="Type of dataset to create."
    )
    dataset_generation.add_argument(
        "--start-date", type=datetime.fromisoformat, help="Start date for data (ISO format)", dest="general.start_date"
    )
    dataset_generation.add_argument(
        "--end-date", type=datetime.fromisoformat, help="Start date for data (ISO format)", dest="general.end_date"
    )
    dataset_generation.add_argument(
        "--jsoc_email", type=str, help="JSOC registered email address to use in exports", dest="jsoc.jsoc_email"
    )
    dataset_generation.add_argument("--data-root", type=Path, help="Root directory for dataset", dest="paths.data_root")

    # Download
    dataset_downlaod = catlog_commands.add_parser("download", help="Download preprocessed datasets")
    dataset_downlaod.add_argument(
        "dataset", choices=["cutout_classification", "region_detection", "all"], help="Type of dataset to create"
    )

    # Train
    train_parser = commands.add_parser("train", help="Train models on datasets")
    train_parser.add_argument("--model_name", type=str, help="Timm model name")
    train_parser.add_argument("--batch_size", type=int, help="Batch size for training.")
    train_parser.add_argument("--num_workers", type=int, help="Number of workers for data loading and preprocessing.")
    train_parser.add_argument("--num_epochs", type=int, help="Number of epochs for training.")
    train_parser.add_argument("--patience", type=int, help="Patience for early stopping.")
    train_parser.add_argument("--learning_rate", type=float, help="Learning rate for optimizer.")
    train_parser.add_argument("--gpu_index", type=int, help="Index of the GPU to use.")
    train_parser.add_argument("--data_folder", type=str, help="Path to the data folder.")
    train_parser.add_argument("--dataset_folder", type=str, help="Path to the dataset folder.")
    train_parser.add_argument("--df_file_name", type=str, help="Name of the dataframe file.")

    # Eval
    inference_parser = commands.add_parser("eval", help="Run inference on FITS data")
    inference_parser.add_argument(
        "--fits_file_path",
        type=str,
        default=os.path.join(
            config_module.data_folder, config_module.dataset_folder, "fits", "20160203_235809_I-12493_HMI_SIDE1.fits"
        ),
        help="Path to the FITS file.",
    )
    inference_parser.add_argument("--project_name", type=str, default="arcaff-v2-qs-ia-a-b-bg", help="Project name.")
    inference_parser.add_argument("--workspace", type=str, default="arcaff", help="Comet.ml workspace name.")
    inference_parser.add_argument("--model_name", type=str, default="resnet18", help="Model name.")
    inference_parser.add_argument("--model_version", type=str, default="1.0.0", help="Model version.")

    options, rest = root_parser.parse_known_args(args)

    options_dict = vars(options)

    return options_dict, rest


def catalog_commands(options):
    if options["catalog"] == "generate":
        if options["dataset"] == "flares":
            process_flares(options)
        if options["dataset"] == "ar_catalog":
            process_ar_catalogs(options)
        if options["dataset"] == "ars":
            catalog = process_ar_catalogs(options)
            process_ars(options, catalog)


def train_commands(options):
    args = argparse.Namespace()

    arg_to_config = {
        "model_name": "model_name",
        "batch_size": "batch_size",
        "num_epochs": "num_epochs",
        "patience": "patience",
        "learning_rate": "learning_rate",
        "gpu_index": "gpu_index",
        "data_folder": "data_folder",
        "dataset_folder": "dataset_folder",
        "df_file_name": "df_file_name",
        "num_workers": "num_workers",
    }

    def get_option_value(options, arg):
        try:
            return options[arg]
        except KeyError:
            return None

    for arg in arg_to_config.keys():
        value = get_option_value(options, arg)
        setattr(args, arg, value)

    run_training(config_module, args)


def inference_commands(options):
    args = argparse.Namespace()

    args.model_name = options.get("model_name", "resnet18")
    args.workspace = options.get("workspace", "arcaff")
    args.model_version = options.get("model_version", "1.0.0")
    args.fits_file_path = options.get("fits_file_path")
    args.project_name = options.get("project_name", "arcaff-v2-qs-ia-a-b-bg")

    predict(args)


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

    # because use interpolation of the config have to set data_root so derived path update and not just use chainmap
    try:
        data_root = nested_cli_options["paths"]["data_root"]
        if cli_config:
            cli_config.set("paths", "data_root", str(data_root))
        if config:
            config.set("paths", "data_root", str(data_root))
    except KeyError:
        pass

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
    command = combined.get("command")
    if command == "catalog":
        catalog_commands(combined)
    elif command == "train":
        train_commands(combined)
    elif command == "eval":
        inference_commands(combined)
    else:
        raise ValueError(f"Unknown command: {command}")
