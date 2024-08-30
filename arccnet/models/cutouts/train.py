import os
import argparse

import torch
from comet_ml import Experiment

import arccnet.models.cutouts.config as config
import arccnet.models.utilities as ut

# Initialize argument parser
parser = argparse.ArgumentParser(description="Training script with configurable options.")
parser.add_argument("--model_name", type=str, help="Timm model name")
parser.add_argument("--batch_size", type=int, help="Batch size for training.")
parser.add_argument("--num_workers", type=int, help="Number of workers for data loading and preprocessing.")
parser.add_argument("--num_epochs", type=int, help="Number of epochs for training.")
parser.add_argument("--patience", type=int, help="Patience for early stopping.")
parser.add_argument("--learning_rate", type=float, help="Learning rate for optimizer.")
parser.add_argument("--gpu_index", type=int, help="Index of the GPU to use.")
parser.add_argument("--data_folder", type=str, help="Path to the data folder.")
parser.add_argument("--dataset_folder", type=str, help="Path to the dataset folder.")
parser.add_argument("--df_file_name", type=str, help="Name of the dataframe file.")

args = parser.parse_args()

# Override config settings with arguments if provided
if args.model_name is not None:
    config.model_name = args.model_name
if args.batch_size is not None:
    config.batch_size = args.batch_size
if args.num_epochs is not None:
    config.num_epochs = args.num_epochs
if args.patience is not None:
    config.patience = args.patience
if args.learning_rate is not None:
    config.learning_rate = args.learning_rate
if args.gpu_index is not None:
    config.gpu_index = args.gpu_index
    config.device = f"cuda:{args.gpu_index}"
if args.data_folder is not None:
    config.data_folder = args.data_folder
if args.dataset_folder is not None:
    config.dataset_folder = args.dataset_folder
if args.df_file_name is not None:
    config.df_file_name = args.df_file_name
if args.num_workers is not None:
    config.num_workers = args.num_workers

run_id, weights_dir = ut.generate_run_id(config)

run_comet = Experiment(project_name=config.project_name, workspace="arcaff")

run_comet.add_tags([config.model_name])
run_comet.log_parameters(
    {
        "model_name": config.model_name,
        "batch_size": config.batch_size,
        "GPU": f"GPU{config.gpu_index}_{torch.cuda.get_device_name()}",
        "num_epochs": config.num_epochs,
        "patience": config.patience,
    }
)

run_comet.log_code(config.__file__)
run_comet.log_code(ut.__file__)

print("Making dataframe...")
df, AR_df = ut.make_dataframe(config.data_folder, config.dataset_folder, config.df_file_name)

df, df_du = ut.undersample_group_filter(
    df, config.label_mapping, long_limit_deg=60, undersample=True, buffer_percentage=0.1
)
fold_df = ut.split_data(df_du, label_col="grouped_labels", group_col="number", random_state=42)
df = ut.assign_fold_sets(df, fold_df)
print("done.")
print("Starting Training...")

(avg_test_loss, test_accuracy, test_precision, test_recall, test_f1, cm_test, report_df) = ut.train_model(
    config, df, weights_dir, experiment=run_comet
)

print("Logging assets...")
script_dir = os.path.dirname(ut.__file__)
save_path = os.path.join(script_dir, "temp", "working_dataset.png")
ut.make_classes_histogram(
    df_du["grouped_labels"], title="Dataset (Grouped Undersampled)", y_off=100, figsz=(7, 5), save_path=save_path
)
run_comet.log_image(save_path)

run_comet.log_asset_data(df.to_csv(index=False), name="dataset.csv")
print("done.")
