import os

import torch
from comet_ml import Experiment

import arccnet.models.cutouts.config as config
import arccnet.models.utilities as ut

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
