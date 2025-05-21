import os
import time
import socket
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model
from IPython.display import display
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

import arccnet.models.cutouts.mcintosh.dataset_utils as mci_ut_d
import arccnet.models.cutouts.mcintosh.train_utils as mci_ut_t
from arccnet.models import train_utils as ut_t
from arccnet.models.cutouts.mcintosh import config
from arccnet.models.cutouts.mcintosh.models import HierarchicalResNet
from arccnet.visualisation import utils as ut_v

pd.set_option("display.max_columns", None)


def main(args):
    # Overwrite config parameters with user-defined values (if provided)
    config.data_folder = args.data_folder
    config.dataset_folder = args.dataset_folder
    config.df_name = args.df_name
    config.plot_histograms = args.plot_histograms

    # Setup device (GPU if available)
    device = f"cuda:{config.gpu_index}" if torch.cuda.is_available() else "cpu"

    # Setup experiment logging if using Comet
    experiment = None
    if config.use_comet:
        experiment = Experiment(project_name=config.project_name, workspace=config.workspace)
        experiment.log_parameters(
            {
                "learning_rate": config.learning_rate,
                "batch_size": config.batch_size,
                "epochs": config.epochs,
                "model": config.resnet_version,
            }
        )
        experiment.log_code(config.__file__)
        experiment.log_code(mci_ut_t.__file__)
        experiment.log_code(mci_ut_d.__file__)
        augmentation_tags = [type(transform).__name__ for transform in config.train_transforms.transforms]
        experiment.add_tags(augmentation_tags)

    # Create weights directory based on current time and system info
    t = time.localtime()
    current_time = time.strftime("%Y%m%d-%H%M%S", t)
    run_id = f"{current_time}_mcintosh_GPU{torch.cuda.get_device_name()}_{socket.gethostname()}"
    ut_t_file_path = os.path.abspath(ut_t.__file__)
    weights_dir = os.path.join(os.path.dirname(ut_t_file_path), "weights", run_id)
    os.makedirs(weights_dir, exist_ok=True)

    # Process the AR dataset
    AR_df, encoders, mappings = mci_ut_d.process_ar_dataset(
        data_folder=config.data_folder,
        dataset_folder=config.dataset_folder,
        df_name=config.df_name,
        plot_histograms=config.plot_histograms,
    )

    # Filter the dataset based on longitude limits
    lonV = np.deg2rad(np.where(AR_df["path_image_cutout_hmi"] != "", AR_df["longitude_hmi"], AR_df["longitude_mdi"]))
    condition = (lonV < -np.deg2rad(config.long_limit_deg)) | (lonV > np.deg2rad(config.long_limit_deg))
    df_filtered = AR_df[~condition]
    df_rear = AR_df[condition]
    AR_df.loc[df_filtered.index, "location"] = "front"
    AR_df.loc[df_rear.index, "location"] = "rear"
    AR_filtered = AR_df[AR_df["location"] != "rear"]

    # Split the dataset into train, validation, and test sets
    train_df, val_df, test_df = mci_ut_d.split_dataset(
        df=AR_filtered,
        group_column="number",
        plot_histograms=config.plot_histograms,
        train_size=config.train_size,
        val_size=config.val_size,
        test_size=config.test_size,
        random_state=42,
        verbose=True,
    )

    if experiment:
        experiment.log_dataset_hash(train_df)
        experiment.log_dataset_hash(val_df)
        experiment.log_dataset_hash(test_df)

    # Create datasets and corresponding loaders
    train_dataset = mci_ut_d.SunspotDataset(
        config.data_folder, config.dataset_folder, train_df, transform=config.train_transforms
    )
    val_dataset = mci_ut_d.SunspotDataset(config.data_folder, config.dataset_folder, val_df)
    test_dataset = mci_ut_d.SunspotDataset(config.data_folder, config.dataset_folder, test_df)

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True
    )

    # Function to print class distribution summaries
    def get_class_distribution_summary(train_df, val_df, test_df, component):
        train_counts = train_df[component].value_counts()
        val_counts = val_df[component].value_counts()
        test_counts = test_df[component].value_counts()

        train_percentages = train_df[component].value_counts(normalize=True) * 100
        val_percentages = val_df[component].value_counts(normalize=True) * 100
        test_percentages = test_df[component].value_counts(normalize=True) * 100

        summary_df = pd.DataFrame(
            {
                "Train": train_counts.astype(str) + " (" + train_percentages.map("{:.2f}%".format) + ")",
                "Val": val_counts.astype(str) + " (" + val_percentages.map("{:.2f}%".format) + ")",
                "Test": test_counts.astype(str) + " (" + test_percentages.map("{:.2f}%".format) + ")",
            }
        ).fillna("0 (0.00%)")
        return summary_df

    components = ["Z_component_grouped", "p_component_grouped", "c_component_grouped"]
    for component in components:
        summary_df = get_class_distribution_summary(train_df, val_df, test_df, component)
        display(summary_df)

    # Plot a histogram of combined classes
    ut_v.make_classes_histogram(
        AR_filtered["Z_component_grouped"] + AR_filtered["p_component_grouped"] + AR_filtered["c_component_grouped"],
        figsz=(21, 8),
        y_off=25,
        text_fontsize=8,
    )

    # Build mapping for valid classes
    valid_combined_classes = sorted(
        list(
            set(
                [
                    (
                        AR_filtered["Z_component_grouped"].iloc[i],
                        AR_filtered["p_component_grouped"].iloc[i],
                        AR_filtered["c_component_grouped"].iloc[i],
                    )
                    for i in range(len(AR_filtered))
                ]
            )
        )
    )
    valid_p_for_z = defaultdict(set)
    valid_c_for_zp = defaultdict(set)
    for z, p, c in valid_combined_classes:
        z_idx = encoders["Z_encoder"].transform([z])[0]
        p_idx = encoders["p_encoder"].transform([p])[0]
        c_idx = encoders["c_encoder"].transform([c])[0]
        valid_p_for_z[z_idx].add(p_idx)
        valid_c_for_zp[(z_idx, p_idx)].add(c_idx)
    valid_p_for_z = {k: sorted(v) for k, v in valid_p_for_z.items()}
    valid_c_for_zp = {k: sorted(v) for k, v in valid_c_for_zp.items()}

    # Determine number of classes for each component
    num_classes_Z = len(AR_filtered["Z_component_grouped"].unique())
    num_classes_P = len(AR_filtered["p_component_grouped"].unique())
    num_classes_C = len(AR_filtered["c_component_grouped"].unique())

    # Initialize the model and replace activations
    model = HierarchicalResNet(
        num_classes_Z=num_classes_Z,
        num_classes_P=num_classes_P,
        num_classes_C=num_classes_C,
        resnet_version=config.resnet_version,
    ).to(device)
    ut_t.replace_activations(model, nn.ReLU, nn.LeakyReLU, negative_slope=0.01)
    num_params = ut_t.count_trainable_parameters(model, print_num=True)
    if experiment:
        experiment.set_model_graph(str(model))
        experiment.log_metric("trainable_parameters", num_params)

    # Define loss functions with class weights
    z_weights = mci_ut_d.compute_weights(train_df["Z_grouped_encoded"], num_classes_Z)
    p_weights = mci_ut_d.compute_weights(train_df["p_grouped_encoded"], num_classes_P)
    c_weights = mci_ut_d.compute_weights(train_df["c_grouped_encoded"], num_classes_C)
    criterion_Z = nn.CrossEntropyLoss(weight=z_weights.to(device))
    criterion_P = nn.CrossEntropyLoss(weight=p_weights.to(device))
    criterion_C = nn.CrossEntropyLoss(weight=c_weights.to(device))

    # Setup optimizer and gradient scaler
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    cuda_version = torch.version.cuda
    if cuda_version and float(cuda_version) < 11.8:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = torch.amp.GradScaler("cuda")

    # Initialize training tracking variables
    teacher_forcing_ratio = config.initial_teacher_forcing_ratio if config.teacher_forcing else None
    best_val_metric = 0.0
    patience_counter = 0

    # Training and validation loop
    for epoch in range(config.epochs):
        # Training step
        train_dict = mci_ut_t.train(
            model=model,
            device=device,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion_z=criterion_Z,
            criterion_p=criterion_P,
            criterion_c=criterion_C,
            teacher_forcing_ratio=teacher_forcing_ratio,
            scaler=scaler,
        )
        # Validation step
        val_dict = mci_ut_t.evaluate(
            model=model,
            device=device,
            loader=val_loader,
            criterion_z=criterion_Z,
            criterion_p=criterion_P,
            criterion_c=criterion_C,
        )

        if experiment:
            metrics = {
                "avg_train_loss": train_dict["train_loss"],
                "train_accuracy": train_dict["avg_train_accuracy"],
                "train_accuracy_z": train_dict["train_accuracy_z"],
                "train_accuracy_p": train_dict["train_accuracy_p"],
                "train_accuracy_c": train_dict["train_accuracy_c"],
                "avg_val_loss": val_dict["val_loss"],
                "val_accuracy": val_dict["avg_val_accuracy"],
                "val_accuracy_z": val_dict["val_accuracy_z"],
                "val_accuracy_p": val_dict["val_accuracy_p"],
                "val_accuracy_c": val_dict["val_accuracy_c"],
                "teacher_forcing_ratio": teacher_forcing_ratio or 0,
            }
            experiment.log_metrics(metrics, epoch=epoch)

        # Print epoch information
        if teacher_forcing_ratio is not None:
            print(f"Epoch {epoch + 1}/{config.epochs}: Teacher Forcing Ratio = {teacher_forcing_ratio:.3f}")
        else:
            print(f"Epoch {epoch + 1}/{config.epochs}")
        print(
            f"Train Loss: {train_dict['train_loss']:.4f}, Train Acc: {train_dict['avg_train_accuracy']:.4f}, "
            f"Train Acc Z/p/c: {train_dict['train_accuracy_z']:.4f}/{train_dict['train_accuracy_p']:.4f}/{train_dict['train_accuracy_c']:.4f}"
        )
        print(
            f"Val   Loss: {val_dict['val_loss']:.4f}, Val   Acc: {val_dict['avg_val_accuracy']:.4f}, "
            f"Val   Acc Z/p/c: {val_dict['val_accuracy_z']:.4f}/{val_dict['val_accuracy_p']:.4f}/{val_dict['val_accuracy_c']:.4f}"
        )

        best_val_metric, patience_counter, stop_training = mci_ut_t.check_early_stopping(
            val_dict["avg_val_accuracy"], best_val_metric, patience_counter, model, weights_dir, config.patience
        )
        if stop_training:
            break

        # Update teacher forcing ratio if applicable
        if config.teacher_forcing and teacher_forcing_ratio is not None:
            teacher_forcing_ratio = max(
                config.min_teacher_forcing_ratio,
                teacher_forcing_ratio * config.teacher_forcing_decay,
            )

    # Load the best model for testing
    model = ut_t.load_model_test(weights_dir, model, device)
    (
        accuracy_z,
        accuracy_p,
        accuracy_c,
        f1_score_z,
        f1_score_p,
        f1_score_c,
        true_labels_z,
        pred_labels_z,
        true_labels_p,
        pred_labels_p,
        true_labels_c,
        pred_labels_c,
    ) = mci_ut_t.test(
        model=model,
        device=device,
        loader=test_loader,
        valid_p_for_z=valid_p_for_z,
        valid_c_for_zp=valid_c_for_zp,
    )

    if experiment:
        log_model(experiment, model=model, model_name=config.resnet_version)
        experiment.log_metrics(
            {
                "test_accuracy_z": accuracy_z,
                "test_accuracy_p": accuracy_p,
                "test_accuracy_c": accuracy_c,
                "test_f1_score_z": f1_score_z,
                "test_f1_score_p": f1_score_p,
                "test_f1_score_c": f1_score_c,
            }
        )

    # Print test results
    mci_ut_t.print_test_scores(accuracy_z, accuracy_p, accuracy_c, f1_score_z, f1_score_p, f1_score_c)

    # Compute and log confusion matrices
    labels_z = [str(label) for label in encoders["Z_encoder"].classes_]
    labels_p = [str(label) for label in encoders["p_encoder"].classes_]
    labels_c = [str(label) for label in encoders["c_encoder"].classes_]

    figsize = (6, 6)
    z_cm_path = os.path.join(os.path.dirname(ut_t_file_path), "temp", "confusion_matrix_z.png")
    p_cm_path = os.path.join(os.path.dirname(ut_t_file_path), "temp", "confusion_matrix_p.png")
    c_cm_path = os.path.join(os.path.dirname(ut_t_file_path), "temp", "confusion_matrix_c.png")

    cm_z = confusion_matrix(true_labels_z, pred_labels_z)
    cm_p = confusion_matrix(true_labels_p, pred_labels_p)
    cm_c = confusion_matrix(true_labels_c, pred_labels_c)

    ut_v.plot_confusion_matrix(cm_z, labels_z, "Z Component", figsize=figsize, save_path=z_cm_path)
    ut_v.plot_confusion_matrix(cm_p, labels_p, "p Component", figsize=figsize, save_path=p_cm_path)
    ut_v.plot_confusion_matrix(cm_c, labels_c, "c Component", figsize=figsize, save_path=c_cm_path)
    if experiment:
        experiment.log_image(z_cm_path, name="Z Component")
        experiment.log_image(p_cm_path, name="p Component")
        experiment.log_image(c_cm_path, name="c Component")
        experiment.log_confusion_matrix(
            matrix=np.array(cm_z),
            title="Confusion Matrix at best val epoch - Z Component",
            file_name="test_confusion_matrix_Z.json",
            labels=labels_z,
        )
        experiment.log_confusion_matrix(
            matrix=np.array(cm_p),
            title="Confusion Matrix at best val epoch - p Component",
            file_name="test_confusion_matrix_p.json",
            labels=labels_p,
        )
        experiment.log_confusion_matrix(
            matrix=np.array(cm_c),
            title="Confusion Matrix at best val epoch - c Component",
            file_name="test_confusion_matrix_c.json",
            labels=labels_c,
        )

    # Compute grouped class confusion matrix and scores
    true_grouped = [(true_labels_z[i], true_labels_p[i], true_labels_c[i]) for i in range(len(true_labels_z))]
    pred_grouped = [(pred_labels_z[i], pred_labels_p[i], pred_labels_c[i]) for i in range(len(pred_labels_z))]
    true_grouped_labels = [
        labels_z[true_grouped[i][0]] + labels_p[true_grouped[i][1]] + labels_c[true_grouped[i][2]]
        for i in range(len(true_grouped))
    ]
    pred_grouped_labels = [
        labels_z[pred_grouped[i][0]] + labels_p[pred_grouped[i][1]] + labels_c[pred_grouped[i][2]]
        for i in range(len(pred_grouped))
    ]
    encoder = LabelEncoder()
    encoder.fit(true_grouped_labels)
    class_mapping = {label: idx for idx, label in enumerate(encoder.classes_)}
    encoded_true = encoder.transform(true_grouped_labels)
    encoded_pred = [class_mapping[pred] for pred in pred_grouped_labels]

    grouped_cm_path = os.path.join(os.path.dirname(ut_t_file_path), "temp", "confusion_matrix_grouped.png")
    cm_gr = confusion_matrix(encoded_true, encoded_pred, labels=range(len(encoder.classes_)))
    ut_v.plot_confusion_matrix(
        cmc=cm_gr,
        labels=list(encoder.classes_),
        title="Confusion Matrix for Grouped Classes",
        figsize=(12, 12),
        save_path=grouped_cm_path,
    )

    grouped_accuracy = np.mean(np.array(encoded_true) == np.array(encoded_pred))
    grouped_f1_score = f1_score(encoded_true, encoded_pred, average="macro")

    print(f"Grouped Accuracy: {grouped_accuracy:.4f}")
    print(f"Grouped F1 Score (Macro): {grouped_f1_score:.4f}")
    if experiment:
        experiment.log_metrics(
            {
                "grouped_accuracy": grouped_accuracy,
                "grouped_f1_score_macro": grouped_f1_score,
            }
        )
        experiment.log_image(grouped_cm_path, name="Grouped Confusion Matrix")
        experiment.log_confusion_matrix(
            matrix=np.array(cm_gr),
            title="Confusion Matrix at best val epoch - Grouped Classes",
            file_name="grouped_confusion_matrix.json",
            labels=list(encoder.classes_),
        )

    # Example: Predict a single sample
    idx = 3567
    row = test_df.iloc[idx]
    region_input, label = test_dataset[idx]
    region_input = region_input.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        outputs_z, outputs_p, outputs_c = model(region_input.to(device))
    _, pred_z = torch.max(outputs_z, 1)
    _, pred_p = torch.max(outputs_p, 1)
    _, pred_c = torch.max(outputs_c, 1)
    final_class_z = encoders["Z_encoder"].inverse_transform(pred_z.cpu().numpy())
    final_class_p = encoders["p_encoder"].inverse_transform(pred_p.cpu().numpy())
    final_class_c = encoders["c_encoder"].inverse_transform(pred_c.cpu().numpy())
    final_class = final_class_z[0] + final_class_p[0] + final_class_c[0]
    print(f"Original class: {row['mcintosh_class']}")
    print(
        f"Original Grouped class: {row['Z_component_grouped']}{row['p_component_grouped']}{row['c_component_grouped']}"
    )
    print(f"Predicted Final Class: {final_class}")
    mci_ut_d.display_sample_image(config.data_folder, config.dataset_folder, test_df, idx)

    if experiment:
        experiment.end()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate the HierarchicalResNet model with configurable options."
    )

    # --- General Parameters ---
    parser.add_argument(
        "--resnet_version", type=str, default=config.resnet_version, help="ResNet version (default: %(default)s)"
    )
    parser.add_argument(
        "--gpu_index", type=int, default=config.gpu_index, help="GPU index to use (default: %(default)s)"
    )
    parser.add_argument("--epochs", type=int, default=config.epochs, help="Number of epochs (default: %(default)s)")
    parser.add_argument(
        "--patience", type=int, default=config.patience, help="Patience for early stopping (default: %(default)s)"
    )
    parser.add_argument("--batch_size", type=int, default=config.batch_size, help="Batch size (default: %(default)s)")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=config.num_workers,
        help="Number of DataLoader workers (default: %(default)s)",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=config.learning_rate, help="Learning rate (default: %(default)s)"
    )
    parser.add_argument(
        "--random_state", type=int, default=config.random_state, help="Random state (default: %(default)s)"
    )

    # --- Teacher Forcing Parameters ---
    parser.add_argument(
        "--initial_teacher_forcing_ratio",
        type=float,
        default=config.initial_teacher_forcing_ratio,
        help="Initial teacher forcing ratio (default: %(default)s)",
    )
    parser.add_argument(
        "--min_teacher_forcing_ratio",
        type=float,
        default=config.min_teacher_forcing_ratio,
        help="Minimum teacher forcing ratio (default: %(default)s)",
    )
    parser.add_argument(
        "--teacher_forcing_decay",
        type=float,
        default=config.teacher_forcing_decay,
        help="Teacher forcing decay (default: %(default)s)",
    )
    parser.add_argument(
        "--teacher_forcing",
        type=bool,
        default=config.teacher_forcing,
        help="Use teacher forcing (default: %(default)s)",
    )

    # --- Dataset Parameters ---
    parser.add_argument(
        "--data_folder", type=str, default=config.data_folder, help="Path to the data folder (default: %(default)s)"
    )
    parser.add_argument(
        "--dataset_folder",
        type=str,
        default=config.dataset_folder,
        help="Name of the dataset folder (default: %(default)s)",
    )
    parser.add_argument(
        "--df_name", type=str, default=config.df_name, help="Name of the parquet file to load (default: %(default)s)"
    )
    parser.add_argument(
        "--long_limit_deg",
        type=int,
        default=config.long_limit_deg,
        help="Longitude limit in degrees (default: %(default)s)",
    )
    parser.add_argument(
        "--train_size",
        type=float,
        default=config.train_size,
        help="Fraction of data for training (default: %(default)s)",
    )
    parser.add_argument(
        "--val_size", type=float, default=config.val_size, help="Fraction of data for validation (default: %(default)s)"
    )
    parser.add_argument(
        "--test_size", type=float, default=config.test_size, help="Fraction of data for testing (default: %(default)s)"
    )
    parser.add_argument(
        "--plot_histograms",
        action="store_true",
        default=config.plot_histograms,
        help="Plot histograms of class distributions (default: %(default)s)",
    )

    # --- Comet Logging Parameters ---
    parser.add_argument(
        "--use_comet", type=bool, default=config.use_comet, help="Use Comet ML logging (default: %(default)s)"
    )
    parser.add_argument(
        "--project_name", type=str, default=config.project_name, help="Comet project name (default: %(default)s)"
    )
    parser.add_argument(
        "--workspace", type=str, default=config.workspace, help="Comet workspace (default: %(default)s)"
    )

    args = parser.parse_args()

    # Overwrite the configuration in config.py with any provided command-line values:
    config.resnet_version = args.resnet_version
    config.gpu_index = args.gpu_index
    config.epochs = args.epochs
    config.patience = args.patience
    config.batch_size = args.batch_size
    config.num_workers = args.num_workers
    config.learning_rate = args.learning_rate
    config.random_state = args.random_state

    config.initial_teacher_forcing_ratio = args.initial_teacher_forcing_ratio
    config.min_teacher_forcing_ratio = args.min_teacher_forcing_ratio
    config.teacher_forcing_decay = args.teacher_forcing_decay
    config.teacher_forcing = args.teacher_forcing

    config.data_folder = args.data_folder
    config.dataset_folder = args.dataset_folder
    config.df_name = args.df_name
    config.long_limit_deg = args.long_limit_deg
    config.train_size = args.train_size
    config.val_size = args.val_size
    config.test_size = args.test_size
    config.plot_histograms = args.plot_histograms

    config.use_comet = args.use_comet
    config.project_name = args.project_name
    config.workspace = args.workspace

    main()
