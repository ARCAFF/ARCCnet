# %%
import io
import os
import sys
import tempfile
from datetime import datetime

import matplotlib.pyplot as plt
import neptune
import numpy as np
import seaborn as sns
import tensorflow as tf
import utilities_cutout as ut
import wandb
from comet_ml import Experiment
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.layers import Activation, BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical
from neptune.integrations.tensorflow_keras import NeptuneCallback
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from wandb.keras import WandbCallback

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Only use one GPU
sys.path.insert(0, "../")
# %%
gpus = tf.config.experimental.list_physical_devices("GPU")

# dynamically allocate necessary GPU memory
if gpus:
    try:
        # Memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPU")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
else:
    print("No GPU found, training will run on CPU.")

# %%
loaded_data = np.load("/ARCAFF/data/sgkf_split.npz")
X_train = loaded_data["X_train"]
Y_train = loaded_data["Y_train"]
X_val = loaded_data["X_val"]
Y_val = loaded_data["Y_val"]

# %%
### Remove bars from images
# Train dataset
valid_matrices_labels = [
    (matrix, Y_train[i])
    for i, matrix in enumerate(X_train)
    if ut.count_and_check_consecutive_zero_lines(matrix)[1] <= 5
]
valid_matrices, valid_labels = zip(*valid_matrices_labels)
X_train = np.array(valid_matrices)
Y_train = np.array(valid_labels)

# Val dataset
valid_matrices_labels = [
    (matrix, Y_val[i]) for i, matrix in enumerate(X_val) if ut.count_and_check_consecutive_zero_lines(matrix)[1] <= 5
]
valid_matrices, valid_labels = zip(*valid_matrices_labels)
X_val = np.array(valid_matrices)
Y_val = np.array(valid_labels)

# %%
config_dict = {
    "number_of_conv_layers": 5,
    "conv_layers_units": [128, 128, 128, 128, 128, 128, 128],
    "kernel_sizes": [(7, 7), (5, 5), (3, 3), (3, 3), (3, 3)],
    "padding_types": "same",
    "epochs": 2000,
    "regularization_lambda": 0.02,
    "initializer": "glorot_uniform",
    "learning_rate": 1e-4,
    "optimizer": "adam",
    "loss_function": "categorical_crossentropy",
    "batch_size": 512,
    "dropout_rate": 0.25,
    "dataset_name": "sgkf_split",
    "num_classes": len(np.unique(Y_train)),
    "MLP_layers": [1024, 512, 256, 128],
    "MLP_activation": "relu",
    "early_stopping_patience": 200,
}

# %%
# Logging initialization
# W&B
wandb.login()
run = wandb.init(project="Active Region Cutout Classification", entity="arcaff", config=config_dict)
config = wandb.config

# Neptune
run_neptune = neptune.init_run(project="ARCAFF/Active-Region-Cutout-Classification")
neptune_callback = NeptuneCallback(run=run_neptune)

# Comet
run_comet = Experiment(project_name="active-region-cutout-classification", workspace="arcaff")

# %%
# vertical and horizontal flip of the images
data_augment = True
if data_augment:
    print("Performing data augmentation ... \n")
    X_train, Y_train = ut.augment_data(X_train, Y_train)
    print("Flipping done.\n")


# %%
print(f"Size of X_train: {sys.getsizeof(X_train)/ (1024.0 ** 2)/1000:.2f} GB")
# %%
number_channels = 1
input_shape = (X_train.shape[1], X_train.shape[2], number_channels)

model = Sequential()

# Loop through the number of convolutional layers
for i in range(config.number_of_conv_layers):
    if i == 0:
        # Add Conv2D layer
        model.add(
            Conv2D(
                config.conv_layers_units[i],
                config.kernel_sizes[i],
                strides=(2, 2),
                padding=config.padding_types,
                kernel_regularizer=regularizers.l2(config.regularization_lambda),
                kernel_initializer=config.initializer,
                input_shape=input_shape,
            )
        )
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(MaxPooling2D((2, 2), strides=(1, 1)))
    else:
        # Add Conv2D layer
        model.add(
            Conv2D(
                config.conv_layers_units[i],
                config.kernel_sizes[i],
                strides=(2, 2),
                padding=config.padding_types,
                kernel_regularizer=regularizers.l2(config.regularization_lambda),
                kernel_initializer=config.initializer,
            )
        )
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(MaxPooling2D((2, 2), strides=(1, 1)))

# MLP
model.add(Flatten())
for units in config.MLP_layers:
    model.add(Dense(units, activation=config.MLP_activation))
    model.add(Dropout(wandb.config.dropout_rate))

# output
num_classes = config.num_classes
model.add(Dense(num_classes, activation="softmax"))

optimizer = Adam(learning_rate=config.learning_rate)

loss = config.loss_function

model.compile(loss=loss, optimizer=optimizer, metrics="accuracy", run_eagerly=True)

print(model.summary())

# %%
# make histogram of train dataset
with plt.style.context("seaborn-v0_8-darkgrid"):
    labels = ["0.0", "Alpha", "Beta", "Beta-Gamma", "Beta-Gamma-Delta", "Beta-Delta"]
    greek_lables = ["0", r"$\alpha$", r"$\beta$", r"$\beta-\gamma$", r"$\beta-\gamma-\delta$", r"$\beta-\delta$"]
    values = [np.count_nonzero(Y_train == j) for j in labels]
    total = sum(values)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(greek_lables, values)

    # Add value labels and percentage above bars
    for bar in bars:
        yval = bar.get_height()
        percentage = f"{yval/total*100:.2f}%"
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 200,
            f"{yval} ({percentage})",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    plt.xticks(rotation=0, ha="center", fontsize=11)  # x-axis ticks labels
    plt.yticks(fontsize=12)  # y-axis ticks labels
    plt.ylabel("Number of Samples", fontsize=14)  # y-axis label
    # plt.xlabel('Categories', fontsize=14)                 # x-axis label
    plt.title("Train Dataset", fontsize=16)  # title

    # Save the plot to a buffer (in-memory file)
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()  # Close the plot to free memory
    buf.seek(0)  # Ensure the buffer's pointer is at the start
    # Create a temporary file and write the buffer's content
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        tmp_file.write(buf.read())
        tmp_file_path = tmp_file.name
    # Upload the temporary file
    run_neptune["heatmaps/train_dataset_histogram"].upload(tmp_file_path)
    wandb.log({"train_dataset_histogram": wandb.Image(tmp_file_path)})
    run_comet.log_image(tmp_file_path, name="train_dataset_histogram")

# %%
# Convert labels to numbers
label_encoder = LabelEncoder()
Y_train_encoded = label_encoder.fit_transform(Y_train)
Y_val_encoded = label_encoder.fit_transform(Y_val)

# On-Hot encoding
Y_train_onehot = to_categorical(Y_train_encoded, config.num_classes)
Y_val_onehot = to_categorical(Y_val_encoded, config.num_classes)

# %%
# Compute class weights
class_weights = class_weight.compute_class_weight("balanced", classes=np.unique(Y_train_encoded), y=Y_train_encoded)
class_weights_dict = dict(enumerate(class_weights))

# %%
early_stopper = EarlyStopping(patience=config.early_stopping_patience, restore_best_weights=True)

history = model.fit(
    X_train,
    Y_train_onehot,
    validation_data=(X_val, Y_val_onehot),
    epochs=config.epochs,
    callbacks=[early_stopper, WandbCallback(save_model=False), neptune_callback, run_comet.get_callback("keras")],
    class_weight=class_weights_dict,
    batch_size=config.batch_size,
)

# %%
# save the model
model_filename = f'/ARCAFF/models/CNN_{datetime.now().strftime("%Y%m%d-%H%M")}---{run.id}.keras'
model.save(model_filename)
artifact = wandb.Artifact(name="model", type="model", description="ARCAFF CNN")
artifact.add_file(model_filename)
wandb.log_artifact(artifact)

# %%
### Test Dataset ###
lbls_test = list(np.unique(Y_val))
y_test_pred = model.predict(X_val)
Y_test_pred_encoded = [np.argmax(y_test_pred[j]) for j in range(len(y_test_pred))]

# %%
# Classification report
report_dict = classification_report(Y_val_encoded, Y_test_pred_encoded, target_names=lbls_test, output_dict=True)
# Log the classification report to WandB
wandb.log({"classification_report_test": report_dict})

# %%
# Confusion Matrix
wandb.log(
    {
        "confusion_matrix_test": wandb.plot.confusion_matrix(
            y_true=Y_val_encoded, preds=Y_test_pred_encoded, class_names=lbls_test
        )
    }
)

run_comet.log_confusion_matrix(y_true=Y_val_encoded, y_predicted=Y_test_pred_encoded, labels=lbls_test)

# %%
greek_lbls = ["QS", r"$\alpha$", r"$\beta$", r"$\beta-\delta$", r"$\beta-\gamma$", r"$\beta-\gamma-\delta$"]
conf_matrix = confusion_matrix(Y_val_encoded, Y_test_pred_encoded)

# Calculate percentages
conf_matrix_percentage = conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
conf_matrix_percentage_with_sign = np.array([[f"{value:.2f}%" for value in row] for row in conf_matrix_percentage])

# Plotting
plt.figure(figsize=(8, 6))
sns.heatmap(
    conf_matrix_percentage,
    annot=conf_matrix_percentage_with_sign,
    fmt="",
    cmap="Blues",
    cbar=False,
    xticklabels=greek_lbls,
    yticklabels=greek_lbls,
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# %%
# Binary Classification
lbls = ["Quiet", "AR"]

Y_test_binary = np.zeros(len(Y_val_encoded))
for j in range(len(Y_val_encoded)):
    if Y_val_encoded[j] >= 1:
        Y_test_binary[j] = 1

Y_test_pred_binary = np.zeros(len(Y_test_pred_encoded))
for j in range(len(Y_test_pred_binary)):
    if Y_test_pred_encoded[j] >= 1:
        Y_test_pred_binary[j] = 1

conf_matrix = confusion_matrix(Y_test_binary, Y_test_pred_binary)
# Calculate percentages
conf_matrix_percentage = conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
conf_matrix_percentage_with_sign = np.array([[f"{value:.2f}%" for value in row] for row in conf_matrix_percentage])
# %%
plt.figure(figsize=(8, 6))
sns.heatmap(
    conf_matrix_percentage,
    annot=conf_matrix_percentage_with_sign,
    fmt="",
    cmap="Blues",
    cbar=False,
    xticklabels=lbls,
    yticklabels=lbls,
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
# Save the plot to a buffer (in-memory file)
buf = io.BytesIO()
plt.savefig(buf, format="png")
plt.close()  # Close the plot to free memory
# Ensure the buffer's pointer is at the start
buf.seek(0)
# Create a temporary file and write the buffer's content
with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
    tmp_file.write(buf.read())
    # Ensure you note the file name to upload
    tmp_file_path = tmp_file.name

# Log binary confusion matrix
run_neptune["heatmaps/binary_confusion_matrix"].upload(tmp_file_path)
wandb.log({"binary_confusion_matrix": wandb.Image(tmp_file_path)})
run_comet.log_image(tmp_file_path, name="binary_confusion_matrix")

# %%
[[TN, FP], [FN, TP]] = conf_matrix.copy()
# Compute the scores
scores = ut.compute_scores(TN, FP, FN, TP)

score_names = ["TSS", "HSS", "CSI", "Recall", "Precision", "Specificity", "Accuracy", "F1 Score", "Balanced Accuracy"]
formatted_scores = zip(score_names, scores)

table = "| Metric            | Value     |\n"
table += "|-------------------|-----------|\n"
for name, score in formatted_scores:
    table += f"| {name:<17} | {score:<9.4f} |\n"

print(table)

# log binary metrics table
table_data = [["Metric", "Value"]] + list(formatted_scores)
wandb.log({"Binary_Metrics": wandb.Table(data=table_data, columns=["Metric", "Value"])})
for name, score in formatted_scores:
    Experiment.log_metric(f"Binary_Metrics/{name}", score)
    run[f"Binary_Metrics/{name}"].log(score)
