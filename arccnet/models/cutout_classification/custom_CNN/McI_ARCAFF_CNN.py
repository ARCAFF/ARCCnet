# Operating System
import io
import os
import sys
import os.path
import tempfile
from datetime import datetime

# Images
import matplotlib.pyplot as plt

# Data Handling
import numpy as np
import seaborn as sns

# TensorFlow
# import tensorrt
import tensorflow as tf

# wandb
import wandb
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.layers import Activation, BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam

# Keras
from keras.utils import to_categorical

# sklearn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from wandb.keras import WandbCallback

os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # only use one GPU
sys.path.insert(0, "../")
gpus = tf.config.experimental.list_physical_devices("GPU")

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
wandb.login()

loaded_data = np.load("/ARCAFF/data/grouped_sktf_McI_100.npz")
# Access individual arrays
X_train = loaded_data["X_train"]
Y_train = loaded_data["Y_train"]
X_val = loaded_data["X_val"]
Y_val = loaded_data["Y_val"]

only_AR = True  # removes quiet sun cutouts
if only_AR:
    # Get the indices where Y_train != '0.0'
    indices = [i for i, y in enumerate(Y_train) if y != "0.0"]
    # Filter both X_train and Y_train
    X_train = np.array([X_train[i] for i in indices])
    Y_train = np.array([Y_train[i] for i in indices])

    indices = [i for i, y in enumerate(Y_val) if y != "0.0"]
    X_val = np.array([X_val[i] for i in indices])
    Y_val = np.array([Y_val[i] for i in indices])

data_augment = True
if data_augment:
    labels = np.unique(Y_train)

    # Initialize an empty list to hold the augmented images and their labels
    augmented_images = []
    augmented_labels = []

    for label in labels:
        # Find indices of the current label
        indices = np.where(np.array(Y_train) == label)[0]
        # Extract the corresponding images
        label_images = X_train[indices]
        # Perform the flips
        label_horflip = label_images[:, :, ::-1]
        label_verflip = label_images[:, ::-1, :]
        label_horverflip = label_images[:, ::-1, ::-1]

        # Concatenate the flipped images
        label_augmented = np.concatenate([label_horflip, label_verflip, label_horverflip], axis=0)

        # Append the augmented images to the list
        augmented_images.append(label_augmented)

        # Create an array of labels for the augmented images (same label for each set of original and flipped images)
        label_augmented_labels = np.array([label] * label_augmented.shape[0])

        # Append the augmented labels to the list
        augmented_labels.append(label_augmented_labels)

    # Concatenate all augmented images and labels into single arrays
    all_augmented_images = np.concatenate(augmented_images, axis=0)
    all_augmented_labels = np.concatenate(augmented_labels, axis=0)

    X_train = np.concatenate([X_train, all_augmented_images], axis=0)
    Y_train = np.concatenate([Y_train, all_augmented_labels], axis=0)


# CNN parameters
run = wandb.init(
    project="McI Active Region Cutout Classification",
    entity="arcaff",
    tags=["McIntosh", "only_AR", "Augmented"],
    config={
        "number_of_conv_layers": 5,
        "conv_layers_units": [128, 64, 32, 32, 32],
        "kernel_sizes": [(7, 7), (5, 5), (3, 3), (3, 3), (3, 3)],
        "padding_types": "same",
        "epochs": 1000,
        "regularization_lambda": 0.03,
        "initializer": "glorot_uniform",
        "learning_rate": 1e-4,
        "optimizer": "adam",
        "loss_function": "categorical_crossentropy",
        "batch_size": 512,
        "dropout_rate": 0.2,
        "num_classes": len(np.unique(Y_train)),
        "MLP_layers": [512, 256, 128],
        "MLP_activation": "relu",
        "early_stopping_patience": 50,
    },
)
config = wandb.config

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
    model.add(Dropout(config.dropout_rate))

# output
num_classes = config.num_classes  # Number of classes in the problem
model.add(Dense(num_classes, activation="softmax"))

optimizer = Adam(learning_rate=config.learning_rate)
loss = config.loss_function

metrics = "accuracy"  # [score_training]

model.compile(loss=loss, optimizer=optimizer, metrics=metrics, run_eagerly=True)

print(model.summary())

## Prepare labels
# Convert labels to numbers
label_encoder = LabelEncoder()
Y_train_encoded = label_encoder.fit_transform(Y_train)
Y_val_encoded = label_encoder.fit_transform(Y_val)

# On-Hot encoding
Y_train_onehot = to_categorical(Y_train_encoded, num_classes)
Y_val_onehot = to_categorical(Y_val_encoded, num_classes)

# Compute class weights
class_weights = class_weight.compute_class_weight("balanced", classes=np.unique(Y_train_encoded), y=Y_train_encoded)
class_weights_dict = dict(enumerate(class_weights))

early_stopper = EarlyStopping(patience=config.early_stopping_patience, restore_best_weights=True)

history = model.fit(
    X_train,
    Y_train_onehot,
    validation_data=(X_val, Y_val_onehot),
    epochs=config.epochs,
    callbacks=[early_stopper, WandbCallback(save_model=False)],
    class_weight=class_weights_dict,
    batch_size=config.batch_size,
)

# save the model
model_filename = f'ARCAFF/models/CNN_{datetime.now().strftime("%Y%m%d-%H%M")}---{run.id}.keras'
model.save(model_filename)
artifact = wandb.Artifact(name="model", type="model", description="ARCAFF CNN")
artifact.add_file(model_filename)
wandb.log_artifact(artifact)


def log_image_wandb(plt_title):
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()  # Close the plot to free memory
    buf.seek(0)
    # Create a temporary file and write the buffer's content
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        tmp_file.write(buf.read())
        # Ensure you note the file name to upload
        tmp_file_path = tmp_file.name
    wandb.log({plt_title: wandb.Image(tmp_file_path)})
    buf.seek(0)
    buf.truncate()


### Train Dataset Plot ###
lbls = list(np.unique(Y_train))

with plt.style.context("seaborn-v0_8-darkgrid"):
    values = [np.count_nonzero(Y_train_encoded == j) for j in np.unique(Y_train_encoded)]
    total = sum(values)
    plt.figure(figsize=(12, 9))
    bars = plt.bar(list(np.unique(Y_train)), values)
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
            fontsize=9,
            rotation=90,
        )
    plt.xticks(rotation=90, ha="center")  # x-axis ticks labels
    # Adjust the upper limit of y-axis to include text
    # Calculate the necessary increase in the y-axis limit
    max_height = max(values)
    text_height = 200  # Estimated space needed for text
    additional_space = max_height * 0.2  # Add an extra 20% of the max height for spacing
    plt.ylim(0, max_height + text_height + additional_space)

    log_image_wandb("Train Dataset Distribution")

### Test Dataset ###
lbls_test = list(np.unique(Y_val))
y_test_pred = model.predict(X_val)
Y_test_pred_encoded = [np.argmax(y_test_pred[j]) for j in range(len(y_test_pred))]

# Classification report
report_dict = classification_report(Y_val_encoded, Y_test_pred_encoded, target_names=lbls_test, output_dict=True)
# Log the classification report to WandB
wandb.log({"classification_report_test": report_dict})

# Confusion Matrix
wandb.log(
    {
        "confusion_matrix_test": wandb.plot.confusion_matrix(
            y_true=Y_val_encoded, preds=Y_test_pred_encoded, class_names=lbls_test
        )
    }
)

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
    xticklabels=lbls_test,
    yticklabels=lbls_test,
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

log_image_wandb("Confusion Matrix")

# Finish the Wandb run
wandb.finish()
