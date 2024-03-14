# %%
# Operating System
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # only use one GPU
import os.path
import time
import sys
sys.path.insert(0,'../')
import io
from datetime import datetime

# Data Handling
import numpy as np
import pandas as pd
import glob
import utilities_cutout as ut
import tempfile

# Logging
import wandb
from wandb.keras import WandbCallback
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

import neptune
from neptune.integrations.tensorflow_keras import NeptuneCallback
from neptune.types import File

from comet_ml import Experiment

# TensorFlow
import tensorflow as tf
from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.client import device_lib

# Keras
from keras.utils import to_categorical
from keras.layers import Dense, Flatten, Dropout, ZeroPadding3D
from keras.models import Sequential, load_model
from keras.layers import (Conv2D, MaxPooling3D, Conv3D, MaxPooling2D)
from keras.layers import BatchNormalization
from keras import backend as K
from keras import regularizers
from keras.layers import Activation
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

# sklearn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight

# Images
import matplotlib.pyplot as plt
import seaborn as sns
from astropy.io import fits
from PIL import Image
import cv2

# %%
gpus = tf.config.experimental.list_physical_devices('GPU')

# dynamically allocate necessary GPU memory
if gpus:
    try:
        # Memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPU")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
else:
    print("No GPU found, training will run on CPU.")

# %%
loaded_data = np.load('/ARCAFF/data/grouped_skft_split.npz')
X_train = loaded_data['X_train']
Y_train = loaded_data['Y_train']
X_val = loaded_data['X_val']
Y_val = loaded_data['Y_val']

# %%
print(f"Size of X_train: {sys.getsizeof(X_train)/ (1024.0 ** 2)/1000:.2f} GB") 

# %%
config_dict = {
        "number_of_conv_layers": 5,
        "conv_layers_units": [128, 128, 128, 128, 128, 128, 128],
        "kernel_sizes": [(7, 7), (5, 5), (3, 3), (3, 3), (3, 3)],
        "padding_types": "same",
        "epochs": 2000,
        "regularization_lambda": 0.03,
        "initializer": 'glorot_uniform',
        "learning_rate": 1e-4,
        "optimizer": "adam",
        "loss_function": 'categorical_crossentropy',
        "batch_size": 512,
        "dropout_rate": 0.25,
        "dataset_name": "grouped_sktf_data",
        "num_classes": len(np.unique(Y_train)),
        "MLP_layers": [512, 512, 256, 128],
        "MLP_activation": 'relu',
        "dropout_rate": 0.25,
        "early_stopping_patience": 200,
}

# %%
# Logging initialization
# W&B
wandb.login()
run = wandb.init(project="Active Region Cutout Classification", entity = "arcaff", 
    config=config_dict)
config = wandb.config

# Neptune
run_neptune = neptune.init_run(
     project="ARCAFF/Active-Region-Cutout-Classification"
)
neptune_callback = NeptuneCallback(run = run_neptune)

# Comet
run_comet = Experiment(project_name="active-region-cutout-classification", workspace="arcaff")

# %%
# vertical and horizontal flip of the images
data_augment = True
if data_augment:
    print("Performing data augmentation ... \n")
    # Assuming X_train and Y_train are defined
    labels = ['Alpha', 'Beta-Gamma', 'Beta-Gamma-Delta', 'Beta-Delta']

    # Initialize an empty list to hold the augmented images and their labels
    augmented_images = []
    augmented_labels = []

    # Find indices of the current label
    indices = np.where(Y_train == 'Beta')[0]
    # Extract the corresponding images
    label_images = X_train[indices]
    # Perform the flips
    label_horverflip = label_images[:, ::-1, ::-1]

    # Concatenate the flipped images
    label_augmented = np.concatenate([
    label_horverflip
    ], axis=0)

    # Append the augmented images to the list
    augmented_images.append(label_augmented)

    # Create an array of labels for the augmented images (same label for each set of original and flipped images)
    label_augmented_labels = np.array(['Beta'] * label_augmented.shape[0])
    # Append the augmented labels to the list
    augmented_labels.append(label_augmented_labels)

    for label in labels:
        # Find indices of the current label
        indices = np.where(Y_train == label)[0]
        # Extract the corresponding images
        label_images = X_train[indices]
        # Perform the flips
        label_horflip = label_images[:, :, ::-1]
        label_verflip = label_images[:, ::-1, :]
        label_horverflip = label_images[:, ::-1, ::-1]
        
        # Concatenate the flipped images
        label_augmented = np.concatenate([
            label_horflip,
            label_verflip,
            label_horverflip
        ], axis=0)
        
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
    print("Flipping done.\n")
# %%
number_channels = 1
input_shape = (X_train.shape[1], X_train.shape[2], number_channels)

model = Sequential()

# Loop through the number of convolutional layers
for i in range(config.number_of_conv_layers):
    if i == 0:
        # Add Conv2D layer
        model.add(Conv2D(
            config.conv_layers_units[i], 
            config.kernel_sizes[i], 
            strides=(2,2),  
            padding=config.padding_types,
            kernel_regularizer=regularizers.l2(config.regularization_lambda),
            kernel_initializer=config.initializer, 
            input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2, 2), strides=(1, 1)))
    else:
        # Add Conv2D layer
        model.add(Conv2D(
        config.conv_layers_units[i], 
        config.kernel_sizes[i], 
        strides=(2,2),  
        padding=config.padding_types,
        kernel_regularizer=regularizers.l2(config.regularization_lambda),
        kernel_initializer=config.initializer))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2, 2), strides=(1, 1)))

# MLP
model.add(Flatten())
for units in config.MLP_layers:
    model.add(Dense(units, activation=config.MLP_activation))
    model.add(Dropout(wandb.config.dropout_rate))

#output
num_classes = config.num_classes 
model.add(Dense(num_classes, activation='softmax'))

optimizer = Adam(learning_rate=config.learning_rate) 

loss = config.loss_function 

metrics = 'accuracy'

model.compile(loss=loss, optimizer=optimizer, metrics=metrics, run_eagerly=True)

print(model.summary())

# %%
# make histogram of train dataset
with plt.style.context('seaborn-v0_8-darkgrid'): 
    labels = ['0.0', 'Alpha', 'Beta', 'Beta-Gamma', 'Beta-Gamma-Delta', 'Beta-Delta']
    greek_lables = ['0', r'$\alpha$', r'$\beta$', r'$\beta-\gamma$', r'$\beta-\gamma-\delta$', r'$\beta-\delta$']
    values = [np.count_nonzero(Y_train == j) for j in labels]
    total = sum(values) 

    plt.figure(figsize=(10, 6))
    bars = plt.bar(greek_lables, values)

    # Add value labels and percentage above bars
    for bar in bars:
        yval = bar.get_height()
        percentage = f"{yval/total*100:.2f}%" 
        plt.text(bar.get_x() + bar.get_width()/2, yval + 200, f"{yval} ({percentage})", ha='center', va='bottom', fontsize=11)

    plt.xticks(rotation=0, ha='center', fontsize=11)       # x-axis ticks labels
    plt.yticks(fontsize=12)                                # y-axis ticks labels
    plt.ylabel('Number of Samples', fontsize=14)           # y-axis label
    #plt.xlabel('Categories', fontsize=14)                 # x-axis label
    plt.title("Train Dataset", fontsize=16)  # title

    # Save the plot to a buffer (in-memory file)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()  # Close the plot to free memory
    buf.seek(0)  # Ensure the buffer's pointer is at the start
    # Create a temporary file and write the buffer's content
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
        tmp_file.write(buf.read())
        tmp_file_path = tmp_file.name
    # Upload the temporary file
    run_neptune['heatmaps/train_dataset_histogram'].upload(tmp_file_path)
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
class_weights = class_weight.compute_class_weight('balanced',
                                                  classes=np.unique(Y_train_encoded),
                                                  y=Y_train_encoded)
class_weights_dict = dict(enumerate(class_weights))

# %%
early_stopper = EarlyStopping(
    patience=config.early_stopping_patience,
    restore_best_weights=True)

history = model.fit(X_train, Y_train_onehot, 
          validation_data=(X_val, Y_val_onehot), 
          epochs = config.epochs,
          callbacks=[
              early_stopper, 
              WandbCallback(save_model=False), 
              neptune_callback, 
              run_comet.get_callback('keras')], 
          class_weight=class_weights_dict,
          batch_size=config.batch_size)

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
wandb.log({"confusion_matrix_test": wandb.plot.confusion_matrix(y_true=Y_val_encoded,
                                                           preds=Y_test_pred_encoded,
                                                           class_names=lbls_test)})

run_comet.log_confusion_matrix(
    y_true=Y_val_encoded, 
    y_predicted=Y_test_pred_encoded, 
    labels=lbls_test)

# %%
greek_lbls = ['QS',r'$\alpha$', r'$\beta$', r'$\beta-\delta$', r'$\beta-\gamma$', r'$\beta-\gamma-\delta$']
conf_matrix = confusion_matrix(Y_val_encoded, Y_test_pred_encoded)

# Calculate percentages
conf_matrix_percentage = conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
conf_matrix_percentage_with_sign = np.array([["{0:.2f}%".format(value) for value in row] for row in conf_matrix_percentage])

# Plotting
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_percentage, annot=conf_matrix_percentage_with_sign, fmt="", cmap="Blues", cbar=False,
            xticklabels=greek_lbls, yticklabels=greek_lbls)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# %%
# Binary Classification
lbls = ["Quiet", "AR"]

Y_test_binary = np.zeros(len(Y_val_encoded))
for j in range(len(Y_val_encoded)):
    if Y_val_encoded[j]>=1:
        Y_test_binary[j] = 1

Y_test_pred_binary = np.zeros(len(Y_test_pred_encoded))
for j in range(len(Y_test_pred_binary)):
    if Y_test_pred_encoded[j]>=1:
        Y_test_pred_binary[j] = 1

conf_matrix = confusion_matrix(Y_test_binary, Y_test_pred_binary)
# Calculate percentages
conf_matrix_percentage = conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
conf_matrix_percentage_with_sign = np.array([["{0:.2f}%".format(value) for value in row] for row in conf_matrix_percentage])
# %%
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_percentage, annot=conf_matrix_percentage_with_sign, fmt="", cmap="Blues", cbar=False,
            xticklabels=lbls, yticklabels=lbls)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
# Save the plot to a buffer (in-memory file)
buf = io.BytesIO()
plt.savefig(buf, format='png')
plt.close()  # Close the plot to free memory
# Ensure the buffer's pointer is at the start
buf.seek(0)
# Create a temporary file and write the buffer's content
with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
    tmp_file.write(buf.read())
    # Ensure you note the file name to upload
    tmp_file_path = tmp_file.name

# Log binary confusion matrix
run_neptune['heatmaps/binary_confusion_matrix'].upload(tmp_file_path)
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
Experiment.log_table("Binary_Metrics.csv", [["Metric", "Value"]] + list(formatted_scores))
for name, score in formatted_scores:
    run[f"Binary_Metrics/{name}"].log(score)

