import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # turn off debugging messages
import tensorflow as tf
import matplotlib.pyplot as plt

DO_STATUS_UPDATES = True

try:
    X_DATA = np.load("x_resized.npy")
except FileNotFoundError:
    print("🚨🚨 CRITICAL RUNTIME ERROR DETECTED 🚨🚨")
    print("Oopsie Daisy! looks like you dont have the x_resized.npy file (the one that was too large to push to the repo)")
    print("Run the cnn_setup.py script to create it in the same directory as the x_data.npy and y_labels.npt files.")
    exit(1)

Y_LABELS = np.load("y_labels.npy")

# control and display constants
EPOCHS = 20
TEST_TRAIN_RATIO = 0.1
IMAGE_SIZE = (240, 240)
BATCH_SIZE = 32

dataset = tf.data.Dataset.from_tensor_slices((X_DATA, Y_LABELS))

# split train/validation using TEST_TRAIN_RATIO
n = X_DATA.shape[0]
val_size = int(n * TEST_TRAIN_RATIO)
train_size = n - val_size

dataset = dataset.shuffle(n, seed=42)

train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)

train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print(f"Total samples: {n}, train: {train_size}, val: {val_size}")