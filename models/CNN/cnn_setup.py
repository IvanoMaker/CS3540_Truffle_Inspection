# This script creates a local copy of the quite large 240x240x3 shaped input data for the neural network to train with
# you should only need to run this and have the file if you are doing trianing tests and what not.

X_DATA_PATH = "x_data_flattened.npy"
Y_LABELS_PATH = "y_labels.npy"
DO_LOGGING = True

import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # turn off debugging messages
if DO_LOGGING:
    print("1/4 - loading tensorflow, may take a second")
import tensorflow as tf

# NOTE: This file is intended to be executed in VSCode with the working directory set to 
# the root of the project. If you run it from a different location or from the terminal in the CNN folder, 
# you may need to adjust X_DATA_PATH and Y_LABELS_PATH accordingly.  -Evan

# reconstruct flattened data to 128x128 color images
X_DATA = np.load(X_DATA_PATH).reshape(-1, 128, 128, 3)
Y_LABELS = np.load(Y_LABELS_PATH)

INPUT_LAYER_SIZE = (240, 240) # input layer size for EfficientNetB1

# convert image to a 3D Numpy array
# takes image path input and returns 240x240x3 matrix (width x height x rgb channels)
def convert_image_to_3dmtx(img, target_size=INPUT_LAYER_SIZE):
    img = tf.image.resize(img, target_size)
    return img.numpy() / 255.0

if __name__ == "__main__":
    if DO_LOGGING:
        print(f"2/4 - n samples: {X_DATA.shape[0]}, image dimensions: {X_DATA.shape[1:]}")

    reshaped_x_data = np.array([convert_image_to_3dmtx(img) for img in X_DATA])

    if DO_LOGGING:
        print(f"3/4 - reshaped x data shape: {reshaped_x_data.shape[1:]}")

    np.save("x_resized.npy", reshaped_x_data)

    if DO_LOGGING:
        print(f"4/4 - saved file successfully")