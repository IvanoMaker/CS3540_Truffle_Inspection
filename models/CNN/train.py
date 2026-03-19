import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # turn off debugging messages
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras import layers, models, optimizers
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
EPOCHS = 20                 # number of training epochs
TEST_TRAIN_RATIO = 0.25     # ratio of validation samples to total samples
IMAGE_SIZE = (240, 240)     # size of input images
BATCH_SIZE = 32             # number of samples per training batch
HIDDEN_LAYER_ACTIVATION = "relu"  # activation function for hidden layers
OUTPUT_LAYER_ACTIVATION = "softmax"  # activation function for output layer
LEARNING_RATE = 1e-3              # learning rate for the optimizer

dataset = tf.data.Dataset.from_tensor_slices((X_DATA, Y_LABELS))

# split train/validation using TEST_TRAIN_RATIO
n = X_DATA.shape[0]
val_size = int(n * TEST_TRAIN_RATIO)
train_size = n - val_size

dataset = dataset.shuffle(n, seed=1738)

train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)

train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

if DO_STATUS_UPDATES:
    print(f"[/] Total samples: {n}\n - training samples: {train_size}\n - validation samples: {val_size}")

# base model before training on the data
base_model = EfficientNetB1(
    input_shape=(240, 240, 3),  # layer shape
    include_top=False,          # exclude the final classification layer
    weights="imagenet"          # use pre-trained weights from ImageNet
)

# Freeze the base model to prevent its weights from being updated during training
base_model.trainable = False

# Add custom classification head
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(128, activation=HIDDEN_LAYER_ACTIVATION)(x)
x = layers.Dropout(0.3)(x)

# Adjust output layer depending on your labels
# there will be 2 classes with 0 being satisfactory and 1 being defective.
# we have two output neurons with softmax activation to get probabilities for each class
num_classes = len(np.unique(Y_LABELS))
outputs = layers.Dense(num_classes, activation=OUTPUT_LAYER_ACTIVATION)(x)
# Create the final model structure
model = models.Model(inputs=base_model.input, outputs=outputs)

# Compile the model with an appropriate optimizer, loss function, and metrics
model.compile(
    optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
    # we use sparse_categorical_crossentropy because our labels are integers (0 and 1) 
    # rather than one-hot encoded vectors
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

if DO_STATUS_UPDATES:
    print("[/] Base model constructed")
    print(" - architectre: EfficientNetB1")
    print(" - weights: ImageNet")
    print(f" - output classes: {num_classes}")
    print(f" - layers: {len(model.layers)}")
    print(f" - total neurons: {model.count_params()}")
    print(f" - hidden layer activation: {HIDDEN_LAYER_ACTIVATION}")
    print(f" - output layer activation: {OUTPUT_LAYER_ACTIVATION}")
    print(f" - learning rate: {LEARNING_RATE}")
    print(f" - optimizer: Adam")
    print(f" - loss function: sparse_categorical_crossentropy")
    print(f" - training epochs: {EPOCHS}")

print("[/] starting model training")
print("[/] grab a snack, this may take a while...")
# Train the model on the training dataset and validate on the validation dataset
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS
)

if DO_STATUS_UPDATES:
    print("[/] training complete!")

print("[?] would you like to save the model to disk?")
temp_input = input("    (y/n): ").lower()

if temp_input == "y":
    model.save("truffle_cnn_model.h5")
    print("[/] model saved as truffle_cnn_model.h5")

print("[/] plotting training history")

plt.plot(history.history["accuracy"], label="train acc")
plt.plot(history.history["val_accuracy"], label="val acc")
plt.legend()
plt.show()