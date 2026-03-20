import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # turn off debugging messages
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt

# DO_STATUS_UPDATES controls whether the script prints status updates during execution. 
# Set to False for a quieter run.
DO_STATUS_UPDATES = True
# SEEDED_SHUFFLE controls whether the dataset is shuffled with a fixed seed for reproducibility.
# If False, the dataset will be shuffled randomly each time, which may lead to different training/validation splits on each run.
SEEDED_SHUFFLE = True

try:
    X_DATA = np.load("x_resized.npy")
except FileNotFoundError:
    print("🚨🚨 CRITICAL RUNTIME ERROR DETECTED 🚨🚨")
    print("Oopsie Daisy! looks like you dont have the x_resized.npy file (the one that was too large to push to the repo)")
    print("Run the cnn_setup.py script to create it in the same directory as the x_data.npy and y_labels.npt files.")
    exit(1)

# left out of the try/catch because the file is in the repo
Y_LABELS = np.load("y_labels.npy")

# control and display constants
EPOCHS = 20                          # number of training epochs
TEST_TRAIN_RATIO = 0.25              # proportion of data to use for validation
IMAGE_SHAPE = (240, 240, 3)          # size of input images, should never change
BATCH_SIZE = 32                      # number of samples per training batch
HIDDEN_LAYER_ACTIVATION = "relu"     # activation function for hidden layers
OUTPUT_LAYER_ACTIVATION = "softmax"  # activation function for output layer
LEARNING_RATE = 0.001                # learning rate for the optimizer
SEED = 1738                          # random seed for reproducibility


# create a TensorFlow dataset from the loaded data
dataset = tf.data.Dataset.from_tensor_slices((X_DATA, Y_LABELS))

# split train/validation using TEST_TRAIN_RATIO
n = X_DATA.shape[0]
val_size = int(n * TEST_TRAIN_RATIO)   # ensure whole sample size
train_size = n - val_size

if SEEDED_SHUFFLE:
    dataset = dataset.shuffle(n, seed=SEED)
else:
    dataset = dataset.shuffle(n)

train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)

train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

if DO_STATUS_UPDATES:
    print(f"[/] Total samples: {n}\n - training samples: {train_size}\n - validation samples: {val_size}")


# base model before training on the data
base_model = EfficientNetB1(
    input_shape=IMAGE_SHAPE,  # layer shape
    include_top=False,          # exclude the final classification layer
    weights="imagenet"          # use pre-trained weights from ImageNet
)

# Freeze the base model to prevent its weights from being updated during training
# we want this because the weights for the base model are already well-tuned for 
# general image feature extraction, and we only want to train the new classification 
# head on top of it.
base_model.trainable = False

# Add custom classification head
x = base_model.output                                           # start with the output of the base model
x = layers.GlobalAveragePooling2D()(x)                          # global average pooling to reduce spatial dimensions
x = layers.BatchNormalization()(x)                              # batch normalization for faster convergence and better performance
x = layers.Dense(128, activation=HIDDEN_LAYER_ACTIVATION)(x)    # fully connected layer with specified activation function
x = layers.Dropout(0.3)(x)                                      # dropout for regularization to prevent overfitting

# Adjust output layer depending on your labels
# there will be 2 classes with 0 being satisfactory and 1 being defective.
# we have two output neurons with softmax activation to get probabilities for each class
num_classes = len(np.unique(Y_LABELS))
outputs = layers.Dense(num_classes, activation=OUTPUT_LAYER_ACTIVATION)(x)
# Create the final model structure
model = models.Model(inputs=base_model.input, outputs=outputs)

# Compile the model with an appropriate optimizer, loss function, and metrics
model.compile(
    # more info on Adam optimizer: https://keras.io/api/optimizers/adam/
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

print("[/] printing final training accuracy and loss")
final_train_acc = history.history["accuracy"][-1]
final_val_acc = history.history["val_accuracy"][-1]
final_train_loss = history.history["loss"][-1]
final_val_loss = history.history["val_loss"][-1]
print(f" - final training accuracy: {final_train_acc:.4f}")
print(f" - final validation accuracy: {final_val_acc:.4f}")
print(f" - final training loss: {final_train_loss:.4f}")
print(f" - final validation loss: {final_val_loss:.4f}") 

print("[/] plotting training history")

plt.plot(history.history["accuracy"], label="train acc")
plt.plot(history.history["val_accuracy"], label="val acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()