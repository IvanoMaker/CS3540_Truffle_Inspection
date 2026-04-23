import numpy as np
import random
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

random.seed(0)
x_data = np.load("x_resized.npy")
labels = np.load("y_labels.npy")
model = load_model("truffle_cnn_model.h5")

SAMPLE_SIZE = 200

size = x_data.shape[0]

SAMPLE_INDICES = [random.randint(0, size - 1) for _ in range(SAMPLE_SIZE)]

SAMPLES = x_data[SAMPLE_INDICES]
SAMPLE_LABELS = labels[SAMPLE_INDICES]

y_pred_probs = model.predict(SAMPLES)
y_pred_probs = y_pred_probs.ravel()

y_pred = (y_pred_probs > 0.5).astype(int)

# Metrics
print("Confusion Matrix:")
print(confusion_matrix(SAMPLE_LABELS, y_pred))

print("\nClassification Report:")
print(classification_report(SAMPLE_LABELS, y_pred))

'''
Original Model with 200 random sampels and seed=0:
              precision    recall  f1-score   support

           0       0.81      1.00      0.90       163
           1       0.00      0.00      0.00        37

    accuracy                           0.81       200
   macro avg       0.41      0.50      0.45       200
weighted avg       0.66      0.81      0.73       200
'''