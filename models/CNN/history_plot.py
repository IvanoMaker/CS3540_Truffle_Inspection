import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

x_data = np.load("x_resized.npy")
labels = np.load("y_labels.npy")
model = load_model("truffle_cnn_model.h5")

size = x_data.shape[0]

y_pred_probs = model.predict(x_data)
y_pred_probs = y_pred_probs.ravel()
y_pred = (y_pred_probs > 0.5).astype(int)

# Metrics
cm = confusion_matrix(labels, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=["Satisfactory", "Defective"])
disp.plot(cmap=plt.cm.RdPu)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for Truffle CNN Model")
plt.show()