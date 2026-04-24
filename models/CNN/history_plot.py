import matplotlib.pyplot as plt
import json

history_dict = json.load(open("models/CNN/training_history.json", "r"))

history = history_dict  # load your JSON

epochs = range(1, len(history["acc"]) + 1)

plt.figure(figsize=(12, 8))

# Accuracy
plt.subplot(2,2,1)
plt.plot(epochs, history["acc"], label="Train")
plt.plot(epochs, history["val_acc"], label="Val")
plt.ylim(0, 1)
plt.title("Accuracy")
plt.grid(True)
plt.legend()

# Loss
plt.subplot(2,2,2)
plt.plot(epochs, history["loss"], label="Train")
plt.plot(epochs, history["val_loss"], label="Val")
plt.ylim(0, 1)
plt.title("Loss")
plt.grid(True)
plt.legend()

# Precision
plt.subplot(2,2,3)
plt.plot(epochs, history["precision"], label="Train")
plt.plot(epochs, history["val_precision"], label="Val")
plt.ylim(0, 1)
plt.title("Precision (Defective)")
plt.grid(True)
plt.legend()

# Recall
plt.subplot(2,2,4)
plt.plot(epochs, history["recall"], label="Train")
plt.plot(epochs, history["val_recall"], label="Val")
plt.ylim(0, 1)
plt.title("Recall (Defective)")
plt.grid(True)
plt.legend()

plt.suptitle("CNN Training Metrics", fontsize=18, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()