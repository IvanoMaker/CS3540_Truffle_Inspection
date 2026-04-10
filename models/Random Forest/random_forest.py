import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import os

script_dir = Path(__file__).resolve().parent
parent_dir = script_dir.parents[1]
truffles_dir = parent_dir / "dataset"

def load_dataset(base_path):
    X = []
    y = []

    for label in ["Satisfactory", "Defective"]:
        folder = os.path.join(base_path, label)

        for file in os.listdir(folder):
            path = os.path.join(folder, file)

            # Get the features from each image
            img = cv2.imread(path)
            if img is None:
                return None

            # Assumes the darkest object on the image will be the truffle.
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None
            contour = max(contours, key=cv2.contourArea)

            # Area
            area = cv2.contourArea(contour)

            # Circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                circularity = 0
            else:
                circularity = 4 * np.pi * area / (perimeter ** 2)

            # Create mask for just the truffle
            truffle_mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(truffle_mask, [contour], -1, 255, -1)

            # Pink ratio within truffle
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # Adjust these if needed
            lower_pink1 = np.array([0, 100, 140])
            upper_pink1 = np.array([5, 255, 255])

            lower_pink2 = np.array([175, 100, 140])
            upper_pink2 = np.array([179, 255, 255])

            mask1 = cv2.inRange(hsv, lower_pink1, upper_pink1)
            mask2 = cv2.inRange(hsv, lower_pink2, upper_pink2)

            pink_mask = mask1 | mask2

            truffle_pixels = np.sum(truffle_mask > 0)
            pink_pixels = np.sum((pink_mask > 0) & (truffle_mask > 0))

            if truffle_pixels == 0:
                pink_ratio = 0
            else:
                pink_ratio = pink_pixels / truffle_pixels
            features = [circularity, pink_ratio, area]
            if features is not None:
                X.append(features)
                y.append(label)

    return np.array(X), np.array(y)

X, names = load_dataset(truffles_dir)

X_train, X_test, y_train, y_test = train_test_split(X, names)
model = RandomForestClassifier(n_estimators = 400, max_depth=15, min_samples_leaf=5, random_state = 42)
model.fit(X_train, y_train)

# Get a bunch of metrics
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=["Satisfactory", "Defective"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Satisfactory", "Defective"])
disp.plot()
plt.show()

scores = cross_val_score(model, X, names, cv=5)
print("CV accuracy: ", scores)
print("Mean: ", scores.mean())
