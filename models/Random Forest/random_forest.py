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


X_A = []
X_B = []
y = []
file_paths = []

for label in ["Satisfactory", "Defective"]:
    folder = os.path.join(truffles_dir, label)

    for file in sorted(os.listdir(folder)):  # sorted = reproducibility
        path = os.path.join(folder, file)

        img = cv2.imread(path)
        if img is None:
            continue

        # Normalize to remove shadows
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_norm = clahe.apply(gray)

        # Threshold the image
        _, mask = cv2.threshold(gray_norm, 80, 255, cv2.THRESH_BINARY_INV)

        # Clean mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find the contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)

        # Features
        area = cv2.contourArea(contour)

        # Stablizie
        contour = cv2.convexHull(contour)

        # Create truffle mask
        truffle_mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(truffle_mask, [contour], -1, 255, -1)

        # Pink detection
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        lower_pink1 = np.array([0, 100, 140])
        upper_pink1 = np.array([5, 255, 255])

        lower_pink2 = np.array([175, 100, 140])
        upper_pink2 = np.array([179, 255, 255])

        mask1 = cv2.inRange(hsv, lower_pink1, upper_pink1)
        mask2 = cv2.inRange(hsv, lower_pink2, upper_pink2)

        pink_mask = mask1 | mask2

        # Multiple truffles detection
        contours = [c for c in contours if cv2.contourArea(c) > 500]
        num_contours = len(contours)

        # Brown pixels
        lower_brown = np.array([5, 50, 20])
        upper_brown = np.array([25, 255, 200])


        truffle_pixels = np.sum(truffle_mask > 0)
        pink_pixels = np.sum((pink_mask > 0) & (truffle_mask > 0))

        if truffle_pixels == 0:
            pink_ratio = 0
        else:
            pink_ratio = pink_pixels / truffle_pixels


        brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
        brown_pixels = np.sum((brown_mask > 0) & (truffle_mask > 0))

        if truffle_pixels == 0:
            brown_ratio = 0
        else:
            brown_ratio = brown_pixels / truffle_pixels

        # HSV variance
        truffle_pixels_hsv = hsv[truffle_mask > 0]
        if len(truffle_pixels_hsv) == 0:
            color_variance = 0
        else:
            color_variance = np.mean(np.var(truffle_pixels_hsv, axis=0))

        # Edge density
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)

        if hull_area == 0:
            solidity = 0
        else:
            solidity = area / hull_area

        pink_edges = cv2.Canny(pink_mask, 50, 150)

        if truffle_pixels == 0:
            pink_edge_density = 0
        else:
            pink_edge_density = np.sum(pink_edges > 0) / truffle_pixels

        # Shape integrity
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.sum(edges > 0) / truffle_pixels

        features_A = [area, solidity, edge_ratio]
        features_B = [pink_ratio, brown_ratio, pink_edge_density, color_variance]

        X_A.append(features_A)
        X_B.append(features_B)
        y.append(label)
        file_paths.append(path)

X_A = np.array(X_A)
X_B = np.array(X_B)
y = np.array(y)

# Split data
X_A_train, X_A_test, X_B_train, X_B_test, y_train, y_test, paths_train, paths_test = train_test_split(
    X_A, X_B, y, file_paths,
    test_size=0.2,
    stratify=y
)

# Run the models
model_A = RandomForestClassifier(
    n_estimators=400,
    max_depth=15,
    min_samples_leaf=5,
    random_state=42
)

model_B = RandomForestClassifier(
    n_estimators=400,
    max_depth=15,
    min_samples_leaf=5,
    random_state=42
)

model_A.fit(X_A_train, y_train)
model_B.fit(X_B_train, y_train)

pred_A = model_A.predict(X_A_test)
pred_B = model_B.predict(X_B_test)

y_pred = []

for a, b in zip(pred_A, pred_B):
    if a == "Defective" or b == "Defective":
        y_pred.append("Defective")
    else:
        y_pred.append("Satisfactory")

y_pred = np.array(y_pred)

# Metrics
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=["Satisfactory", "Defective"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Satisfactory", "Defective"])
disp.plot()
plt.show()

train_acc = (model_A.score(X_A_train, y_train) + model_B.score(X_B_train, y_train)) / 2
test_acc = (model_A.score(X_A_test, y_test) + model_B.score(X_B_test, y_test)) / 2

print("Train:", train_acc)
print("Test:", test_acc)


