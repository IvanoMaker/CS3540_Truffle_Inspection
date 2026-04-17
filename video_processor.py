import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model

# =========================
# LOAD MODELS
# =========================
yolo_model = YOLO("truffle_yolo_model.pt")  # pretrained YOLOv8 (auto-downloads)
cnn_model = load_model("truffle_cnn_model.h5")

# Must match your training setup
IMG_SIZE = (240, 240)

# =========================
# CLASS LABELS (EDIT THIS)
# =========================
CLASS_NAMES = ["Good", "Defective"]  # adjust if you have more classes

# =========================
# CNN CLASSIFIER
# =========================
def classify_truffle(image):
    img = cv2.resize(image, IMG_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    preds = cnn_model.predict(img, verbose=0)[0]
    class_idx = np.argmax(preds)
    confidence = preds[class_idx]

    return CLASS_NAMES[class_idx], confidence


# =========================
# VIDEO INPUT
# =========================
# cap = cv2.VideoCapture(0)  # webcam
cap = cv2.VideoCapture("Truffle Defect recording.mp4")  # use video file instead

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # =========================
    # YOLO DETECTION
    # =========================
    results = yolo_model(frame)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Crop detected region
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # =========================
        # CNN CLASSIFICATION
        # =========================
        label, conf = classify_truffle(crop)

        # =========================
        # DRAW RESULTS
        # =========================
        color = (0, 255, 0) if label == "Good" else (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{label} ({conf:.2f})",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    cv2.imshow("Truffle Defect Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()