import cv2
import numpy as np
import os
from pathlib import Path
from ultralytics import YOLO
from tensorflow.keras.models import load_model

# =========================
# MODEL PATHS
# =========================
DEFAULT_YOLO_MODEL = "truffle_yolo_model_video.pt"
FALLBACK_YOLO_MODEL = "runs/detect/train6/weights/best.pt"
CNN_MODEL = "truffle_cnn_model.h5"
VIDEO_FILE = "Truffle Defect recording.mp4"

# =========================
# LOAD MODELS
# =========================
if Path(DEFAULT_YOLO_MODEL).exists():
    yolo_model_path = DEFAULT_YOLO_MODEL
else:
    yolo_model_path = FALLBACK_YOLO_MODEL
    print(f"Warning: {DEFAULT_YOLO_MODEL} not found, falling back to {FALLBACK_YOLO_MODEL}")

print(f"Loading YOLO model from: {yolo_model_path}")
yolo_model = YOLO(yolo_model_path)
print("YOLO model loaded")
print("Classes:", yolo_model.names)

if not Path(CNN_MODEL).exists():
    raise FileNotFoundError(f"CNN model not found: {CNN_MODEL}")
cnn_model = load_model(CNN_MODEL)

# Must match your training setup
IMG_SIZE = (240, 240)

# =========================
# CLASS LABELS
# =========================
CLASS_NAMES = ["Good", "Defective"]

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
    return CLASS_NAMES[class_idx], confidence, preds


# =========================
# VIDEO INPUT
# =========================
cap = cv2.VideoCapture(VIDEO_FILE)

if not cap.isOpened():
    print(f"Error: Could not open video file: {VIDEO_FILE}")
    exit()

# Get video properties for output
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define output video path
output_video_path = "processed_" + VIDEO_FILE

# Create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for AVI format if needed
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

print(f"Processing video and saving to: {output_video_path}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo_model(frame)[0]

    if len(results.boxes) > 0:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            label, conf, preds = classify_truffle(crop)
            color = (0, 255, 0) if label == "Good" else (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{label} ({conf:.2f})",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

    # Write the processed frame to the output video
    out.write(frame)

    cv2.imshow("Truffle Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processed video saved to: {output_video_path}")