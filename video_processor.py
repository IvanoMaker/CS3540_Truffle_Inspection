import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import joblib

# =========================
# MODEL PATHS
# =========================
DEFAULT_YOLO_MODEL = "truffle_yolo_model_video.pt"
FALLBACK_YOLO_MODEL = "runs/detect/train6/weights/best.pt"
CNN_MODEL = "truffle_cnn_model.h5"
SVM_MODEL = "SVM_model.joblib"
VIDEO_FILE = "Truffle Defect recording.mp4"

# =========================
# USER CONFIGURATION
# =========================
# Set to "CNN" or "SVM"
CLASSIFIER = "SVM"

# Must match your training setup for each model
CNN_IMG_SIZE = (240, 240)
SVM_IMG_SIZE = (128, 128)

# =========================
# CLASS LABELS
# =========================
CLASS_NAMES = ["Good", "Defective"]

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

if CLASSIFIER == "CNN":
    if not Path(CNN_MODEL).exists():
        raise FileNotFoundError(f"CNN model not found: {CNN_MODEL}")
    prediction_model = load_model(CNN_MODEL)
    IMG_SIZE = CNN_IMG_SIZE
    print("Using CNN classifier")
elif CLASSIFIER == "SVM":
    if not Path(SVM_MODEL).exists():
        raise FileNotFoundError(f"SVM model not found: {SVM_MODEL}")
    prediction_model = joblib.load(SVM_MODEL)
    IMG_SIZE = SVM_IMG_SIZE
    print("Using SVM classifier")
else:
    raise ValueError(f"Unknown classifier: '{CLASSIFIER}'. Choose 'CNN' or 'SVM'.")

# =========================
# CLASSIFIER FUNCTION
# =========================
def classify_truffle(image):
    target_w, target_h = IMG_SIZE
    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2
    half = min(cx, cy, target_w // 2, target_h // 2)

    # Center crop matching the training pipeline
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w, cx + half)
    y2 = min(h, cy + half)
    crop = image[y1:y2, x1:x2]

    # Resize to exact target size
    resized = cv2.resize(crop, (target_w, target_h))

    # Convert BGR to RGB to match PIL's channel ordering used during training
    resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # cv2.imshow("Cropped Truffle", resized)

    if CLASSIFIER == "SVM":
        img_array = np.array(resized_rgb).flatten().reshape(1, -1)
        label_idx = prediction_model.predict(img_array)[0]
        proba = prediction_model.predict_proba(img_array)[0]
        confidence = proba[label_idx]
        return CLASS_NAMES[label_idx], confidence

    elif CLASSIFIER == "CNN":
        img = resized_rgb / 255.0
        img = np.expand_dims(img, axis=0)
        preds = prediction_model.predict(img, verbose=0)[0]
        label_idx = np.argmax(preds)
        confidence = preds[label_idx]
        return CLASS_NAMES[label_idx], confidence

# =========================
# VIDEO INPUT
# =========================
cap = cv2.VideoCapture(VIDEO_FILE)

if not cap.isOpened():
    print(f"Error: Could not open video file: {VIDEO_FILE}")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

output_video_path = "processed_" + VIDEO_FILE

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

print(f"Processing video and saving to: {output_video_path}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    raw_frame = frame.copy()

    results = yolo_model(frame)[0]

    if len(results.boxes) > 0:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if x2 <= x1 or y2 <= y1:
                continue

            crop = raw_frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            label, confidence = classify_truffle(crop)
            color = (0, 255, 0) if label == "Good" else (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{label} ({confidence:.2f})",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

    out.write(frame)
    cv2.imshow("Truffle Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processed video saved to: {output_video_path}")