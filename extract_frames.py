import cv2
import os
from pathlib import Path

# Configuration
VIDEO_FILE = "Truffle Defect recording.mp4"
OUTPUT_DIR = "dataset/video_frames"
FRAME_INTERVAL = 30  # Extract every 30th frame (adjust for more/fewer frames)

# Create output directory
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_FILE)

if not cap.isOpened():
    print(f"Error: Cannot open {VIDEO_FILE}")
    exit()

frame_count = 0
saved_count = 0
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video FPS: {fps}")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    if frame_count % FRAME_INTERVAL == 0:
        filename = f"{OUTPUT_DIR}/frame_{saved_count:04d}.jpg"
        cv2.imwrite(filename, frame)
        saved_count += 1
        print(f"Saved: {filename}")
    
    frame_count += 1

cap.release()
print(f"\nExtracted {saved_count} frames to {OUTPUT_DIR}/")
print(f"Total frames in video: {frame_count}")
