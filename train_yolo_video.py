"""
Train YOLO on video annotations only
"""

from ultralytics import YOLO
import os
from pathlib import Path
import shutil
import yaml
import torch

# Paths
VIDEO_DATASET = "dataset/video_dataset"
VIDEO_TRAIN_DATASET = "dataset/video_train_dataset"

def prepare_video_dataset():
    """Prepare video dataset for training with train/val split."""
    
    print("Preparing video dataset for training...")
    
    # Create directory structure
    Path(f"{VIDEO_TRAIN_DATASET}/images/train").mkdir(parents=True, exist_ok=True)
    Path(f"{VIDEO_TRAIN_DATASET}/images/val").mkdir(parents=True, exist_ok=True)
    Path(f"{VIDEO_TRAIN_DATASET}/labels/train").mkdir(parents=True, exist_ok=True)
    Path(f"{VIDEO_TRAIN_DATASET}/labels/val").mkdir(parents=True, exist_ok=True)
    
    # Get all images and labels
    video_imgs = list(Path(f"{VIDEO_DATASET}/images").glob("*"))
    video_lbls = list(Path(f"{VIDEO_DATASET}/labels").glob("*"))
    
    # Sort to ensure matching
    video_imgs.sort()
    video_lbls.sort()
    
    n = len(video_imgs)
    if n == 0:
        print("No images found in video_dataset")
        return None
    
    # Split 80/20 for train/val
    train_size = int(0.8 * n)
    
    print(f"Total video frames: {n}")
    print(f"Training: {train_size}, Validation: {n - train_size}")
    
    for i, (img, lbl) in enumerate(zip(video_imgs, video_lbls)):
        if i < train_size:
            shutil.copy(img, f"{VIDEO_TRAIN_DATASET}/images/train/{img.name}")
            shutil.copy(lbl, f"{VIDEO_TRAIN_DATASET}/labels/train/{lbl.name}")
        else:
            shutil.copy(img, f"{VIDEO_TRAIN_DATASET}/images/val/{img.name}")
            shutil.copy(lbl, f"{VIDEO_TRAIN_DATASET}/labels/val/{lbl.name}")
    
    # Create data.yaml
    data_yaml = {
        'path': os.path.abspath(VIDEO_TRAIN_DATASET),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 1,
        'names': ['truffle']
    }
    
    yaml_path = f"{VIDEO_TRAIN_DATASET}/data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f)
    
    print(f"Video dataset prepared at {VIDEO_TRAIN_DATASET}")
    
    return yaml_path

def train_model(data_yaml):
    """Train YOLO model on video dataset."""
    
    print("\n=== Training YOLO Model on Video Annotations Only ===")
    
    # Load pre-trained model
    model = YOLO("yolov8n.pt")
    
    # Determine device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    # Train
    results = model.train(
        data=data_yaml,
        epochs=50,  # More epochs since smaller dataset
        imgsz=640,
        patience=10,
        device=device,
        name="video_only_trained"
    )
    
    # Save best model
    model.save("truffle_yolo_model_video_only.pt")
    print("\nModel saved as truffle_yolo_model_video_only.pt")
    
    return results

if __name__ == "__main__":
    import sys
    
    # Check if video annotations exist
    if not os.path.exists(VIDEO_DATASET):
        print("Error: Video annotations not found.")
        print("Please run annotate_frames.py first.")
        sys.exit(1)
    
    dataset_count = len(list(Path(f"{VIDEO_DATASET}/labels").glob("*.txt")))
    if dataset_count == 0:
        print("Error: No annotated frames found in dataset/video_dataset/")
        sys.exit(1)
    
    print(f"Found {dataset_count} annotated video frames")
    
    # Prepare video dataset
    yaml_path = prepare_video_dataset()
    
    if yaml_path is None:
        print("Failed to prepare dataset")
        sys.exit(1)
    
    # Train model
    train_model(yaml_path)
    
    print("\n=== Training Complete ===")
    print("Next: Test the model with python video_processor.py")
    print("(Update video_processor.py to use 'truffle_yolo_model_video_only.pt')")
