"""
Train YOLO on combined dataset (original + video annotations)
"""

from ultralytics import YOLO
import os
from pathlib import Path
import shutil
import yaml
import torch

# Paths
ORIGINAL_DATASET = "dataset"
VIDEO_DATASET = "dataset/video_dataset"
COMBINED_DATASET = "dataset/combined_dataset"

def create_combined_dataset():
    """Combine original dataset with video annotations."""
    
    print("Creating combined dataset...")
    
    # Create combined directory structure
    Path(f"{COMBINED_DATASET}/images/train").mkdir(parents=True, exist_ok=True)
    Path(f"{COMBINED_DATASET}/images/val").mkdir(parents=True, exist_ok=True)
    Path(f"{COMBINED_DATASET}/labels/train").mkdir(parents=True, exist_ok=True)
    Path(f"{COMBINED_DATASET}/labels/val").mkdir(parents=True, exist_ok=True)
    
    # Copy original training data
    print("Copying original training data...")
    original_imgs = Path(f"{ORIGINAL_DATASET}/train/truffles")
    original_lbls = Path(f"{ORIGINAL_DATASET}/train/truffles")
    
    if original_imgs.exists():
        for img in original_imgs.glob("*.png"):
            shutil.copy(img, f"{COMBINED_DATASET}/images/train/{img.name}")
        
        for lbl in original_lbls.glob("*.txt"):
            shutil.copy(lbl, f"{COMBINED_DATASET}/labels/train/{lbl.name}")
        
        print(f"  Copied original training images")
    
    # Copy video annotations
    print("Copying video annotations...")
    video_imgs = Path(f"{VIDEO_DATASET}/images")
    video_lbls = Path(f"{VIDEO_DATASET}/labels")
    
    if video_imgs.exists():
        for img in video_imgs.glob("*"):
            shutil.copy(img, f"{COMBINED_DATASET}/images/train/{img.name}")
        
        for lbl in video_lbls.glob("*"):
            shutil.copy(lbl, f"{COMBINED_DATASET}/labels/train/{lbl.name}")
        
        print(f"  Copied video annotations")
    
    # Copy validation data
    original_val = Path(f"{ORIGINAL_DATASET}/val/truffles")
    if original_val.exists():
        for img in original_val.glob("*.png"):
            shutil.copy(img, f"{COMBINED_DATASET}/images/val/{img.name}")
        
        for lbl in original_val.glob("*.txt"):
            shutil.copy(lbl, f"{COMBINED_DATASET}/labels/val/{lbl.name}")
        
        print(f"  Copied validation data")
    
    # Create data.yaml
    data_yaml = {
        'path': os.path.abspath(COMBINED_DATASET),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 1,
        'names': ['truffle']
    }
    
    yaml_path = f"{COMBINED_DATASET}/data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f)
    
    print(f"\nCombined dataset created at {COMBINED_DATASET}")
    
    # Count files
    train_imgs = len(list(Path(f"{COMBINED_DATASET}/images/train").glob("*")))
    val_imgs = len(list(Path(f"{COMBINED_DATASET}/images/val").glob("*")))
    print(f"Training images: {train_imgs}")
    print(f"Validation images: {val_imgs}")
    
    return yaml_path

def train_model(data_yaml):
    """Train YOLO model on combined dataset."""
    
    print("\n=== Training YOLO Model ===")
    
    # Load pre-trained model
    model = YOLO("yolov8n.pt")
    
    # Determine device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    # Train
    results = model.train(
        data=data_yaml,
        epochs=30,  # Increased for better learning
        imgsz=640,   # Larger size for better detection
        patience=10, # Early stopping
        device=device,
        name="video_trained"
    )
    
    # Save best model
    model.save("truffle_yolo_model_video.pt")
    print("\nModel saved as truffle_yolo_model_video.pt")
    
    return results

if __name__ == "__main__":
    import sys
    
    # Check if video annotations exist
    if not os.path.exists("dataset/video_dataset"):
        print("Error: Video annotations not found.")
        print("Please run annotate_frames.py first.")
        sys.exit(1)
    
    dataset_count = len(list(Path("dataset/video_dataset/labels").glob("*.txt")))
    if dataset_count == 0:
        print("Error: No annotated frames found in dataset/video_dataset/")
        sys.exit(1)
    
    print(f"Found {dataset_count} annotated video frames")
    
    # Create combined dataset
    yaml_path = create_combined_dataset()
    
    # Train model
    train_model(yaml_path)
    
    print("\n=== Training Complete ===")
    print("Next: Test the model with python video_processor.py")
    print("(Update video_processor.py to use 'truffle_yolo_model_video.pt')")
