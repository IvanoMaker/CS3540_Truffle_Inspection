import cv2
import os
from pathlib import Path
import json

# Configuration
FRAMES_DIR = "dataset/video_frames"
ANNOTATIONS_DIR = "dataset/video_annotations"
OUTPUT_DATASET_DIR = "dataset/video_dataset"

# Create output directories
Path(ANNOTATIONS_DIR).mkdir(parents=True, exist_ok=True)
Path(f"{OUTPUT_DATASET_DIR}/images").mkdir(parents=True, exist_ok=True)
Path(f"{OUTPUT_DATASET_DIR}/labels").mkdir(parents=True, exist_ok=True)

class FrameAnnotator:
    def __init__(self, frames_dir):
        self.frames_dir = frames_dir
        self.frames = sorted([f for f in os.listdir(frames_dir) if f.endswith(('.jpg', '.png'))])
        self.current_idx = 0
        self.boxes = []  # List of (x1, y1, x2, y2) in pixel coordinates
        self.drawing = False
        self.start_point = None
        self.image = None
        self.original_image = None
        
    def load_frame(self, idx):
        if 0 <= idx < len(self.frames):
            self.current_idx = idx
            path = os.path.join(self.frames_dir, self.frames[idx])
            self.original_image = cv2.imread(path)
            self.image = self.original_image.copy()
            self.boxes = []  # Clear boxes when loading new frame
            self.load_existing_annotations()
            return True
        return False
    
    def load_existing_annotations(self):
        """Load existing annotations if they exist."""
        annotation_path = os.path.join(ANNOTATIONS_DIR, f"{Path(self.frames[self.current_idx]).stem}.json")
        if os.path.exists(annotation_path):
            with open(annotation_path) as f:
                data = json.load(f)
                self.boxes = [tuple(box) for box in data['boxes']]
                print(f"Loaded {len(self.boxes)} existing boxes")
    
    def save_annotations(self):
        """Save annotations to JSON and YOLO format."""
        frame_name = Path(self.frames[self.current_idx]).stem
        
        # Save to JSON
        json_path = os.path.join(ANNOTATIONS_DIR, f"{frame_name}.json")
        with open(json_path, 'w') as f:
            json.dump({'boxes': self.boxes, 'frame': self.frames[self.current_idx]}, f)
        
        # Save to YOLO format (normalized coordinates)
        h, w = self.original_image.shape[:2]
        yolo_annotations = []
        for x1, y1, x2, y2 in self.boxes:
            # Convert to center coordinates and normalize
            cx = (x1 + x2) / 2 / w
            cy = (y1 + y2) / 2 / h
            box_w = (x2 - x1) / w
            box_h = (y2 - y1) / h
            # Class 0 is truffle
            yolo_annotations.append(f"0 {cx:.6f} {cy:.6f} {box_w:.6f} {box_h:.6f}")
        
        yolo_path = os.path.join(ANNOTATIONS_DIR, f"{frame_name}.txt")
        with open(yolo_path, 'w') as f:
            f.write('\n'.join(yolo_annotations))
        
        print(f"Saved {len(self.boxes)} annotations: {json_path}")
    
    def draw_boxes(self):
        """Redraw all boxes on current image."""
        self.image = self.original_image.copy()
        for x1, y1, x2, y2 in self.boxes:
            cv2.rectangle(self.image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Show box size
            cv2.putText(self.image, f"{x2-x1}x{y2-y1}", (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
        
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing and self.start_point:
                # Ensure coordinates are in correct order
                x1, x2 = min(self.start_point[0], x), max(self.start_point[0], x)
                y1, y2 = min(self.start_point[1], y), max(self.start_point[1], y)
                
                # Only add if box has reasonable size
                if (x2 - x1) > 20 and (y2 - y1) > 20:
                    self.boxes.append((x1, y1, x2, y2))
                    print(f"Added box: ({x1}, {y1}, {x2}, {y2})")
            self.drawing = False
            self.start_point = None
    
    def display_frame(self):
        """Display current frame with instructions."""
        self.draw_boxes()
        
        # Add instruction text
        h, w = self.image.shape[:2]
        info = f"Frame {self.current_idx+1}/{len(self.frames)} - {len(self.boxes)} boxes"
        cv2.putText(self.image, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(self.image, "Click & drag to draw | 'S'=Save | 'D'=Delete last | 'N'=Next | 'P'=Prev | 'Q'=Quit",
                   (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow("Annotation Tool", self.image)
    
    def delete_last_box(self):
        if self.boxes:
            self.boxes.pop()
            print("Deleted last box")
    
    def run(self):
        print("\n=== Frame Annotation Tool ===")
        print("Instructions:")
        print("  Click and drag to draw bounding boxes around truffles")
        print("  'S' - Save annotations for current frame")
        print("  'D' - Delete last box")
        print("  'N' - Next frame")
        print("  'P' - Previous frame")
        print("  'Q' - Quit")
        
        cv2.namedWindow("Annotation Tool")
        cv2.setMouseCallback("Annotation Tool", self.mouse_callback)
        
        if not self.load_frame(0):
            print("No frames found!")
            return
        
        while True:
            self.display_frame()
            
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_annotations()
            elif key == ord('d'):
                self.delete_last_box()
            elif key == ord('n'):
                if self.current_idx < len(self.frames) - 1:
                    self.load_frame(self.current_idx + 1)
            elif key == ord('p'):
                if self.current_idx > 0:
                    self.load_frame(self.current_idx - 1)
        
        cv2.destroyAllWindows()
        
        # Organize annotated data
        self.organize_dataset()
    
    def organize_dataset(self):
        """Copy annotated frames to dataset directory."""
        print("\nOrganizing dataset...")
        annotated_count = 0
        
        for frame_file in self.frames:
            frame_stem = Path(frame_file).stem
            txt_path = os.path.join(ANNOTATIONS_DIR, f"{frame_stem}.txt")
            
            if os.path.exists(txt_path):
                # Check if annotation has boxes
                with open(txt_path) as f:
                    content = f.read().strip()
                    if content:  # Only copy if has annotations
                        src_img = os.path.join(self.frames_dir, frame_file)
                        dst_img = os.path.join(OUTPUT_DATASET_DIR, "images", frame_file)
                        dst_txt = os.path.join(OUTPUT_DATASET_DIR, "labels", f"{frame_stem}.txt")
                        
                        import shutil
                        shutil.copy(src_img, dst_img)
                        shutil.copy(txt_path, dst_txt)
                        annotated_count += 1
        
        print(f"\nOrganized {annotated_count} annotated frames to {OUTPUT_DATASET_DIR}")
        
        # Create data.yaml for dataset
        yaml_content = f"""path: {os.path.abspath(OUTPUT_DATASET_DIR)}
train: images
val: images

nc: 1
names: ['truffle']
"""
        
        yaml_path = os.path.join(OUTPUT_DATASET_DIR, "data.yaml")
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        print(f"Created {yaml_path}")


if __name__ == "__main__":
    if not os.path.exists(FRAMES_DIR):
        print(f"Error: {FRAMES_DIR} not found. Run extract_frames.py first.")
        exit()
    
    annotator = FrameAnnotator(FRAMES_DIR)
    annotator.run()
