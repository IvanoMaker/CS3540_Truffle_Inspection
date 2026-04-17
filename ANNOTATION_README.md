# Video Frame Annotation Guide

## Overview
You now have 59 frames extracted from your video that need to be annotated with bounding boxes around truffles. This will fix the domain mismatch issue where your YOLO model was trained only on cropped truffle images.

## Step 1: Annotate Frames

Run the annotation tool:
```bash
python annotate_frames.py
```

### How to Use the Annotation Tool

A window will open showing the first frame. You can:

**Drawing Boxes:**
- **Click and drag** to draw a bounding box around each truffle
- Start from top-left, drag to bottom-right
- The box will be highlighted in green once drawn

**Navigation:**
- **'N'** - Next frame
- **'P'** - Previous frame
- **'S'** - Save annotations for current frame (IMPORTANT!)
- **'D'** - Delete the last box drawn
- **'Q'** - Quit annotation tool

**Important Notes:**
1. **Always press 'S' after drawing boxes** to save annotations
2. Try to annotate at least 30-40 frames to get good training data
3. Focus on frames where truffles are visible
4. Draw boxes tightly around each truffle
5. If a frame has no truffles, press 'S' to save (with 0 boxes)

### Expected Output
The tool will create:
- `dataset/video_annotations/` - JSON and YOLO format annotations
- `dataset/video_dataset/` - Organized annotated frames (only frames with boxes)

## Step 2: Train Model on Combined Data

Once you've annotated enough frames, run:
```bash
python train_yolo_video.py
```

This will:
1. Combine your original dataset with video annotations
2. Create `dataset/combined_dataset/`
3. Train a new YOLO model for 30 epochs
4. Save the model as `truffle_yolo_model_video.pt`

Training time: ~10-30 minutes depending on GPU

## Step 3: Test on Video

Update `video_processor.py` to use the new model:

Change this line:
```python
yolo_model = YOLO("runs/detect/train6/weights/best.pt")
```

To this:
```python
yolo_model = YOLO("truffle_yolo_model_video.pt")
```

Then run:
```bash
python video_processor.py
```

## Tips for Better Annotations

1. **Label all visible truffles** - Even small or partially visible ones
2. **Be consistent** - Similar size boxes for similar sized truffles
3. **Avoid over-cropping** - Include a bit of context around the truffle
4. **Check frames with most truffles first** - These are the most valuable

## Troubleshooting

**Tool crashes or won't start:**
- Make sure `dataset/video_frames/` exists and contains .jpg files
- Check that frame files are readable

**Can't see buttons clearly:**
- The instructions are shown at the bottom of the window
- Use keyboard shortcuts instead of looking for buttons

**Model trains but still no detections:**
- You may need more annotated frames (50+)
- Retrain with more epochs
- Check if annotations are in correct YOLO format

## Quality Check

To verify annotations were saved correctly:
```bash
ls dataset/video_annotations/  # Should see .txt and .json files
ls dataset/video_dataset/       # Should see images/labels directories
```

Each image should have a corresponding .txt file with YOLO format annotations (class_id center_x center_y width height, normalized 0-1).
