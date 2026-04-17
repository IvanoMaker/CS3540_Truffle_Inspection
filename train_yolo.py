from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="data.yaml", epochs=10, imgsz=224)

# Save the model
model.save("truffle_yolo_model.pt")