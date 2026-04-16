from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n-cls.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="dataset", epochs=10, imgsz=224)

# Save the model
model.save("truffle_yolo_model.pt")