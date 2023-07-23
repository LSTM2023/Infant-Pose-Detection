from ultralytics import YOLO

# Load a model
model = YOLO('yolov8m-pose.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8l-pose.pt')  # load a pretrained model (recommended for training)

# Train the model
model.train(data='SyRIP-pose.yaml', batch=16, epochs=1000, imgsz=640, patience=200, device=[0, 1])