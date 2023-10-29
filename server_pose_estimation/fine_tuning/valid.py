from ultralytics import YOLO

# Load a model
model = YOLO('yolov8m-pose.pt')  # load an official model
# model = YOLO('yolov8l-pose.pt')  # load an official model
# model = YOLO('runs/pose/train/weights/best.pt')  # load a custom model

# Validate the model
metrics = model.val(data='SyRIP-pose.yaml', device=[0, 1])  # no arguments needed, dataset and settings remembered