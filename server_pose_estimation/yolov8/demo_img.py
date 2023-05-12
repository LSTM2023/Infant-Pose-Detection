import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8x-pose.pt')

results = model("baby.png", conf=0.5, verbose=False)

for result in results:
    result = result.cpu().numpy()
            
    # probs = result.probs  # Class probabilities for classification outputs
    boxes = result.boxes  # Boxes object for bbox outputs
    # masks = result.masks  # Masks object for segmentation masks outputs
    kpts = result.keypoints

# Visualize the results on the frame
annotated_frame = result.plot(labels=False)

cv2.imwrite('test.jpg', annotated_frame)