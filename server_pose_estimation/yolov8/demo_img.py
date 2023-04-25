import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8m-pose.pt')

results = model("baby_1.jpg", conf=0.5, verbose=False)

for result in results:
    result = result.cpu().numpy()
            
    # boxes = result.boxes  # Boxes object for bbox outputs
    # masks = result.masks  # Masks object for segmentation masks outputs
    # probs = result.probs  # Class probabilities for classification outputs
    kpts = result.keypoints

# Visualize the results on the frame
annotated_frame = results[0].plot(labels=False)

cv2.imwrite('test.jpg', annotated_frame)