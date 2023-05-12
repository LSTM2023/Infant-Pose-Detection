import cv2
import time
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8x-pose.pt')

# Open the video file
# video_path = "./rgb/syn_%5d.png"
# video_path = "baby_video_1.mp4"
video_path = "http://203.249.22.164:5000/video_feed"
cap = cv2.VideoCapture(video_path)

# cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)

# Loop through the video frames
prevTime = 0
th = 0
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame, stream=True, conf=0.5, verbose=False)
        
        for result in results:
            result = result.cpu().numpy()
            
            # probs = result.probs  # Class probabilities for classification outputs
            boxes = result.boxes  # Boxes object for bbox outputs
            # masks = result.masks  # Masks object for segmentation masks outputs
            kpts = result.keypoints
        
        try: # 예외 처리 부분
            w = boxes.orig_shape[1]
            h = boxes.orig_shape[0]    
            single_box = boxes.xywh[0] # (Center x, Center y, w, h)
            single_box_n = boxes.xywhn[0] # Normalized (Center x, Center y, w, h)
            
            single_kpts = kpts[0]
        except Exception as e: # 예외 발생 o
            print('Error')
        else: # 예외 발생 x
            if single_kpts[5][0] < single_kpts[6][0]:
                th = th + 1
        finally: # 반드시 실행
            if th >= 100:
                print('single_kpts[5][0] < single_kpts[6][0]')
                th = 0
            
        # Visualize the results on the frame
        annotated_frame = result.plot(labels=False)
        
        # Calculate FPS
        curTime = time.time()
        sec = curTime - prevTime
        prevTime = curTime
        fps = 1. / sec
        str = "Server FPS : %0.01f" % fps
        cv2.putText(annotated_frame, str, (0, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()