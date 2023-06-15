import os, sys
import time
import cv2
import numpy as np
from ultralytics import YOLO

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from firebase_flutter_notification.notification import push_notification
from degrees import get_angle

# Load the YOLOv8 model
model = YOLO('yolov8x-pose.pt')

# Open the video file
# video_path = "./rgb/syn_%5d.png"
# video_path ="real_baby_1.mp4"
video_path = "http://203.249.22.164:5000/video_feed"

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
resize_resolution = (360, 480)

# fourcc = cv2.VideoWriter_fourcc(*'DIVX')
# out = cv2.VideoWriter("save_video.mp4", fourcc, fps, resize_resolution)

# if not out.isOpened():
#     cap.release()
#     sys.exit()

prevTime = 0
bad_stack, no_stack = 0, 0

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        # frame = cv2.resize(frame, resize_resolution)
        results = model(frame, stream=True, conf=0.6, verbose=False)
        
        for result in results:
            
            result = result.cpu().numpy()
            
            # probs = result.probs  # Class probabilities for classification outputs
            boxes = result.boxes  # Boxes object for bbox outputs
            # masks = result.masks  # Masks object for segmentation masks outputs
            kpts = result.keypoints
        
        try: # 예외 처리 부분
            resolution_w = boxes.orig_shape[1] # 640
            resolution_h = boxes.orig_shape[0] # 480
            single_box = boxes.xywh[0] # (Center x, Center y, w, h)
            single_box_n = boxes.xywhn[0] # Normalized (Center x, Center y, w, h)
            single_kpts = kpts[0] # (17, 3) : (17 kpts, (x, y, conf))
            
            if single_box[2] < single_box[3]: # 세로
                coordinate = 0
            else: # 가로
                coordinate = 1
                
            get_angle(single_kpts)
            
        except Exception as e: # 예외 발생 o
            no_stack = no_stack + 1
            pose_string = "There is no BBox."
            
        else: # 예외 발생 x
            no_stack = max(0, no_stack - 1)
                    
            hand_kpts = (single_kpts[9][coordinate], single_kpts[10][coordinate])
            condition1 = all(val > single_kpts[6][coordinate] for val in hand_kpts)
            condition2 = all(val < single_kpts[5][coordinate] for val in hand_kpts)
            if (single_kpts[5][coordinate] < single_kpts[6][coordinate]) and (single_kpts[11][coordinate] < single_kpts[12][coordinate]): # 완전 뒤집힌 자세
                bad_stack = bad_stack + 1
                pose_string = "Dangerous Sleeping Pose"
            elif condition1 or condition2: # 옆으로 누운 자세
                bad_stack = bad_stack + 1
                pose_string = "Bad Sleeping Pose"
            else: # 정상 자세
                bad_stack = max(0, bad_stack - 1)
                pose_string = "Normal Sleeping Pose"
                
        finally:
            if no_stack == 150:
                no_stack = 0
                push_notification("아이 미탐지", "아이의 수면 자세가 탐지되지 않습니다. 확인해주세요!")
            if bad_stack == 150:
                bad_stack = 0
                push_notification("비정상 수면 자세", "아이의 수면 자세가 위험할 수 있으니, 확인해주세요!")
                
        # Visualize the results on the frame
        annotated_frame = result.plot(labels=False)
                
        # Calculate FPS
        curTime = time.time()
        sec = curTime - prevTime
        prevTime = curTime
        fps = 1. / sec
        fps_string = "Server FPS : %0.01f" % fps
        # cv2.putText(annotated_frame, fps_string, (0, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        if pose_string == "There is no BBox.":
            cv2.putText(annotated_frame, f"no_stack : {no_stack} / 150", (0, 425), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
        else:
            cv2.putText(annotated_frame, f"no_stack : {no_stack} / 150", (0, 425), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
        if pose_string == "Normal Sleeping Pose":
            cv2.putText(annotated_frame, f"bad_stack : {bad_stack} / 150", (0, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 230, 255), 2)
            # cv2.putText(annotated_frame, "Normal Pose", (0, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 230, 255), 2)
        elif pose_string == "Bad Sleeping Pose":
            cv2.putText(annotated_frame, f"bad_stack : {bad_stack} / 150", (0, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 110, 205), 2)
            # cv2.putText(annotated_frame, "Bad Pose", (0, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 110, 205), 2)
        else: # "Danger Sleeping Pose"
            cv2.putText(annotated_frame, f"bad_stack : {bad_stack} / 150", (0, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 215), 2)
            # cv2.putText(annotated_frame, "Danger Pose", (0, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 215), 2) """

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)
        # out.write(annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
# out.release()
cv2.destroyAllWindows()