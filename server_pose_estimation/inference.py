import os, sys
import time
import cv2
from ultralytics import YOLO

from notification import push_notification
from utils.pose_utils import get_pose_info, detect_pose

# Load the YOLOv8 model
model = YOLO('yolov8x-pose.pt')

# Open the video file
video_path ="./baby_source/real_baby_1.mp4"
# video_path = "http://203.249.22.164:5000/video_feed"

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
            result_boxes = result.boxes  # Boxes object for bbox outputs
            # masks = result.masks  # Masks object for segmentation masks outputs
            result_kpts = result.keypoints
        
        try: # 예외 처리 부분
            single_kpts, aspect_ratio = get_pose_info(result_boxes, result_kpts)
            
        except Exception as e: # 예외 발생 o
            no_stack = no_stack + 1
            pose_string = "There is no BBox."
            
        else: # 예외 발생 x
            bad_stack, pose_string = detect_pose(single_kpts, aspect_ratio)
                
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