import sys

import cv2

from ultralytics import YOLO

from utils.pose_utils import determine_pose_orientation, get_pose_status
from utils.text_utils import calculate_fps, put_text
from notification import push_notification_for_abnormal_status

model = YOLO('yolov8m-pose.pt')
# model = YOLO('./runs/pose/train_m_16_640/weights/best.pt')

# Open the input video file
video_path ="./dataset/test/real_baby_1.mp4"
# video_path = "http://203.249.22.164:5000/video_feed"

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
resize_resolution = (480, 640)

# fourcc = cv2.VideoWriter_fourcc(*'DIVX')
# out = cv2.VideoWriter("save_video.mp4", fourcc, fps, resize_resolution)

# if not out.isOpened():
#     cap.release()
#     sys.exit()

no_stack, bad_stack = 0, 0 # Stack
stack_th = 150 # Stack Threshold

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        frame = cv2.resize(frame, resize_resolution) # Predict 전 resize
        results = model(frame, stream=True, conf=0.6, verbose=False) # YOLOv8 inference on the frame
        fps_str = calculate_fps()
        
        for result in results:
            result = result.cpu().numpy()
            
            result_boxes = result.boxes  # BBoxes for outputs
            result_kpts = result.keypoints # Keypoints for outputs
        
        try: # 예외 처리 부분
            first_box = result_boxes.xywh[0] # (Center x, Center y, w, h)
            first_kpts = result_kpts.data[0] # (17, 3) : (17 kpts, (x, y, conf))
            
        except Exception as e: # 예외 발생 o
            no_stack += 1 # no_stack 1 증가
            pose_status = "There is no BBox."
            
        else: # 예외 발생 x
            no_stack = max(0, no_stack - 1) # no_stack 1 감소 (최소: 0)

            pose_orientation = determine_pose_orientation(first_box)
            pose_status = get_pose_status(first_kpts, pose_orientation)
            
            warning_type = ['Bad', 'Dangerous']
            if any(warning in pose_status for warning in warning_type): # 'Bad' or 'Dangerous' in pose_status
                bad_stack += 1 # bad_stack 1 증가
            else: # 'Normal' in pose_status
                bad_stack = max(0, bad_stack - 1) # bad_stack 1 감소 (최소: 0)
                
        finally: # stack이 stack_th에 도달하면 stack을 초기화하고 사용자에게 알림 전송
            no_stack, bad_stack = push_notification_for_abnormal_status(no_stack, bad_stack, stack_th)
                
        # Visualize the results and put text on the frame -> annotated_frame
        annotated_frame = result.plot(labels=False)
        # annotated_frame = cv2.resize(annotated_frame, resize_resolution) # Predict 후 resize
        annotated_frame = put_text(annotated_frame, fps_str, pose_status, no_stack, bad_stack, stack_th)

        # Display the annotated frame
        cv2.imshow("Infant Pose Detection", annotated_frame)
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