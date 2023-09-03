import sys

import cv2

from ultralytics import YOLO

from utils.pose_utils import get_pose_status
from utils.text_utils import calculate_fps, put_text
from utils.notification import send_notification

# model = YOLO('yolov8m-pose.pt') # Base Model
model = YOLO('./runs/pose/train_m_16_640/weights/best.pt') # Fine-Tuned Model
# model = YOLO('./runs/pose/train_l_16_640/weights/best.pt') # Fine-Tuned Model

# Open the input video file
# video_path = "http://203.249.22.164:5001/video_feed" # Flask Streaming Server
# video_path = "rtsp://210.99.70.120:1935/live/cctv001.stream" # RSTP Sample
# video_path = "rtsp://203.249.22.164:8080/unicast" # v4l2 RTSP Server
video_path = "./dataset/test/real_baby_1.mp4" # Test

cap = cv2.VideoCapture(video_path)
# cap.set(cv2.CAP_PROP_BUFFERSIZE, 30)
# print(cap.get(cv2.CAP_PROP_BUFFERSIZE))
    
# fourcc = cv2.VideoWriter_fourcc(*'DIVX')
# fps = cap.get(cv2.CAP_PROP_FPS)
# out = cv2.VideoWriter("save_video.mp4", fourcc, fps, resize_resolution)

# if not out.isOpened():
#     cap.release()
#     sys.exit()

resize_ratio = 0.4
wrong_stack, danger_stack = 0, 0 # Stack
stack_th = 200 # Stack Threshold

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        fps_str = calculate_fps()
        
        frame = cv2.resize(frame, (int(frame.shape[1]*resize_ratio), int(frame.shape[0]*resize_ratio)))
        results = model(frame, stream=True, conf=0.7, verbose=False) # YOLOv8 inference on the frame
        
        for result in results:
            result = result.cpu().numpy()
            
            result_boxes = result.boxes  # BBoxes for outputs
            result_kpts = result.keypoints # Keypoints for outputs
        
        try: # 예외 처리 부분
            first_box = result_boxes.xywh[0] # first_box : (4) : (Center x, Center y, w, h)
            first_kpts = result_kpts.data[0] # first_kpts : (17, 3) : (17 kpts, (x, y, conf))
            
        except Exception as e: # 예외 발생 o
            pose_status = "No Infant Detected."
            wrong_stack += 1 # wrong_stack 1 증가
            
        else: # 예외 발생 x
            pose_status = get_pose_status(first_box, first_kpts)
            
            if 'Wrong' in pose_status: # 'Wrong' in pose_status
                wrong_stack += 1 # wrong_stack 1 증가
            else: # 'Bad' or 'Dangerous' or 'Normal' in pose_status
                wrong_stack = max(0, wrong_stack - 1) # wrong_stack 1 감소 (최소: 0)
                
                warning_type = ['Bad', 'Dangerous']
                if any(warning in pose_status for warning in warning_type): # 'Bad' or 'Dangerous' in pose_status
                    danger_stack += 1 # danger_stack 1 증가
                else: # 'Normal' in pose_status
                    danger_stack = max(0, danger_stack - 1) # danger_stack 1 감소 (최소: 0)
                
        finally: # stack이 stack_th에 도달하면 stack을 초기화하고 사용자에게 알림 전송
            if wrong_stack == stack_th:
                wrong_stack = 0 # w_stack 초기화
                send_notification("아이 미탐지", "아이의 수면 자세가 탐지되지 않습니다. 확인해주세요!")
                
            if danger_stack == stack_th:
                danger_stack = 0 # d_stack 초기화
                send_notification("비정상 수면 자세", "아이의 수면 자세가 위험할 수 있으니, 확인해주세요!")
                
        # Visualize the results and put text on the frame -> annotated_frame
        annotated_frame = result.plot(labels=False)
        # annotated_frame = cv2.resize(annotated_frame, (int(frame.shape[1]*resize_ratio), int(frame.shape[0]*resize_ratio)))
        annotated_frame = put_text(annotated_frame, fps_str, pose_status, wrong_stack, danger_stack, stack_th)

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