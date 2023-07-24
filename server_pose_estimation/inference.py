import sys
import time
import cv2
from ultralytics import YOLO

from utils.pose_utils import determine_pose_orientation, get_pose_status
from utils.text_utils import calculate_fps, put_text
from notification import push_notification

model = YOLO('yolov8m-pose.pt')
# model = YOLO('./runs/pose/train_m_16_640/weights/best.pt')

# Open the input video file
video_path ="./dataset/test/real_baby_1.mp4"
# video_path = "http://203.249.22.164:5000/video_feed"

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
resize_resolution = (640, 480)

# fourcc = cv2.VideoWriter_fourcc(*'DIVX')
# out = cv2.VideoWriter("save_video.mp4", fourcc, fps, resize_resolution)

# if not out.isOpened():
#     cap.release()
#     sys.exit()

no_stack, bad_stack = 0, 0

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # frame = cv2.resize(frame, resize_resolution) # TODO: Predict 후 resize 되게 변경하기. (?)
        results = model(frame, stream=True, conf=0.6, verbose=False) # YOLOv8 inference on the frame
        fps_string = calculate_fps()
        
        for result in results:
            result = result.cpu().numpy()
            
            result_boxes = result.boxes  # BBoxes for outputs
            result_kpts = result.keypoints # Keypoints for outputs
        
        try: # 예외 처리 부분
            first_box = result_boxes.xywh[0] # (Center x, Center y, w, h)
            first_kpts = result_kpts.data[0] # (17, 3) : (17 kpts, (x, y, conf))
            
        except Exception as e: # 예외 발생 o
            no_stack += 1 # no_stack 1 증가
            pose_status_string = "There is no BBox."
            
        else: # 예외 발생 x
            no_stack = max(0, no_stack - 1) # no_stack 1 감소 (최소: 0)

            pose_orientation = determine_pose_orientation(first_box)
            pose_status_string = get_pose_status(first_kpts, pose_orientation)
            
            alert_strings = ['Bad', 'Dangerous']
            if any(alert in pose_status_string for alert in alert_strings): # 'Bad' or 'Dangerous' in pose_status_string
                bad_stack += 1 # bac_stack 1 증가
            else: # 'Normal' in pose_status_string
                bad_stack = max(0, bad_stack - 1) # bad_stack 1 감소 (최소: 0)
                
        finally:
            if no_stack == 150:
                no_stack = 0 # no_stack 초기화
                # push_notification("아이 미탐지", "아이의 수면 자세가 탐지되지 않습니다. 확인해주세요!")
            if bad_stack == 150:
                bad_stack = 0 # bad_stack 초기화
                # push_notification("비정상 수면 자세", "아이의 수면 자세가 위험할 수 있으니, 확인해주세요!")
                
        # Visualize the results and put text on the frame -> annotated_frame
        annotated_frame = result.plot(labels=False) # TODO: 눈 어떻게 잡는거지?
        annotated_frame = cv2.resize(annotated_frame, resize_resolution) # TODO: Predict 후 resize 되게 변경하기. (?)
        annotated_frame = put_text(annotated_frame, fps_string, pose_status_string, no_stack, bad_stack)

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