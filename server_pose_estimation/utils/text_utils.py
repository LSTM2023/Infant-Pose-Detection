import time
import cv2

previous_time = 0

def calculate_fps():
    global previous_time
    current_time = time.time()

    sec = current_time - previous_time
    previous_time = current_time

    fps = 1. / sec
    fps_string = f"Server FPS : {fps:.01f}"

    return fps_string


def put_text(frame, fps_text, pose_status_text, n_stack, b_stack):
    cv2.putText(frame, fps_text, (0, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
    if pose_status_text == "There is no BBox.":
        cv2.putText(frame, f"no_stack : {n_stack} / 150", (0, 425), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
    else:
        cv2.putText(frame, f"no_stack : {n_stack} / 150", (0, 425), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
    if pose_status_text == "Normal Sleeping Pose":
        cv2.putText(frame, f"bad_stack : {b_stack} / 150", (0, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 230, 255), 2)
        # cv2.putText(frame, "Normal Pose", (0, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 230, 255), 2)
    elif pose_status_text == "Bad Sleeping Pose":
        cv2.putText(frame, f"bad_stack : {b_stack} / 150", (0, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 110, 205), 2)
        # cv2.putText(frame, "Bad Pose", (0, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 110, 205), 2)
    else: # "Danger Sleeping Pose"
        cv2.putText(frame, f"bad_stack : {b_stack} / 150", (0, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 215), 2)
        # cv2.putText(frame, "Danger Pose", (0, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 215), 2) """
        
    return frame