import time

import cv2

previous_time = 0

def calculate_fps():
    global previous_time
    current_time = time.time()

    sec = current_time - previous_time
    previous_time = current_time

    fps = 1. / sec
    fps_str = f"Server FPS : {fps:.01f}"

    return fps_str


def put_text(frame, fps_string, pose_status_string, n_stack, b_stack, stack_th):
    cv2.putText(frame, fps_string, (0, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
    if pose_status_string == "There is no BBox.":
        cv2.putText(frame, f"no_stack : {n_stack} / {stack_th}", (0, 425), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
    else:
        cv2.putText(frame, f"no_stack : {n_stack} / {stack_th}", (0, 425), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
    if pose_status_string == "Bad Sleeping Pose":
        cv2.putText(frame, f"bad_stack : {b_stack} / {stack_th}", (0, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 110, 205), 2)
    elif pose_status_string == "Danger Sleeping Pose":
        cv2.putText(frame, f"bad_stack : {b_stack} / {stack_th}", (0, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 215), 2)
    else: # pose_status_string == "Normal Sleeping Pose":
        cv2.putText(frame, f"bad_stack : {b_stack} / {stack_th}", (0, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 230, 255), 2)
        
    return frame