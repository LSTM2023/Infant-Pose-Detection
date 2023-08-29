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
    width, height = frame.shape[1], frame.shape[0]
    default_width, default_height = 480, 640
    
    expand_ratio = ((width / default_width) + (height / default_height)) / 2
    font_size = 0.8 * expand_ratio
    font_thickness = int(2 * expand_ratio)
    
    cv2.putText(frame, fps_string, (0, int(height*0.05)), 3, font_size, (255, 255, 255), font_thickness)
        
    if pose_status_string == "There is no BBox.":
        cv2.putText(frame, f"no_stack : {n_stack} / {stack_th}", (0, int(height*0.93)), 3, font_size, (82, 82, 82), font_thickness)
    else:
        cv2.putText(frame, f"no_stack : {n_stack} / {stack_th}", (0, int(height*0.93)), 3, font_size, (173, 173, 173), font_thickness)
        
    if pose_status_string == "Bad Sleeping Pose":
        cv2.putText(frame, f"bad_stack : {b_stack} / {stack_th}", (0, int(height*0.98)), 3, font_size, (7, 151, 247), font_thickness)
    elif pose_status_string == "Danger Sleeping Pose":
        cv2.putText(frame, f"bad_stack : {b_stack} / {stack_th}", (0, int(height*0.98)), 3, font_size, (5, 8, 227), font_thickness)
    else: # pose_status_string == "Normal Sleeping Pose":
        cv2.putText(frame, f"bad_stack : {b_stack} / {stack_th}", (0, int(height*0.98)), 3, font_size, (2, 220, 245), font_thickness)
        
    return frame