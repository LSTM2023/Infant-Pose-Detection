def determine_pose_orientation(box):
    if box[2] < box[3]: # w < h
        pose_orientation = 'vertical'
    else: # w >= h
        pose_orientation = 'horizontal'
        
    return pose_orientation


def get_kpt_coordinate(kpts):    
    nose = kpts[0]
    left_eye = kpts[1]
    right_eye = kpts[2]    
    left_ear = kpts[3]    
    right_ear = kpts[4]    
    left_shoulder = kpts[5]    
    right_shoulder = kpts[6]    
    left_elbow = kpts[7]    
    right_elbow = kpts[8]    
    left_wrist = kpts[9]    
    right_wrist = kpts[10]    
    left_hip = kpts[11]    
    right_hip = kpts[12]    
    left_knee = kpts[13]    
    right_knee = kpts[14]    
    left_ankle = kpts[15]    
    right_ankle = kpts[16]
    
    return nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle


def get_pose_status(kpts, orientation):  
    orientation_dict = {
        'vertical': 0,
        'horizontal': 1
    }
    
    nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle = get_kpt_coordinate(kpts)
    
    x_or_y = orientation_dict[orientation]
    
    wrist_kpts = (left_wrist[x_or_y], right_wrist[x_or_y])
    condition_1 = all(val > right_shoulder[x_or_y] for val in wrist_kpts)
    condition_2 = all(val < left_shoulder[x_or_y] for val in wrist_kpts)
    
    if (left_shoulder[x_or_y] < right_shoulder[x_or_y]) and (left_hip[x_or_y] < right_hip[x_or_y]): # 완전 뒤집힌 자세
        pose_status = "Dangerous Sleeping Pose"
    elif condition_1 or condition_2: # 옆으로 누운 자세
        pose_status = "Bad Sleeping Pose"
    else: # 정상 자세
        pose_status = "Normal Sleeping Pose"
        
    return pose_status