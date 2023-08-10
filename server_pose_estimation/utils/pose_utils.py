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


def get_pose_status(kpts, orientation): # TODO: 모든 방향에 Pose Detection Algorithm 적용되게 수정하기
    orientation_dict = {
        'vertical': 0,
        'horizontal': 1
    }
    x_or_y = orientation_dict[orientation]
    
    nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle = get_kpt_coordinate(kpts) # TODO: [x_or_y] 미리 다 적용하기
    
    both_wrist = (left_wrist[x_or_y], right_wrist[x_or_y])
    left_arm = (left_elbow[x_or_y], left_wrist[x_or_y])
    right_arm = (right_elbow[x_or_y], right_wrist[x_or_y])
    
    # TODO: Pose Detection Algorithm 수정
    is_lying_side = (all(wrist > right_shoulder[x_or_y] for wrist in both_wrist)) or (all(wrist < left_shoulder[x_or_y] for wrist in both_wrist)) # 왼쪽 or 오른쪽으로 누운 자세
    is_lying_back = (all(left_arm_joint < left_shoulder[x_or_y] for left_arm_joint in left_arm)) and (all(right_arm_joint > right_shoulder[x_or_y] for right_arm_joint in right_arm)) # 뒤집혀서 누운 자세
        
    if is_lying_side: # 옆으로 누운 자세
        pose_status = "Bad Sleeping Pose"
    elif is_lying_back: # 완전 뒤집힌 자세
        pose_status = "Dangerous Sleeping Pose"
    else: # 정상 자세
        pose_status = "Normal Sleeping Pose"
        
    return pose_status