def get_kpt_coordinate(kpts):    
    nose = kpts[0]
    left_eye, right_eye = kpts[1], kpts[2]
    left_ear, right_ear = kpts[3], kpts[4]
    left_shoulder, right_shoulder = kpts[5], kpts[6]
    left_elbow, right_elbow = kpts[7], kpts[8]
    left_wrist, right_wrist = kpts[9], kpts[10]
    left_hip, right_hip = kpts[11], kpts[12]
    left_knee, right_knee = kpts[13], kpts[14]
    left_ankle, right_ankle = kpts[15], kpts[16]
    
    return nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle


def get_pose_direction(kpts): # TODO: 수정 BBox와 kpts 좌표를 기준으로 up, down, right, left 판단
    nose, _, _, _, _, left_shoulder, right_shoulder, _, _, _, _, left_hip, right_hip, _, _, _, _ = get_kpt_coordinate(kpts)

    if abs(left_shoulder[0] - right_shoulder[0]) > abs(left_shoulder[1] - right_shoulder[1]): # vertical
        if nose[1] <= (left_shoulder[1] + right_shoulder[1]) / 2:  # 머리가 위쪽에 있는 경우
            pose_direction = 'up'
        else:  # 머리가 아래쪽에 있는 경우
            pose_direction = 'down'
    else: # horizontal
        if nose[0] < (left_hip[0] + right_hip[0]) / 2: # 머리가 왼쪽에 있는 경우
            pose_direction = 'left'
        else: # 머리가 오른쪽에 있는 경우
            pose_direction = 'right'

    return pose_direction


def get_pose_status(kpts):
    pose_direction = get_pose_direction(kpts)
    
    nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle = get_kpt_coordinate(kpts)
    
    direction_dict = {
        'up': 0,
        'down': 0,
        'left': 1,
        'right': 1
    }
    
    is_x_or_y = direction_dict[pose_direction]
        
    # if pose_direction == 'up': # TODO: up, down, left, right에 따라 Algorithm 수정
    is_lying_side = (all(wrist > right_shoulder[is_x_or_y] for wrist in (left_wrist[is_x_or_y], right_wrist[is_x_or_y]))) or (all(wrist < left_shoulder[is_x_or_y] for wrist in (left_wrist[is_x_or_y], right_wrist[is_x_or_y]))) # 왼쪽 or 오른쪽으로 누운 자세

    is_lying_back = (left_elbow[is_x_or_y] < left_shoulder[is_x_or_y]) and (right_elbow[is_x_or_y] > right_shoulder[is_x_or_y]) and ((left_shoulder[is_x_or_y] < right_shoulder[is_x_or_y]) and (left_hip[is_x_or_y] < right_hip[is_x_or_y])) # 뒤집혀서 누운 자세
    
    if is_lying_side: # 옆으로 누운 자세
        pose_status = "Bad Sleeping Pose"
    elif is_lying_back: # 완전 뒤집힌 자세
        pose_status = "Dangerous Sleeping Pose"
    else: # 정상 자세
        pose_status = "Normal Sleeping Pose"
        
    return pose_status