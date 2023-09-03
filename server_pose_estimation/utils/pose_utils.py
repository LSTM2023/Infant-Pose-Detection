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


def get_pose_direction(bbox, kpts):
    center_x, center_y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle = get_kpt_coordinate(kpts)
    
    face = [nose, left_eye, right_eye, left_ear, right_ear]

    if w < h: # vertical
        if all(face_part[1] < center_y for face_part in face): # 머리가 위쪽에 있는 경우
            pose_direction = 'up'
        else: # 머리가 아래쪽에 있는 경우
            pose_direction = 'down'
    else: # horizontal
        if all(face_part[0] < center_x for face_part in face): # 머리가 왼쪽에 있는 경우
            pose_direction = 'left'
        else: # 머리가 오른쪽에 있는 경우
            pose_direction = 'right'

    return pose_direction


def get_pose_status(bbox, kpts):
    pose_direction = get_pose_direction(bbox, kpts)
    
    nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle = get_kpt_coordinate(kpts)
    
    if pose_direction == 'up':
        is_lying_side = (all(wrist > right_shoulder[0] for wrist in (left_wrist[0], right_wrist[0]))) or (all(wrist < left_shoulder[0] for wrist in (left_wrist[0], right_wrist[0]))) # 왼쪽 or 오른쪽으로 누운 자세

        is_lying_back = (left_elbow[0] < left_shoulder[0]) and (right_elbow[0] > right_shoulder[0]) and ((left_shoulder[0] < right_shoulder[0]) and (left_hip[0] < right_hip[0])) # 뒤집혀서 누운 자세
    
        if is_lying_side: # 옆으로 누운 자세
            pose_status = "Bad Sleeping Pose"
        elif is_lying_back: # 완전 뒤집힌 자세
            pose_status = "Dangerous Sleeping Pose"
        else: # 정상 자세
            pose_status = "Normal Sleeping Pose"
    else: # pose_direction is 'down' or 'left' or 'right'
        pose_status = 'Wrong Pose Direction'
        
    return pose_status