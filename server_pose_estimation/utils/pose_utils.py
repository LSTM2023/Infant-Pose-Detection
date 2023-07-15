def get_pose_info(boxes, kpts):
    resolution_w = boxes.orig_shape[1] # Frame width
    resolution_h = boxes.orig_shape[0] # Frame height
    single_box = boxes.xywh[0] # (Center x, Center y, w, h)
    single_box_n = boxes.xywhn[0] # Normalized (Center x, Center y, w, h)
    
    if single_box[2] < single_box[3]: # w < h
        aspect_ratio = 0
    else: # w > h
        aspect_ratio = 1
        
    single_kpts = kpts[0] # (17, 3) : (17 kpts, (x, y, conf))
    
    return single_kpts, aspect_ratio


def detect_pose(kpts, aspect):   
    no_stack = max(0, no_stack - 1)
            
    hand_kpts = (kpts[9][aspect], kpts[10][aspect])
    condition1 = all(val > kpts[6][aspect] for val in hand_kpts)
    condition2 = all(val < kpts[5][aspect] for val in hand_kpts)
    
    if (kpts[5][aspect] < kpts[6][aspect]) and (kpts[11][aspect] < kpts[12][aspect]): # 완전 뒤집힌 자세
        bad_stack += 1
        pose_string = "Dangerous Sleeping Pose"
    elif condition1 or condition2: # 옆으로 누운 자세
        bad_stack += 1
        pose_string = "Bad Sleeping Pose"
    else: # 정상 자세
        bad_stack = max(0, bad_stack - 1)
        pose_string = "Normal Sleeping Pose"
        
    return bad_stack, pose_string