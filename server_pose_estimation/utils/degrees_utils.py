import numpy as np

def vector_between_points(p1, p2):
    return np.array(p2) - np.array(p1)


def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    
    return dot_product / (v1_norm * v2_norm)


def angle_between_vectors(v1, v2):
    cos_sim = cosine_similarity(v1, v2)
    cos_sim = np.clip(cos_sim, -1.0, 1.0) # arccos 함수는 -1 ~ 1 에만 정상 작동
    angle = np.arccos(cos_sim)  # Cosine Similarity -> Radian
    degrees = round(np.degrees(angle), 1)  # Radian -> Degrees
    
    return degrees


def get_angle(single_kpts):
    point_a = single_kpts[5]
    point_b = single_kpts[7]
    point_c = single_kpts[9]
    
    vector_a = vector_between_points(point_b, point_a)
    vector_b = vector_between_points(point_b, point_c)

    angle = angle_between_vectors(vector_a, vector_b)
    
    return angle