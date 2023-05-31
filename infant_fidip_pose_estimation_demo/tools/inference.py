# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import cv2
import numpy as np
import math

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

from ultralytics import YOLO

import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from core.function import validate
from utils.utils import create_logger

import dataset
import models


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()
    return args

CTX = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

COCO_KEYPOINT_INDEXES = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle'
}

# To Run : python tools/inference.py --cfg experiments/coco/hrnet/w48_384x288_adam_lr1e-3.yaml
NUM_KPTS = 17
SKELETON = [[1, 3], [1, 0], [2, 0], [2, 4], [5, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16]] # coco format

CocoColors = [[0, 0, 255], [0, 0, 255], [255, 0, 0], [255, 0 ,0], [0, 0, 0], [0, 0, 255], [0, 0, 255], [255, 0, 0], [255, 0, 0], [0, 0, 255], [255, 0, 0], [0, 0, 0], [0, 0, 255], [0, 0, 255], [255, 0, 0], [255, 0, 0]]

def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)
    # print(idx, maxvals)
    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def get_final_preds(config, batch_heatmaps, center, scale):
    coords, maxvals = get_max_preds(batch_heatmaps)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing
    if config.TEST.POST_PROCESS:
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = batch_heatmaps[n][p]
                px = int(math.floor(coords[n][p][0] + 0.5))
                py = int(math.floor(coords[n][p][1] + 0.5))
                if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                    diff = np.array(
                        [
                            hm[py][px+1] - hm[py][px-1],
                            hm[py+1][px]-hm[py-1][px]
                        ]
                    )
                    coords[n][p] += np.sign(diff) * .25

    preds = coords.copy()

    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(
            coords[i], center[i], scale[i], [heatmap_width, heatmap_height]
        )

    return preds, maxvals

def get_pose_estimation_prediction(pose_model, image, center, scale):
    rotation = 0

    # pose estimation transformation
    trans = get_affine_transform(center, scale, rotation, cfg.MODEL.IMAGE_SIZE)
    # model_input = cv2.resize(image, (256, 256))
    model_input = cv2.warpAffine(
        image,
        trans,
        (int(cfg.MODEL.IMAGE_SIZE[0]), int(cfg.MODEL.IMAGE_SIZE[1])),
        flags=cv2.INTER_LINEAR)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # pose estimation inference
    model_input = model_input/255.
    model_input = model_input - np.array([0.485, 0.456, 0.406])
    model_input = model_input / np.array([0.229, 0.224, 0.225])

    model_input = torch.from_numpy(model_input).permute(2, 0, 1).unsqueeze(0).float()

    # model_input = transform(model_input).unsqueeze(0)
    # switch to evaluate mode
    pose_model.eval()
    with torch.no_grad():
        # compute output heatmap
        model_input = model_input.cuda()
        output = pose_model(model_input)
        # print(output.shape)
        preds, _ = get_final_preds(
            cfg,
            output.clone().cpu().numpy(),
            np.asarray([center]),
            np.asarray([scale]))

        return preds

def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

def transform_preds(coords, center, scale, output_size):
    # print(scale)
    target_coords = np.zeros(coords.shape)

    trans = get_affine_transform(center, scale, 0, output_size, inv=1)

    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)

    return target_coords

def get_affine_transform(
        center, scale, rot, output_size,
        shift=np.array([0, 0], dtype=np.float32), inv=0
):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def calculate_bbox_points(source):
    model = YOLO("models/yolov8m.pt")
    # accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
    # results = model.predict(source="0")
    results = model.predict(source=source, stream=True, show=False) # Display preds. Accepts all YOLO predict arguments

    for result in results:
        # detection
        result.boxes.xyxy   # box with xyxy format, (N, 4)
        result.boxes.xywh   # box with xywh format, (N, 4)
        result.boxes.xyxyn  # box with xyxy format but normalized, (N, 4)
        result.boxes.xywhn  # box with xywh format but normalized, (N, 4)
        result.boxes.conf   # confidence score, (N, 1)
        result.boxes.cls    # cls, (N, 1)
        
    result = result.cpu().numpy()

    for index, bbox_class in enumerate(result.boxes.cls):
        if bbox_class == 0:
            x = result.boxes.xywh[index][0]
            y = result.boxes.xywh[index][1]
            w = result.boxes.xywh[index][2]
            h = result.boxes.xywh[index][3]

    return x, y, w, h

def get_distance(prev, cur, pred_f):
    x = pow(pred_f[prev][0] - pred_f[cur][0], 2)
    y = pow(pred_f[prev][1] - pred_f[cur][1], 2)
    distance = math.sqrt(x + y)
    
    return distance

def draw_pose(keypoints, img):
    """draw the keypoints and the skeletons.
    :params keypoints: the shape should be equal to [17,2]
    :params img:
    """

    assert keypoints.shape == (NUM_KPTS, 2)
    for i in range(len(SKELETON)):
        kpt_a, kpt_b = SKELETON[i][0], SKELETON[i][1]
        x_a, y_a = keypoints[kpt_a][0], keypoints[kpt_a][1]
        x_b, y_b = keypoints[kpt_b][0], keypoints[kpt_b][1] 
        cv2.circle(img, (int(x_a), int(y_a)), 4, CocoColors[i], -1)
        cv2.circle(img, (int(x_b), int(y_b)), 4, CocoColors[i], -1)
        cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), CocoColors[i], 2)

def main(): # To Run : python tools/inference.py --cfg experiments/coco/hrnet/w48_384x288_adam_lr1e-3.yaml
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(cfg, args.cfg, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    # model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')()

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )

    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    else:
        model_state_file = os.path.join(
            final_output_dir, 'model_best.pth'
        )
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))

    model = model.cuda()

    isimage = 0
    
    # image_source = 'test/image/baby_2.jpg'
    image_source = 'test/video/test_video_golf.mp4'
    x, y, w, h = calculate_bbox_points(image_source)
    # print(x, y, w, h)
    
    if isimage == True:
        image = cv2.imread(image_source)
 
        image_pose = image.copy()
        center = np.zeros((2), dtype=np.float32)

        # bbox로 탐지하거나 수동으로 넣어야 하는 값
        center[0] = x # 사람 bbox x 중심
        center[1] = y # 사람 bbox y 중심
        pixel_std = 200 # 고정
        box_height = h * 0.9 # bbox 세로 길이 | 사람 키 픽셀 길이라고 생각하면 편함
        scale = np.array([box_height * 1.0 / pixel_std, box_height * 1.0 / pixel_std], dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25

        pose_preds = get_pose_estimation_prediction(model, image_pose, center, scale)
        for kpt in pose_preds:
            draw_pose(kpt, image) # draw the poses

        cv2.imshow('test_image', image)
        cv2.waitKey(0)
    else:
        cap = cv2.VideoCapture('test/video/test_video_golf.mp4')
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        out = cv2.VideoWriter('test/video/test_video_golf_result.mp4', fourcc, 60.0, (int(width), int(height)))
        while(cap.isOpened()):
            ret, image = cap.read()
            if image is None:
                break

            image_pose = image.copy()
            center = np.zeros((2), dtype=np.float32)
            center[0] = x
            center[1] = y
            pixel_std = 200
            box_height = h * 0.9 # bbox 세로 길이 | 사람 키 픽셀 길이라고 생각하면 편함
            scale = np.array([box_height * 1.0 / pixel_std, box_height * 1.0 / pixel_std], dtype=np.float32)
            if center[0] != -1:
                scale = scale * 1.25
            pose_preds = get_pose_estimation_prediction(model, image_pose, center, scale)
            
            for kpt in pose_preds:
                draw_pose(kpt, image) # draw the poses

            cv2.imshow('test_video', image)
            out.write(image)
            cv2.waitKey(1)
            
        cap.release()
        out.release()

if __name__ == '__main__':
    main()