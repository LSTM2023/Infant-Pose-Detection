from ultralytics.yolo.data.converter import convert_coco

convert_coco(labels_dir='./SyRIP_COCO/annotations/200R_1000S/', use_keypoints=True)