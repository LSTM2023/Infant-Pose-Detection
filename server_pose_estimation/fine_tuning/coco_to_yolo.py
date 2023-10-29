from ultralytics.data.converter import convert_coco

convert_coco(labels_dir='./dataset/SyRIP_COCO/annotations/200R_1000S/', use_keypoints=True)