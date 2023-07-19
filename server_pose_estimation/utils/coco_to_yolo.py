from ultralytics.data.converter import convert_coco

convert_coco(labels_dir='./dataset/SyRIP_COCO/annotations/200R/', use_keypoints=True)