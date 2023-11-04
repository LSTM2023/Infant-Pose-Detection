# Infant Pose Detection with Raspberry Pi Streaming System

## Introduction
* Predicts the baby's posture through Deep Learning - Pose Estimation technology and performs algorithm-based inappropriate posture detection based on the predicted posture.

* When inappropriate posture is detected, a notification is sent to the user's app.

* In this project, we used Raspberry Pi, camera module, v4l2rtspserver library as IoT means for video.

## Environment
The code is developed using python 3.10 on Ubuntu 22.04. NVIDIA GPUs are needed. The code is developed and tested using two NVIDIA RTX 2080 SUPER GPU card. Other platforms or GPU cards are not fully tested.

## Get Started
1. Clone this repository.
```shell
git clone https://github.com/LSTM2023/Infant-Pose-Detection
```

2. Set up a virtual environment such as venv or conda and install the requirements required.
```shell
python3 -m pip install -r requirements.txt
```

3. Make directory named firebase_cloud_messaging in '${Repo Root}/server_pose_estimation/firebase_cloud_messaging'.
```shell
mkdir firebase_cloud_messaging
```

4. Place the json file related to the FCM service account private key in the ${Repo Root}/server_pose_estimation/firebase_cloud_messaging for sending notifications .
```shell
${Repo Root}
├── raspberry_pi
└── server_pose_estimation
    └── firebase_cloud_messaging
        └── fcm_service_account_key.json
```

5. Create a file fcm.json, write it in the format shown below, and place it in the same directory, ${Repo Root}/server_pose_estimation/firebase_cloud_messaging.
```JSON
{
    "private_service_key": "fcm_service_account_key.json",
    "emul_TOKEN": "TOKEN of Emulator",
    "phone_TOKEN": "TOKEN of Phone Device",
    "tablet_TOKEN": "TOKEN of Tablet Device"
}
```
```shell
${Repo Root}
├── raspberry_pi
└── server_pose_estimation
    └── firebase_cloud_messaging
        ├── fcm_service_account_key.json
        └── fcm.json
```

6. Just run inference.py in the ${Repo Root}/server_pose_estimation directory.
```
python3 inference.py
```

## Model Fine-Tuning
### Data preparation
1. Please download SyRIP dataset from here. Download and extract them under ${Repo_ROOT}/server_pose_estimation/fine_tuning/dataset with the name of 'SyRIP_COCO', and make them look like this:
```shell
${Repo Root}
├── raspberry_pi
└── server_pose_estimation
    └── fine_tuning
        └── dataset
            └── SyRIP_COCO
                ├── annotations
                ├── images
                └── README.md
```

2. Convert the dataset from coco format annotations to yolo format labels using coco_to_yolo.py in ${Repo_ROOT}/server_pose_estimation/fine_tuning/dataset. You can specify the dataset to be converted by changing the path name specified in the code. Then, you can check the .txt format files containing the labels for each image in a directory named yolo_labels.
```shell
python3 coco_to_yolo.py
```

3. Create a SyRIP_YOLO directory in ${Repo_ROOT}/server_pose_estimation/fine_tuning/dataset and place the images and labels to be used for train and validation in it as follows.
```shell
${Repo Root}
├── firebase_cloud_messaging
├── raspberry_pi
└── server_pose_estimation
    └── fine_tuning
        └── dataset
            ├── SyRIP_COCO
            └── SyRIP_YOLO
                ├── ${for_train}
                │   ├── images
                │   │   ├── train00001.jpg
                │   │   ├── train00002.jpg
                │   │   ├── ...
                │   │   └── train10999.jpg
                │   └── labels
                │       ├── train00001.txt
                │       ├── train00002.txt
                │       ├── ...
                │       └── train10999.txt
                └── ${for_valid}
                    ├── images
                    │   ├── test0.jpg
                    │   ├── test1.jpg
                    │   ├── ...
                    │   └── test499.jpg
                    └── labels
                        ├── test0.txt
                        ├── test1.txt
                        ├── ...
                        └── test499.txt
```

4. After that, modify the path in SyRIP-pose.yaml to match \${for_train} and \${for_valid}.
```yaml
train: ${for_train}  # train images
val: ${for_valid}  # val images
```

### Train and validation (Results)
1. Just run train.py in the ${Repo Root}/server_pose_estimation/finetuning directory. You can edit You can edit You can edit You can edit You can edit You can edit 
```
python3 train.py
```

2. And you can check the performance of each model through valid.py.
```
python3 valid.py
```

우리는 다르게 사용한거







## Directory Structure
```shell
${Repo Root}
├── raspberry_pi
│   ├── humid_temp_sensor
│   └── v4l2rtspserver
├── server_pose_estimation
│   ├── fine_tuning
│   │    ├── dataset
│   │    │   └── ...
│   │    ├── coco_to_yolo.py
│   │    ├── train.py
│   │    ├── valid.py
│   │    └── SyRIP-pose.yaml
│   ├── firebase_cloud_messaging
│   │   ├── fcm.json
│   │   └── fcm_service_account_key.json
│   ├── utils
│   │    ├── degrees_utils.py
│   │    ├── notification.py
│   │    ├── pose_utils.py
│   │    └── text_utils.py
│   ├── img_demo.py
│   └── inference.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Raspberry Pi

## How it works?

## License
Infant-Pose-Detection is released under the [AGPL-3.0 License](https://github.com/LSTM2023/Infant-Pose-Detection/blob/main/LICENSE).

## Citation
```bibtex
@inproceedings{huang2021infant,
  title={Invariant Representation Learning for Infant Pose Estimation with Small Data},
  author={Huang, Xiaofei and Fu, Nihang and Liu, Shuangjun and Ostadabbas, Sarah},
  booktitle={IEEE International Conference on Automatic Face and Gesture Recognition (FG), 2021},
  month     = {December},
  year      = {2021}
}
```
```bibtex
@software{Jocher_YOLO_by_Ultralytics_2023,
  author = {Jocher, Glenn and Chaurasia, Ayush and Qiu, Jing},
  license = {AGPL-3.0},
  month = jan,
  title = {{YOLO by Ultralytics}},
  url = {https://github.com/ultralytics/ultralytics},
  version = {8.0.0},
  year = {2023}
}
```