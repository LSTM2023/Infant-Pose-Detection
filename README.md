# Infant Pose Detection with Raspberry Pi Streaming System

## Introduction
* Predicts the baby's posture through Deep Learning - Pose Estimation technology and performs algorithm-based inappropriate posture detection based on the predicted posture.

* When inappropriate posture is detected, a notification is sent to the user's app.

* This project used Raspberry Pi and Camera Module as IoT means for video.

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
3. Make directory.
```shell
mkdir firebase_cloud_messaging
```
4. Place the json file related to the FCM service account private key in the "firebase_cloud_messaging" directory for sending notifications .
```
${Repo Root}
├── firebase_cloud_messaging
│   └── fcm_service_account_key.json
├── raspberry_pi
└── server_pose_estimation
```
5. Create a file fcm.json, write it in the format shown below, and place it in the same directory, "firebase_cloud_messaging".
```JSON
{
    "private_service_key": "fcm_service_account_key.json",
    "emul_TOKEN": "TOKEN of Emulator",
    "phone_TOKEN": "TOKEN of Phone Device",
    "tablet_TOKEN": "TOKEN of Tablet Device"
}
```
```
${Repo Root}
├── firebase_cloud_messaging
│   ├── fcm.json
│   └── fcm_service_account_key.json
├── raspberry_pi
└── server_pose_estimation
```
6. Just run inference.py in the server_pose_estimation directory.
```shell
cd server_pose_estimation

python3 inference.py
```

## Directory Structure
```shell
${Repo Root}
├── firebase_cloud_messaging
│   ├── fcm.json
│   └── fcm_service_account_key.json
├── raspberry_pi
│   ├── humid_temp_sensor
│   ├── v4l2rtspserver
├── server_pose_estimation
│   ├── fine_tuning
│   ├── utils
│   ├── img_demo.py
│   └── inference.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## How it works?

## Train
#### Data Preparation

## Raspberry Pi

## Documentation
노션?

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
