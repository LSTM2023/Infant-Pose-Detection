import os
import json

import firebase_admin
from firebase_admin import credentials
from firebase_admin import messaging


def send_notification(title, body):
    fcm_path = '/home/psw/LSTM/Infant-Pose-Detection/firebase_cloud_messaging/'
    
    with open(os.path.join(fcm_path, 'fcm.json'), 'r') as f:
        fcm_json = json.load(f)
        
    if not firebase_admin._apps:
        cred = credentials.Certificate(os.path.join(fcm_path, fcm_json['firebase_key_path']))
        firebase_admin.initialize_app(cred)

    NOTIFICATION = messaging.Notification(title=title, body=body)
    
    TOKENS = [fcm_json['jh_TOKEN'], fcm_json['sh_TOKEN']]
    
    message = messaging.MulticastMessage(notification=NOTIFICATION, tokens=TOKENS)
    messaging.send_multicast(message)
    print("Successfully Send Notification.")
    
    
if __name__ == '__main__':
    send_notification("For Test", "지금은 Testing")