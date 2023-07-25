import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import messaging

def send_notification(title, body):
    if not firebase_admin._apps:
        cred = credentials.Certificate(os.path.join(os.path.dirname(os.path.realpath(__file__)), "lstm-bob-firebase-adminsdk-gmh89-fac4fac367.json"))
        firebase_admin.initialize_app(cred)

    NOTIFICATION = messaging.Notification(title=title, body=body)
    
    jh_TOKEN = "cAqsSkPgTLmb8uxBJsZ38F:APA91bEyJ3-ugDNuYphBx_jJJv7w72Kh37sV1XKdC5fPKXR7lQxNZqpuI-9N_S3o7FJHGnELbdRwiiHgoJfxFAamvnhpUIxbLA4tf-uKit2kBKRnpOI7lP4FkwN4DLUjmn1p89-D0-Yh"
    cu_TOKEN = "eTP0p7mO_Ex5uibUFRkEuL:APA91bFw60PkaSOzGV4qEVCtxYXL7kAMJ7f0_68mDdN4Cn0hnlDaO7CV7HuGxBTgtEBIZY8Ph1Xm4wv2WZdhv0AVieF53O_WkLtYyZV0HTBCVxN-twc9cOyIURdNd9jlks1mlqccs3Po"
    sh_TOKEN = "c8I_Mw5kQhq5Ub1gQ0Pc7d:APA91bF9XtUXSnubty-W-ltgxElC9ZCT3TsOZ5O2omDYeruXrnqV7HaNb5UZt0I1bT5COaoqm0wz29thdak_CUQf24lLPaFTRmWwuj-zcmHD_I_j5nwkA7QWjWXIq2ge4wEyxU2Gt7yK"
    TOKENS = [jh_TOKEN, cu_TOKEN, sh_TOKEN]
    
    message = messaging.MulticastMessage(notification=NOTIFICATION, tokens=TOKENS)
    messaging.send_multicast(message)
    print("Successfully Send Notification.")


def push_notification_for_abnormal_status(n_stack, b_stack, th):
    if n_stack == th:
        n_stack = 0 # n_stack 초기화
        send_notification("아이 미탐지", "아이의 수면 자세가 탐지되지 않습니다. 확인해주세요!")
        
    if b_stack == th:
        b_stack = 0 # b_stack 초기화
        send_notification("비정상 수면 자세", "아이의 수면 자세가 위험할 수 있으니, 확인해주세요!")
        
    return n_stack, b_stack
    
    
if __name__ == '__main__':
    send_notification("For Test", "지금은 Testing")