import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import messaging
    
def push_notification(title, body):
    if not firebase_admin._apps:
        cred = credentials.Certificate(os.path.join(os.path.dirname(os.path.realpath(__file__)), "lstm-bob-firebase-adminsdk-gmh89-3c82003d7d.json"))
        firebase_admin.initialize_app(cred)

    NOTIFICATION = messaging.Notification(title=title, body=body)
    jh_TOKEN = "f9k2g8Q7TBOgFe-Bfkp1mw:APA91bGZYdj8evuB4Hr-ERwfYnaoGhLe2ANnzJ0t5B8dBz-ABoJ1RPBjh6DJb1yBzCHV0Wu9Yv5zHqeNbr-vCi2mZDMZDhlNb_dQx8dGyi3iONjIJj4UdmS9THZZ5190R17qSr1vxt69"
    jh2_TOKEN = "eoOBpJALTGik5NjSSuUXIT:APA91bG2zCeHj3KW2GnGeb4XN9q20oGATEM8Zjd9X1HL6p_FIaSPjhMOEKEe8GxGSlS4fAuaz2vWk4R6pBFfWyExYjrTA1N5fMwWeHdBzC5h1kA9R2MqdA_uDlZ8jm6RPw-RnTmLdvkJ"
    cu_TOKEN = "eTP0p7mO_Ex5uibUFRkEuL:APA91bFw60PkaSOzGV4qEVCtxYXL7kAMJ7f0_68mDdN4Cn0hnlDaO7CV7HuGxBTgtEBIZY8Ph1Xm4wv2WZdhv0AVieF53O_WkLtYyZV0HTBCVxN-twc9cOyIURdNd9jlks1mlqccs3Po"
    TOKENS = [jh_TOKEN, jh2_TOKEN, cu_TOKEN]
    
    message = messaging.MulticastMessage(notification=NOTIFICATION, tokens=TOKENS)
    # messaging.send_multicast(message)
    print("Successfully Send Notification.")
    
if __name__ == '__main__':
    push_notification("", "")