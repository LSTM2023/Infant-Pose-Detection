from flask import Flask, render_template, Response
import cv2
import time

app = Flask(__name__)

camera = cv2.VideoCapture(0) # use 0 for web camera
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # 2592
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # 1944
camera.set(cv2.CAP_PROP_FPS, 20)

def gen_frames(): # generate frame by frame from camera
    prevTime = 0
    frame_index = 0
    
    while True:
        # Capture frame-by-frame
        success, frame = camera.read() # read the camera frame
        if not success:
            break
        else:
            frame = cv2.flip(frame, -1)
            
            curTime = time.time()
            sec = curTime - prevTime
            prevTime = curTime
            fps = 1. / sec
            fps_str = f"RPi FPS : {fps:.01f}"
            cv2.putText(frame, fps_str, (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            
            frame_index_str = f"Frame : {frame_index}"
            # cv2.putText(frame, frame_index_str, (0, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            frame_index += 1
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            # yield (b'--frame\r\n'
            #       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') # concat frame one by one and show result
            
            yield (b'--frame\r\n'
                   b'Content-Type:image/jpeg\r\n'
                   b'Content-Length: ' + f"{len(frame)}".encode() + b'\r\n'
                   b'\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)