# streaming_service.py
from flask import Flask, Response, render_template, jsonify
import multiprocessing as mp
import cv2
import time
import numpy as np
from detect_and_classify import detect_and_classify

app = Flask(__name__)

frame_queue = mp.Queue(maxsize=1)
violation_count = mp.Value('i', 0)

@app.route("/")
def index():
    return render_template("index.html")  # Put your HTML in 'templates/index.html'

@app.route("/violations")
def get_violation_count():
    return jsonify({"count": violation_count.value})

def generate_video_stream():
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            ret, jpeg = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        time.sleep(0.05)

@app.route("/video_feed")
def video_feed():
    return Response(generate_video_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    p = mp.Process(target=detect_and_classify, args=(frame_queue, violation_count))
    p.start()
    app.run(debug=False)
    p.join()
