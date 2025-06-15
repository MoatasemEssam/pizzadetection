# detect_and_classify.py
import cv2
from ultralytics import YOLO
from datetime import datetime

def detect_and_classify(queue, violation_counter):
    model = YOLO(r'C:\Users\HP\Desktop\testing\models/best.pt')
    ROI = (200, 150, 400, 300)
    cap = cv2.VideoCapture(r'C:\Users\HP\Desktop\testing\data\videos\pizza.mp4')

    def is_inside_roi(box, roi):
        x1, y1, x2, y2 = box
        rx1, ry1, rx2, ry2 = roi
        return x1 > rx1 and y1 > ry1 and x2 < rx2 and y2 < ry2

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break

        result = model(img)[0]
        boxes = result.boxes

        has_hand, has_scooper = False, False
        for box in boxes:
            label = model.names[int(box.cls[0])]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if label == "hand" and is_inside_roi((x1, y1, x2, y2), ROI):
                has_hand = True
            if label == "scooper":
                has_scooper = True
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if has_hand and not has_scooper:
            with violation_counter.get_lock():
                violation_counter.value += 1

        if queue.full():
            queue.get()
        queue.put(img)

        cv2.waitKey(1)
