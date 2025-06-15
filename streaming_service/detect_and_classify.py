import cv2
from ultralytics import YOLO
import time

def detect_and_classify(queue, violation_counter):
    model = YOLO(r'C:\Users\HP\Desktop\testing\models\yolo12m-v2.pt')
    cap = cv2.VideoCapture(r'C:\Users\HP\Desktop\testing\data\videos\pizza.mp4')

    ROI = (405, 256, 521, 621)

    def is_inside(box, roi):
        x1, y1, x2, y2 = box
        rx1, ry1, rx2, ry2 = roi
        return x1 > rx1 and y1 > ry1 and x2 < rx2 and y2 < ry2

    def is_outside(box, roi):
        x1, y1, x2, y2 = box
        rx1, ry1, rx2, ry2 = roi
        return x2 < rx1 or x1 > rx2 or y2 < ry1 or y1 > ry2

    hand_was_inside = False
    entry_exit_count = 0
    timer_started = False
    timer_start_time = None
    violation_counted = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        result = model(frame, verbose=False)[0]
        boxes = result.boxes

        hands_in_roi = []
        scooper_outside_roi = []

        for box in boxes:
            label = model.names[int(box.cls[0])]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            coords = (x1, y1, x2, y2)

            if label == "hand" and is_inside(coords, ROI):
                hands_in_roi.append(coords)
            elif label == "scooper" and is_outside(coords, ROI):
                scooper_outside_roi.append(coords)

            # Draw bounding boxes
            color = (0, 255, 255) if label == "scooper" else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

        # Draw ROI
        cv2.rectangle(frame, (ROI[0], ROI[1]), (ROI[2], ROI[3]), (255, 0, 0), 2)
        cv2.putText(frame, f"Violations: {violation_counter.value}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        scooper_present_outside = len(scooper_outside_roi) > 0
        hand_now_inside = len(hands_in_roi) > 0
        current_time = time.time()

        if not hand_now_inside and hand_was_inside:
            # Hand just exited ROI
            entry_exit_count += 1
            print(f"[INFO] Hand exited ROI, total exits: {entry_exit_count}")
            hand_was_inside = False
        elif hand_now_inside:
            hand_was_inside = True

        required_exits = 2 if scooper_present_outside else 1

        if entry_exit_count >= required_exits and not timer_started:
            timer_started = True
            timer_start_time = current_time
            violation_counted = False
            print("[INFO] Timer started")

        if timer_started:
            elapsed = current_time - timer_start_time
            if elapsed >= 4:
                if len(scooper_outside_roi) == 0 and not violation_counted:
                    with violation_counter.get_lock():
                        violation_counter.value += 1
                    print("[VIOLATION] No scooper detected within 4 seconds")
                    violation_counted = True

                # Reset after timer completes
                timer_started = False
                entry_exit_count = 0
                hand_was_inside = False
                timer_start_time = None
                violation_counted = False

        if queue.full():
            queue.get()
        queue.put(frame)

        time.sleep(0.03)
        cv2.waitKey(1)
