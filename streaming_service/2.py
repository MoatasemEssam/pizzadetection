import cv2
from ultralytics import YOLO
import time

def detect_and_classify(queue, violation_counter):
    model = YOLO(r'C:\Users\HP\Desktop\testing\models/yolo12m-v2.pt')
    cap = cv2.VideoCapture(r'C:\Users\HP\Desktop\testing\data\videos\pizza.mp4')

    ROI = (405, 256, 521, 621)  # (x1, y1, x2, y2)

    def is_inside_roi(box, roi):
        x1, y1, x2, y2 = box
        rx1, ry1, rx2, ry2 = roi
        return x1 > rx1 and y1 > ry1 and x2 < rx2 and y2 < ry2

    # State variables
    hand_inside_roi = False
    hand_entered = False
    hand_exited = False
    exit_timer = 0
    violation_counted = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        result = model(frame, verbose=False)[0]
        boxes = result.boxes

        hands_in_roi = []
        hands_outside_roi = []
        scoopers_outside_roi = []

        for box in boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            current_box = (x1, y1, x2, y2)

            # Separate hands and scoopers
            if label == "hand":
                if is_inside_roi(current_box, ROI):
                    hands_in_roi.append(current_box)
                else:
                    hands_outside_roi.append(current_box)

            if label == "scooper" and not is_inside_roi(current_box, ROI):
                scoopers_outside_roi.append(current_box)

            # Draw box
            color = (255, 255, 0) if label == "hand" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw ROI
        rx1, ry1, rx2, ry2 = ROI
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)

        # Logic
        if hands_in_roi:
            hand_inside_roi = True
            hand_entered = True
            hand_exited = False
            exit_timer = 0
            violation_counted = False

        elif hand_entered and not hands_in_roi:
            if not hand_exited:
                print("🟡 Hand exited ROI. Monitoring for scooper...")
                hand_exited = True

            exit_timer += 1

            if exit_timer >= 6:
                if scoopers_outside_roi:
                    # Scooper appeared after 6 frames — cancel cycle
                    print("✅ Scooper appeared after 6 frames. Resetting...")
                    hand_entered = False
                    hand_exited = False
                    exit_timer = 0
                    violation_counted = False
                elif not violation_counted:
                    # No scooper found → count violation
                    with violation_counter.get_lock():
                        violation_counter.value += 1
                    violation_counted = True
                    print("❌ Violation counted!")
                    hand_entered = False
                    hand_exited = False
                    exit_timer = 0

        # Show violation count
        cv2.putText(frame, f"Violations: {violation_counter.value}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Push frame to queue
        if queue.full():
            queue.get()
        queue.put(frame)

        cv2.waitKey(1)
        time.sleep(0.03)
