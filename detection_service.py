import cv2
from ultralytics import YOLO
import datetime
import os

class RoiState:
    def __init__(self, roi_coords, name):
        self.roi_coords = roi_coords
        self.name = name
        self.hand_entered = False
        self.hand_exited = False
        self.exit_timer_frames = 0
        self.violation_counted = False
        self.last_hand_bbox = None

    def is_hand_inside(self, hand_box):
        x1_h, y1_h, x2_h, y2_h = hand_box
        rx1, ry1, rx2, ry2 = self.roi_coords
        hand_center_x = (x1_h + x2_h) // 2
        hand_center_y = (y1_h + y2_h) // 2
        return (rx1 <= hand_center_x <= rx2 and ry1 <= hand_center_y <= ry2)

    def reset_state(self):
        self.hand_entered = False
        self.hand_exited = False
        self.exit_timer_frames = 0
        self.violation_counted = False
        self.last_hand_bbox = None

def detect_and_classify(queue, violation_counter, model_path=None, video_path=None):
    if model_path is None:
        model_path = r'C:\Users\HP\Desktop\testing\models/yolo12m-v2.pt'
    if video_path is None:
        video_path = r'C:\Users\HP\Desktop\testing\data\videos/pizza.mp4'

    if not os.path.exists(model_path) or not os.path.exists(video_path):
        return

    try:
        model = YOLO(model_path)
    except:
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"recorded_violation_{timestamp}.mp4"
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    ROIS_COORDS = [
        (461, 342, 510, 390),
        (447, 395, 496, 449),
        (418, 476, 481, 539)
    ]

    def is_box_inside_any_roi(box_coords, all_roi_definitions):
        cx = (box_coords[0] + box_coords[2]) // 2
        cy = (box_coords[1] + box_coords[3]) // 2
        for rx1, ry1, rx2, ry2 in all_roi_definitions:
            if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                return True
        return False

    GRACE_PERIOD_FRAMES = 30
    PROXIMITY_THRESHOLD = 250
    roi_states = [RoiState(coords, f"ROI_{i+1}") for i, coords in enumerate(ROIS_COORDS)]
    free_pass_active = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        result = model(frame, verbose=False)[0]
        boxes = result.boxes
        current_hands = []
        all_detected_scoopers = []

        for box in boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            current_box = (x1, y1, x2, y2)

            if label == "hand":
                current_hands.append(current_box)
            elif label == "scooper":
                all_detected_scoopers.append(current_box)

            color = (255, 255, 0) if label == "hand" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        scoopers_outside_rois = [s for s in all_detected_scoopers if not is_box_inside_any_roi(s, ROIS_COORDS)]
        if scoopers_outside_rois and not free_pass_active:
            free_pass_active = True

        for roi_state in roi_states:
            rx1, ry1, rx2, ry2 = roi_state.roi_coords
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)
            cv2.putText(frame, roi_state.name, (rx1, ry1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            hands_in_roi = [h for h in current_hands if roi_state.is_hand_inside(h)]

            if hands_in_roi:
                if not roi_state.hand_entered:
                    roi_state.reset_state()
                roi_state.hand_entered = True
                roi_state.last_hand_bbox = hands_in_roi[0]

            elif roi_state.hand_entered and not hands_in_roi:
                if not roi_state.hand_exited:
                    roi_state.hand_exited = True
                roi_state.exit_timer_frames += 1

                if roi_state.exit_timer_frames >= GRACE_PERIOD_FRAMES:
                    near_hand = False
                    if roi_state.last_hand_bbox:
                        hx = (roi_state.last_hand_bbox[0] + roi_state.last_hand_bbox[2]) // 2
                        hy = (roi_state.last_hand_bbox[1] + roi_state.last_hand_bbox[3]) // 2
                        for s in scoopers_outside_rois:
                            sx = (s[0] + s[2]) // 2
                            sy = (s[1] + s[3]) // 2
                            dist = ((hx - sx)**2 + (hy - sy)**2)**0.5
                            if dist < PROXIMITY_THRESHOLD:
                                near_hand = True
                                break

                    if near_hand or free_pass_active:
                        roi_state.reset_state()
                        if free_pass_active and not scoopers_outside_rois:
                            free_pass_active = False
                    elif not roi_state.violation_counted:
                        with violation_counter.get_lock():
                            violation_counter.value += 1
                        roi_state.violation_counted = True
                        roi_state.reset_state()
            else:
                if not roi_state.hand_entered and roi_state.exit_timer_frames > 0:
                    roi_state.reset_state()

        cv2.putText(frame, f"Total Violations: {violation_counter.value}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if free_pass_active:
            cv2.putText(frame, "FREE PASS ACTIVE", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        out_writer.write(frame)

        if queue.full():
            try:
                queue.get_nowait()
            except:
                pass
        try:
            queue.put_nowait(frame)
        except:
            pass

    cap.release()
    out_writer.release()
