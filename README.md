# pizzadetection

## Features
This project implements a real-time computer vision system to detect scooper usage violations in a pizza store. Using a YOLO-based object detection model and a Flask streaming service, it monitors worker hand movements and scooper presence in specified regions of interest (ROIs) to count violations when a worker fails to use a scooper properly.

## Features
- Detects and classifies hands and scoopers using a custom YOLOv9 model.
- Monitors activity inside predefined Regions of Interest (ROIs).
- Counts and records violations when hands exit the ROI without a scooper.
- Displays live video stream with detection overlays using Flask.
- Tracks total violations and shows real-time results in the browser.

##  System Architecture

![arc](https://github.com/user-attachments/assets/729da144-2951-4203-876d-d0dd1676fe43)


## 🚀 How to Run

1. **Install dependencies**:
   ```bash
   pip install flask opencv-python ultralytics
Set paths:
model_path = 'models/yolo12m-v2.pt'
video_path = 'data/videos/pizza.mp4'

Run the app:
python streaming_service.py

Open your browser:
Stream video: http://localhost:5000
Get violation count (JSON): http://localhost:5000/violations

Violation Logic
A hand must enter and exit the ROI.
If a scooper is not detected outside the ROI within 4 seconds after the hand exits, it is counted as a violation.
If a scooper is already outside the ROI when the hand exits, no violation is counted and the system waits for a new cycle.
