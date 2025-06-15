# pizzadetection

## Overview
This project implements a real-time computer vision system to detect scooper usage violations in a pizza store. Using a YOLO-based object detection model and a Flask streaming service, it monitors worker hand movements and scooper presence in specified regions of interest (ROIs) to count violations when a worker fails to use a scooper properly.

## Features
-Real-time video stream with bounding boxes for hands and scoopers.
-Violation counting logic with configurable ROI boundaries.
-Handles false positives from scooper detection by applying time and spatial conditions.
-Flask-based web interface showing live video feed and total violation count.
-Configurable timers and logic cycles to minimize false alarms.

##Requirements
-Python 3.8+
-Ultralytics YOLO (ultralytics package)
-OpenCV (opencv-python)
-Flask (flask)
-Multiprocessing (built-in Python library)

## How It Works
-The YOLO model detects hands and scoopers in each video frame.
-The system tracks hand entries and exits within a defined ROI.
-If hands enter and exit the ROI and no scooper is detected outside the ROI for 4 seconds, it counts a violation.
-The logic resets in cycles to continuously monitor.
-Scooper detection inaccuracies are handled via timing and presence checks to reduce false violations.
