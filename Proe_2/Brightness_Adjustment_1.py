# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 14:40:33 2025

@author: Lenovo
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import math
import pygame
from screen_brightness_control import set_brightness

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize sound system
pygame.mixer.init()
sound_file = "beep-warning-6387.mp3"  # Replace with the correct file path

def play_sound(file_path):
    """
    Plays the sound file using pygame.
    """
    try:
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
    except Exception as e:
        print(f"Error playing sound: {e}")

def calculate_angle(a, b, c):
    """
    Calculate the angle between three points.
    a, b, c are tuples representing points (x, y).
    """
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    cos_angle = (ba[0] * bc[0] + ba[1] * bc[1]) / (
        math.sqrt(ba[0] ** 2 + ba[1] ** 2) * math.sqrt(bc[0] ** 2 + bc[1] ** 2)
    )
    angle = math.degrees(math.acos(cos_angle))
    return angle

def draw_angle(frame, p1, p2, p3, angle, color):
    """
    Draws the angle at point p2 on the frame.
    """
    cv2.line(frame, p1, p2, color, 2)
    cv2.line(frame, p2, p3, color, 2)
    cv2.putText(frame, f"{int(angle)}\u00b0", (p2[0] + 10, p2[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

def adjust_brightness_based_on_light(frame):
    """
    Adjust screen brightness based on room light intensity.
    """
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray_frame)
    brightness_level = int((avg_brightness / 255) * 100)
    set_brightness(brightness_level)
    print(f"Adjusted brightness to {brightness_level}%")

def suggest_breaks(last_break_time, current_time):
    """
    Suggest breaks every 30 minutes.
    """
    if current_time - last_break_time > 1800:  # 30 minutes
        print("It's time to stand up and stretch for a few minutes!")
        play_sound(sound_file)
        return current_time
    return last_break_time

# Calibration variables
is_calibrated = False
calibration_frames = 0
calibration_shoulder_angles = []
calibration_neck_angles = []
shoulder_threshold = 0  # Placeholder; will be set after calibration
neck_threshold = 0      # Placeholder; will be set after calibration

# Alert system variables
alert_cooldown = 5  # Time in seconds to wait between alerts
last_alert_time = 0
last_break_time = time.time()

# Initialize MediaPipe Pose and webcam
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)

# Preprocessing configurations
kernel = np.ones((5, 5), np.uint8)

# Video writer setup
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('posture_output.avi', fourcc, 20.0, (640, 480))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Apply Gaussian blur to reduce noise
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), cv2.BORDER_DEFAULT)

    # Convert to grayscale for analysis
    gray_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization for better visibility
    equalized_frame = cv2.equalizeHist(gray_frame)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    # Adjust screen brightness based on ambient light
    adjust_brightness_based_on_light(frame)

    # Suggest breaks
    last_break_time = suggest_breaks(last_break_time, time.time())

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Extract key body landmarks
        left_shoulder = (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * frame.shape[1]),
                         int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * frame.shape[0]))
        right_shoulder = (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * frame.shape[1]),
                          int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * frame.shape[0]))
        left_ear = (int(landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x * frame.shape[1]),
                    int(landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y * frame.shape[0]))
        left_hip = (int(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * frame.shape[1]),
                    int(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * frame.shape[0]))

        # Calculate angles
        shoulder_angle = calculate_angle(left_shoulder, right_shoulder, (right_shoulder[0], 0))
        neck_angle = calculate_angle(left_ear, left_shoulder, (left_shoulder[0], 0))
        spine_angle = calculate_angle(left_shoulder, left_hip, (left_hip[0], left_hip[1] + 100))

        # Calibration
        if not is_calibrated and calibration_frames < 30:
            calibration_shoulder_angles.append(shoulder_angle)
            calibration_neck_angles.append(neck_angle)
            calibration_frames += 1
            cv2.putText(frame, f"Calibrating... {calibration_frames}/30", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        elif not is_calibrated:
            shoulder_threshold = np.mean(calibration_shoulder_angles) - 10
            neck_threshold = np.mean(calibration_neck_angles) - 10
            is_calibrated = True
            print(f"Calibration complete. Shoulder threshold: {shoulder_threshold:.1f}, Neck threshold: {neck_threshold:.1f}")

        # Draw skeleton and angles
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        midpoint = ((left_shoulder[0] + right_shoulder[0]) // 2, (left_shoulder[1] + right_shoulder[1]) // 2)
        draw_angle(frame, left_shoulder, midpoint, (midpoint[0], 0), shoulder_angle, (255, 0, 0))
        draw_angle(frame, left_ear, left_shoulder, (left_shoulder[0], 0), neck_angle, (0, 255, 0))
        draw_angle(frame, left_shoulder, left_hip, (left_hip[0], left_hip[1] + 100), spine_angle, (0, 0, 255))

        # Feedback
        if is_calibrated:
            current_time = time.time()
            if shoulder_angle < shoulder_threshold or neck_angle < neck_threshold or spine_angle < 160:
                status = "Poor Posture"
                color = (0, 0, 255)  # Red
                if current_time - last_alert_time > alert_cooldown:
                    print("Poor posture detected! Please sit up straight.")
                    play_sound(sound_file)
                    last_alert_time = current_time
            else:
                status = "Good Posture"
                color = (0, 255, 0)  # Green

            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            cv2.putText(frame, f"Shoulder Angle: {shoulder_angle:.1f}/{shoulder_threshold:.1f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Neck Angle: {neck_angle:.1f}/{neck_threshold:.1f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Spine Angle: {spine_angle:.1f}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Posture Corrector', frame)

    # Save the processed frame to the video
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
import cv2
import mediapipe as mp
import numpy as np
import time
import math
import pygame
from screen_brightness_control import set_brightness

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
model_config = 'models/yolov3.cfg'  # Update with the correct path
model_weights = 'models/yolov3.weights'  # Update with the correct path
coco_names = 'models/coco.names'  # Update with the correct path
with open('models/yolov3.cfg', 'r') as f:
    print(f.read())

# Load class names
classes = []
with open(coco_names, 'r') as f:
    classes = [line.strip() for line in f.readlines()]
# Initialize sound system
pygame.mixer.init()
sound_file = "beep-warning-6387.mp3"  # Replace with the correct file path

# Initialize object detection model for bottle detection (using YOLO)
model_config = 'yolov3.cfg'  # Path to YOLO config file
model_weights = 'yolov3.weights'  # Path to YOLO weights file
net = cv2.dnn.readNetFromDarknet(model_config, model_weights)
classes = []
with open('coco.names', 'r') as f:  # Path to COCO class names
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Sip tracking variables
sip_count = 0
bottle_detected = False
last_bottle_position = None

def detect_bottle(frame):
    """
    Detects a bottle in the frame using YOLO and returns its position.
    """
    global bottle_detected, last_bottle_position

    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == "bottle":
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            bottle_detected = True
            last_bottle_position = (x + w // 2, y + h // 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Bottle", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            return last_bottle_position
    bottle_detected = False
    return None

def track_sips(frame):
    """
    Tracks the sips taken by the user based on bottle movement.
    """
    global sip_count, last_bottle_position

    if bottle_detected and last_bottle_position:
        current_bottle_position = last_bottle_position

        # Logic to detect bottle movement towards the mouth
        # For simplicity, assume a region near the top-center of the frame as the "mouth area"
        mouth_region = (frame.shape[1] // 2, frame.shape[0] // 4)
        if current_bottle_position[1] < mouth_region[1]:
            sip_count += 1
            print(f"Sip detected! Total sips: {sip_count}")

        # Update the last known position
        last_bottle_position = current_bottle_position

# Calibration variables
is_calibrated = False
calibration_frames = 0
calibration_shoulder_angles = []
calibration_neck_angles = []
shoulder_threshold = 0  # Placeholder; will be set after calibration
neck_threshold = 0      # Placeholder; will be set after calibration

# Alert system variables
alert_cooldown = 5  # Time in seconds to wait between alerts
last_alert_time = 0
last_break_time = time.time()

# Initialize MediaPipe Pose and webcam
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)

# Preprocessing configurations
kernel = np.ones((5, 5), np.uint8)

# Video writer setup
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('posture_output.avi', fourcc, 20.0, (640, 480))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Apply Gaussian blur to reduce noise
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), cv2.BORDER_DEFAULT)

    # Convert to grayscale for analysis
    gray_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization for better visibility
    equalized_frame = cv2.equalizeHist(gray_frame)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    # Adjust screen brightness based on ambient light
    adjust_brightness_based_on_light(frame)

    # Suggest breaks
    last_break_time = suggest_breaks(last_break_time, time.time())

    # Detect bottle and track sips
    bottle_position = detect_bottle(frame)
    track_sips(frame)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Extract key body landmarks
        left_shoulder = (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * frame.shape[1]),
                         int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * frame.shape[0]))
        right_shoulder = (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * frame.shape[1]),
                          int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * frame.shape[0]))
        left_ear = (int(landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x * frame.shape[1]),
                    int(landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y * frame.shape[0]))
        left_hip = (int(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * frame.shape[1]),
                    int(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * frame.shape[0]))

        # Calculate angles
        shoulder_angle = calculate_angle(left_shoulder, right_shoulder, (right_shoulder[0], 0))
        neck_angle = calculate_angle(left_ear, left_shoulder, (left_shoulder[0], 0))
        spine_angle = calculate_angle(left_shoulder, left_hip, (left_hip[0], left_hip[1] + 100))

        # Calibration
        if not is_calibrated and calibration_frames < 30:
            calibration_shoulder_angles.append(shoulder_angle)
            calibration_neck_angles.append(neck_angle)
            calibration_frames += 1
            cv2.putText(frame, f"Calibrating... {calibration_frames}/30", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        elif not is_calibrated:
            shoulder_threshold = np.mean(calibration_shoulder_angles) - 10
            neck_threshold = np.mean(calibration_neck_angles) - 10
            is_calibrated = True
            print(f"Calibration complete. Shoulder threshold: {shoulder_threshold:.1f}, Neck threshold: {neck_threshold:.1f}")

        # Draw skeleton and angles
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        midpoint = ((left_shoulder[0] + right_shoulder[0]) // 2, (left_shoulder[1] + right_shoulder[1]) // 2)
        draw_angle(frame, left_shoulder, midpoint, (midpoint[0], 0), shoulder_angle, (255, 0, 0))
        draw_angle(frame, left_ear, left_shoulder, (left_shoulder[0], 0), neck_angle, (0, 255, 0))
        draw_angle(frame, left_shoulder, left_hip, (left_hip[0], left_hip[1] + 100), spine_angle, (0, 0, 255))

        # Feedback
        if is_calibrated:
            current_time = time.time()
            if shoulder_angle < shoulder_threshold or neck_angle < neck_threshold or spine_angle < 160:
                status = "Poor Posture"
                color = (0, 0, 255)  # Red
                if current_time - last_alert_time > alert_cooldown:
                    print("Poor posture detected! Please sit up straight.")
                    play_sound(sound_file)
                    last_alert_time = current_time
            else:
                status = "Good Posture"
                color = (0, 255, 0)  # Green

            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            cv2.putText(frame, f"Shoulder Angle: {shoulder_angle:.1f}/{shoulder_threshold:.1f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Neck Angle: {neck_angle:.1f}/{neck_threshold:.1f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Spine Angle: {spine_angle:.1f}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # Display sip count
        cv2.putText(frame, f"Sips Taken: {sip_count}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

    # Display the frame
   
