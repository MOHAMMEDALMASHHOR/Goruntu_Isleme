# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 14:07:29 2025

@author: Lenovo
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import math
import pygame

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

# Calibration variables
is_calibrated = False
calibration_frames = 0
calibration_shoulder_angles = []
calibration_neck_angles = []
calibration_shoulder_distances = []  # New: Store shoulder distances during calibration
shoulder_threshold = 0  # Placeholder; will be set after calibration
neck_threshold = 0      # Placeholder; will be set after calibration
calibrated_shoulder_distance = 0  # New: Calibrated shoulder distance

# Alert system variables
alert_cooldown = 5  # Time in seconds to wait between alerts
last_alert_time = 0

# Initialize MediaPipe Pose and webcam
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)

# Preprocessing configurations
kernel = np.ones((5, 5), np.uint8)

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

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Extract key body landmarks
        left_shoulder = (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * frame.shape[1]),
                         int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * frame.shape[0]))
        right_shoulder = (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * frame.shape[1]),
                          int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * frame.shape[0]))
        left_ear = (int(landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x * frame.shape[1]),
                    int(landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y * frame.shape[0]))

        # Calculate angles
        shoulder_angle = calculate_angle(left_shoulder, right_shoulder, (right_shoulder[0], 0))
        neck_angle = calculate_angle(left_ear, left_shoulder, (left_shoulder[0], 0))

        # Calculate shoulder distance
        shoulder_distance = math.hypot(right_shoulder[0] - left_shoulder[0], right_shoulder[1] - left_shoulder[1])

        # Calibration
        if not is_calibrated and calibration_frames < 30:
            calibration_shoulder_angles.append(shoulder_angle)
            calibration_neck_angles.append(neck_angle)
            calibration_shoulder_distances.append(shoulder_distance)  # New: Store shoulder distance
            calibration_frames += 1
            cv2.putText(frame, f"Calibrating... {calibration_frames}/30", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        elif not is_calibrated:
            shoulder_threshold = np.mean(calibration_shoulder_angles) - 10
            neck_threshold = np.mean(calibration_neck_angles) - 10
            calibrated_shoulder_distance = np.mean(calibration_shoulder_distances)  # New: Set calibrated shoulder distance
            is_calibrated = True
            print(f"Calibration complete. Shoulder threshold: {shoulder_threshold:.1f}, Neck threshold: {neck_threshold:.1f}")
            print(f"Calibrated shoulder distance: {calibrated_shoulder_distance:.2f}")

        # Draw skeleton and angles
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        midpoint = ((left_shoulder[0] + right_shoulder[0]) // 2, (left_shoulder[1] + right_shoulder[1]) // 2)
        draw_angle(frame, left_shoulder, midpoint, (midpoint[0], 0), shoulder_angle, (255, 0, 0))
        draw_angle(frame, left_ear, left_shoulder, (left_shoulder[0], 0), neck_angle, (0, 255, 0))

        # Draw shoulder line
        line_color = (0, 255, 0)  # Green by default
        if is_calibrated and shoulder_distance < calibrated_shoulder_distance * 0.9:  # 90% of calibrated distance
            line_color = (0, 0, 255)  # Red if slouching
        cv2.line(frame, left_shoulder, right_shoulder, line_color, 2)

        # Feedback
        if is_calibrated:
            current_time = time.time()
            if shoulder_angle < shoulder_threshold or neck_angle < neck_threshold:
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

    # Display the frame
    cv2.imshow('Posture Corrector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()