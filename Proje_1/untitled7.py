# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 14:55:43 2025

@author: Lenovo
"""

import cv2
import mediapipe as mp
import time
import numpy as np

# Initialize MediaPipe solutions
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Set up pose detection
pose_detection = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Counters and parameters
sip_count = 0
last_sip_time = 0
sip_cooldown = 1  # Time in seconds between sips
estimated_sips_per_cup = 10  # Adjust this value as needed
wrist_positions = []
sip_threshold = 0.1  # Movement threshold to count as sip

# Start video capture
cap = cv2.VideoCapture(0)

# Function to estimate cups of water based on sips
def estimate_cups(sips):
    return sips / estimated_sips_per_cup

# Function to apply Gaussian blur and edge detection
def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Apply Canny Edge Detection
    edges = cv2.Canny(blurred, 50, 150)
    return edges

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Failed to capture frame")
        continue

    # Preprocess the image (grayscale + edge detection)
    edges = preprocess_image(image)

    # Convert the image for pose detection
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image for pose detection
    pose_results = pose_detection.process(image_rgb)

    # Check if pose landmarks are detected
    if pose_results.pose_landmarks:
        # Access the right wrist position
        right_wrist = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

        # Append the wrist position to a list to check movement
        wrist_positions.append((right_wrist.x, right_wrist.y))
        if len(wrist_positions) > 10:
            wrist_positions.pop(0)

        # Check if the wrist is moving toward the face (drinking movement)
        if len(wrist_positions) > 5:
            # Calculate movement direction by comparing recent wrist positions
            start_pos = wrist_positions[0]
            end_pos = wrist_positions[-1]
            movement = abs(end_pos[1] - start_pos[1])  # Vertical movement

            # If wrist movement is above threshold, count as sip
            if movement > sip_threshold and (time.time() - last_sip_time) > sip_cooldown:
                sip_count += 1
                last_sip_time = time.time()  # Reset cooldown timer
                print(f"Sip detected! Total sips: {sip_count}")
                print(f"Estimated cups of water: {estimate_cups(sip_count):.2f}")

    # Display the image with overlay text
    cv2.putText(image, f'Cups: {estimate_cups(sip_count):.2f}', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(image, f'Sips: {sip_count}', (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the processed image with edge detection (optional for debugging)
    cv2.imshow('Edge Detection', edges)

    # Show the original image
    cv2.imshow('Water Intake Tracker', image)

    # Exit on 'q' key
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
