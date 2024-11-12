# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 17:32:34 2024

@author: Lenovo
"""

import cv2
import mediapipe as mp
import time

# Initialize MediaPipe solutions
mp_face_detection = mp.solutions.face_detection
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Set up face and pose detection
face_detection = mp_face_detection.FaceDetection()
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

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Failed to capture frame")
        continue

    # Convert to RGB for processing
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image for face and pose detection
    face_results = face_detection.process(image_rgb)
    pose_results = pose_detection.process(image_rgb)

    # Check if face and pose landmarks are detected
    if face_results.detections and pose_results.pose_landmarks:
        # Access the first face detection
        face = face_results.detections[0].location_data.relative_bounding_box
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
    cv2.imshow('Water Intake Tracker', image)

    # Exit on 'q' key
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
