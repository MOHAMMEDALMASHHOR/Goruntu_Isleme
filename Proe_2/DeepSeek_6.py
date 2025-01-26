# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 15:33:52 2025

@author: Lenovo
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 14:52:03 2025

@author: Lenovo
"""

import cv2
import mediapipe as mp
import time
import math

# Initialize MediaPipe Pose and FaceMesh
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Parameters
sip_count = 0
last_sip_time = 0
sip_cooldown = 1  # seconds
sip_threshold = 0.05  # Distance threshold for wrist-to-mouth
wrist_bend_threshold = 150  # Angle threshold for wrist bending
mouth_open_threshold = 0.03  # Distance threshold for mouth open

# Function to calculate distance between two landmarks
def calculate_distance(lm1, lm2):
    return ((lm1.x - lm2.x) ** 2 + (lm1.y - lm2.y) ** 2) ** 0.5

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# Function to detect wrist bending
def is_wrist_bent(landmarks):
    wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
    elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
    shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

    angle = calculate_angle(wrist, elbow, shoulder)
    return angle < wrist_bend_threshold

# Function to detect mouth open
def is_mouth_open(face_landmarks):
    upper_lip = face_landmarks.landmark[13]  # Upper lip landmark
    lower_lip = face_landmarks.landmark[14]  # Lower lip landmark
    distance = calculate_distance(upper_lip, lower_lip)
    return distance > mouth_open_threshold

# Start Video Capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    # Convert to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process with MediaPipe
    pose_results = pose.process(image_rgb)
    face_results = face_mesh.process(image_rgb)

    # Draw Pose and FaceMesh (only for mouth landmarks)
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            pose_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
        )

    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            # Draw mouth landmarks
            for idx in [13, 14, 78, 308]:  # Indices for mouth landmarks
                mouth_point = face_landmarks.landmark[idx]
                x, y = int(mouth_point.x * frame.shape[1]), int(mouth_point.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)

            # Draw lines between mouth landmarks
            cv2.line(frame,
                     (int(face_landmarks.landmark[13].x * frame.shape[1]), int(face_landmarks.landmark[13].y * frame.shape[0])),
                     (int(face_landmarks.landmark[14].x * frame.shape[1]), int(face_landmarks.landmark[14].y * frame.shape[0])),
                     (0, 255, 0), 2)
            cv2.line(frame,
                     (int(face_landmarks.landmark[78].x * frame.shape[1]), int(face_landmarks.landmark[78].y * frame.shape[0])),
                     (int(face_landmarks.landmark[308].x * frame.shape[1]), int(face_landmarks.landmark[308].y * frame.shape[0])),
                     (0, 255, 0), 2)

    # Detect gestures
    if pose_results.pose_landmarks and face_results.multi_face_landmarks:
        landmarks = pose_results.pose_landmarks.landmark
        face_landmarks = face_results.multi_face_landmarks[0]

        # Calculate wrist-to-mouth distance
        wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        mouth = face_landmarks.landmark[13]  # Use landmark 13 for the lower part of the mouth
        wrist_to_mouth_distance = calculate_distance(wrist, mouth)

        # Check if wrist is bent
        wrist_bent = is_wrist_bent(landmarks)

        # Check if mouth is open
        mouth_open = is_mouth_open(face_landmarks)

        # Check for sipping gesture
        if (wrist_to_mouth_distance < sip_threshold or wrist_bent) and mouth_open:
            current_time = time.time()
            if current_time - last_sip_time > sip_cooldown:
                sip_count += 1
                last_sip_time = current_time
                print(f"Sip detected! Total sips: {sip_count}")

    # Display sip count and feedback on the frame
    cv2.putText(frame, f'Sips: {sip_count}', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Wrist Bent: {"Yes" if wrist_bent else "No"}', (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Mouth Open: {"Yes" if mouth_open else "No"}', (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Water Intake Tracker", frame)

    # Break on 'q' key
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()