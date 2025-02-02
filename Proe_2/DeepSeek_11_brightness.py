# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 20:05:06 2025

@author: Lenovo
"""

import cv2
import mediapipe as mp
import numpy as np
import screen_brightness_control as sbc

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Brightness thresholds
MIN_BRIGHTNESS = 10  # Minimum screen brightness (10%)
MAX_BRIGHTNESS = 100  # Maximum screen brightness (100%)
BRIGHTNESS_STEP = 5  # Step size for brightness adjustment

# Brightness thresholds for background and face
BACKGROUND_BRIGHT_THRESHOLD = 150  # Brightness threshold for background
FACE_BRIGHT_THRESHOLD = 120  # Brightness threshold for face

def calculate_brightness(region):
    """
    Calculate the average brightness of a region.
    """
    gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    return np.mean(gray_region)

def adjust_brightness(background_brightness, face_brightness):
    """
    Adjust the screen brightness based on the brightness of the background and face.
    """
    current_brightness = sbc.get_brightness()[0]  # Get current screen brightness

    # Adjust based on background brightness
    if background_brightness > BACKGROUND_BRIGHT_THRESHOLD:
        # Increase brightness if the background is bright
        new_brightness = min(current_brightness + BRIGHTNESS_STEP, MAX_BRIGHTNESS)
    else:
        # Decrease brightness if the background is dark
        new_brightness = max(current_brightness - BRIGHTNESS_STEP, MIN_BRIGHTNESS)

    # Adjust based on face brightness
    if face_brightness > FACE_BRIGHT_THRESHOLD:
        # Decrease brightness if the face is too bright
        new_brightness = max(new_brightness - BRIGHTNESS_STEP, MIN_BRIGHTNESS)

    # Set the new brightness
    sbc.set_brightness(new_brightness)

# Initialize webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture frame.")
        break

    # Flip the frame horizontally for a mirror-like view
    frame = cv2.flip(frame, 1)

    # Detect faces in the frame
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            # Get the bounding box of the face
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

            # Extract the face region
            face_region = frame[y:y+h, x:x+w]

            # Calculate the brightness of the face
            face_brightness = calculate_brightness(face_region)

            # Extract the background region (everything except the face)
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)
            background_region = cv2.bitwise_and(frame, frame, mask=~mask)

            # Calculate the brightness of the background
            background_brightness = calculate_brightness(background_region)

            # Adjust the screen brightness
            adjust_brightness(background_brightness, face_brightness)

            # Draw the bounding box and display brightness values
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Face Brightness: {int(face_brightness)}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Background Brightness: {int(background_brightness)}", (x, y+h+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Frame", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()