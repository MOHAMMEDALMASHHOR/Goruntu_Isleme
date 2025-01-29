# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 19:51:56 2025

@author: Lenovo
"""

import cv2
import mediapipe as mp
import numpy as np
import screen_brightness_control as sbc

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Brightness thresholds and buffer range
OPTIMAL_BRIGHTNESS_RANGE = (120, 180)  # Optimal brightness range
BUFFER_RANGE = 10  # Buffer range to avoid continuous adjustments
MIN_BRIGHTNESS = 10  # Minimum screen brightness (10%)
MAX_BRIGHTNESS = 100  # Maximum screen brightness (100%)

def detect_bright_areas(face_region):
    """
    Analyze the brightness of the face region and adjust screen brightness accordingly.
    """
    # Convert the face region to grayscale
    gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

    # Calculate the average brightness of the face region
    avg_brightness = np.mean(gray_face)

    # Adjust screen brightness based on the average brightness
    current_brightness = sbc.get_brightness()[0]  # Get current screen brightness

    if avg_brightness < OPTIMAL_BRIGHTNESS_RANGE[0] - BUFFER_RANGE:
        # Increase brightness if the face is too dark
        new_brightness = min(current_brightness + 10, MAX_BRIGHTNESS)
        sbc.set_brightness(new_brightness)
        print(f"Increasing brightness to {new_brightness}% due to dark face.")
    elif avg_brightness > OPTIMAL_BRIGHTNESS_RANGE[1] + BUFFER_RANGE:
        # Decrease brightness if the face is too bright
        new_brightness = max(current_brightness - 10, MIN_BRIGHTNESS)
        sbc.set_brightness(new_brightness)
        print(f"Decreasing brightness to {new_brightness}% due to bright face.")

    return avg_brightness

def preprocess_frame(frame):
    """
    Preprocess the frame with grayscale conversion, normalization, Gaussian blur,
    histogram equalization, and Sobel edge detection.
    """
    # Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Normalize the intensity
    normalized_frame = cv2.normalize(gray_frame, None, 0, 255, cv2.NORM_MINMAX)

    # Apply Gaussian blur for noise reduction
    blurred_frame = cv2.GaussianBlur(normalized_frame, (5, 5), 0)

    # Apply histogram equalization to enhance contrast
    equalized_frame = cv2.equalizeHist(blurred_frame)

    # Apply Sobel edge detection
    sobel_x = cv2.Sobel(equalized_frame, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(equalized_frame, cv2.CV_64F, 0, 1, ksize=5)
    sobel_frame = cv2.magnitude(sobel_x, sobel_y)
    sobel_frame = cv2.normalize(sobel_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return equalized_frame, sobel_frame

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

            # Analyze brightness and adjust screen brightness
            avg_brightness = detect_bright_areas(face_region)

            # Draw the bounding box and display the average brightness
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Brightness: {int(avg_brightness)}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Preprocess the frame
    equalized_frame, sobel_frame = preprocess_frame(frame)

    # Display the frames in separate windows
    cv2.imshow("Original Frame", frame)
    cv2.imshow("Equalized Frame", equalized_frame)
    cv2.imshow("Sobel Edge Detection", sobel_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
