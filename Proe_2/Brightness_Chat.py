import cv2
import mediapipe as mp
import numpy as np
from screen_brightness_control import set_brightness

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

def detect_bright_areas(frame):
    """
    Detect excessively bright areas on the face and reduce screen brightness.
    Incorporates advanced image preprocessing, feature extraction, and edge detection.
    """
    global face_detection

    # Step 1: Detect the face region
    results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

            # Crop the face region
            face_region = frame[y:y + h, x:x + w]

            # Step 2: Convert face region to grayscale
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

            # Step 3: Normalize the input frame for better processing
            normalized_face = cv2.normalize(gray_face, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

            # Step 4: Apply Gaussian blur for noise reduction
            blurred_face = cv2.GaussianBlur(normalized_face, (5, 5), 0)

            # Step 5: Apply Sobel edge detection
            sobel_x = cv2.Sobel(blurred_face, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(blurred_face, cv2.CV_64F, 0, 1, ksize=3)
            sobel = cv2.magnitude(sobel_x, sobel_y)

            # Step 6: Define a threshold for brightness detection
            brightness_threshold = 255
            bright_mask = cv2.inRange(np.uint8(sobel), brightness_threshold, 255)

            # Step 7: Calculate the percentage of bright pixels
            bright_area_ratio = np.sum(bright_mask > 0) / (face_region.shape[0] * face_region.shape[1]) * 100

            # Step 8: Adjust brightness if too much light is detected
            if bright_area_ratio > 10:  # If more than 10% of the face region is too bright
                current_brightness = set_brightness()  # Get current brightness
                new_brightness = max(current_brightness - 10, 10)  # Reduce brightness by 10%, minimum 10%
                set_brightness(new_brightness)
                print(f"Detected bright areas on face. Reducing screen brightness to {new_brightness}%")

            # Highlight the bright areas on the face region
            bright_overlay = cv2.bitwise_and(face_region, face_region, mask=bright_mask)
            frame[y:y + h, x:x + w] = bright_overlay

    return frame

def adjust_brightness_based_on_light(frame):
    """
    Adjust screen brightness based on room light intensity.
    """
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray_frame)
    brightness_level = int((avg_brightness / 255) * 100)
    set_brightness(brightness_level)
    print(f"Adjusted brightness to {brightness_level}%")

# Initialize webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Detect and handle excessively bright areas
    frame = detect_bright_areas(frame)

    # Adjust screen brightness based on ambient light
    adjust_brightness_based_on_light(frame)

    # Display the processed frame
    cv2.imshow("Brightness Adjustment", frame)

    # Exit on 'q' key
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
