# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 20:13:19 2025

@author: Lenovo
"""

import cv2
import mediapipe as mp
import numpy as np
import screen_brightness_control as sbc
import time
import sys  # Add this import

# Initialize MediaPipe solutions
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Check available cameras
camera_index = None
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera found at index {i}")
        camera_index = i
        cap.release()
        break

if camera_index is None:
    print("Error: No camera found. Please connect a camera and try again.")
    sys.exit()  # Use sys.exit() instead of exit()

# Initialize video capture
cap = cv2.VideoCapture(camera_index)

# Initialize hand tracking and face detection
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Virtual keyboard setup
keys = [
    ['1', '2', '3'],
    ['4', '5', '6'],
    ['7', '8', '9'],
    ['Switch', '0', '10']
]

# Initialize variables
mode = "Manual"
current_brightness = sbc.get_brightness()[0]
last_adjustment_time = 0
cooldown_period = 0.3  # Reduced cooldown for faster adjustments

# Brightness thresholds and buffer range
OPTIMAL_BRIGHTNESS_RANGE_FACE = (120, 180)  # LAB L channel values
OPTIMAL_BRIGHTNESS_RANGE_BG = (90, 150)     # LAB L channel values
BUFFER_RANGE = 5
MIN_BRIGHTNESS = 15  # Increased minimum for better visibility
MAX_BRIGHTNESS = 100

# Constants for brightness calculation
FACE_WEIGHT = 0.7
BG_WEIGHT = 0.3
IDEAL_FACE_LIGHTNESS = 150  # Mid-bright for good visibility
IDEAL_BG_LIGHTNESS = 120    # Slightly darker than face
SCALING_FACTOR = 0.3        # Increased scaling for more aggressive adjustments
SMOOTHING_FACTOR = 0.2      # Reduced smoothing for quicker responses
ADJUSTMENT_THRESHOLD = 1.0  # Lowered threshold for more sensitivity

def draw_keyboard(frame, selected_key=None):
    h, w, _ = frame.shape
    start_x = 10
    start_y = h // 2 - 150
    key_size = 60
    padding = 10

    for i, row in enumerate(keys):
        for j, key in enumerate(row):
            x = start_x + j * (key_size + padding)
            y = start_y + i * (key_size + padding)
            
            if key == selected_key:
                cv2.rectangle(frame, (x, y), (x + key_size, y + key_size), (0, 255, 0), -1)
            else:
                cv2.rectangle(frame, (x, y), (x + key_size, y + key_size), (255, 255, 255), 2)
            
            cv2.putText(frame, key, (x + 10, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def get_key_at_position(x, y):
    h, w, _ = cap.read()[1].shape
    start_x = 10
    start_y = h // 2 - 150
    key_size = 60
    padding = 10

    for i, row in enumerate(keys):
        for j, key in enumerate(row):
            key_x = start_x + j * (key_size + padding)
            key_y = start_y + i * (key_size + padding)
            if key_x < x < key_x + key_size and key_y < y < key_y + key_size:
                return key
    return None

def preprocess_region(region):
    # Convert to LAB color space for better brightness perception
    lab_region = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)
    l_channel = lab_region[:,:,0]  # L channel represents lightness
   
    
    # Apply adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
    enhanced_l = clahe.apply(l_channel)
    
    # Calculate local contrast
    sobel_x = cv2.Sobel(enhanced_l, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(enhanced_l, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sobel_x, sobel_y)
    # Normalize the frame
    normalized = cv2.normalize(lab_region, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # Apply Gaussian blur for noise reduction
    blurred = cv2.GaussianBlur(normalized, (5, 5), 0)
    normalized = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
    
    # Normalize the results
    normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    
    return normalized.astype(np.uint8), np.mean(l_channel)

def calculate_optimal_brightness(face_lightness, bg_lightness, current_brightness):
    # Calculate adjustment factors
    face_adjustment = (IDEAL_FACE_LIGHTNESS - face_lightness) * FACE_WEIGHT
    bg_adjustment = (IDEAL_BG_LIGHTNESS - bg_lightness) * BG_WEIGHT
    
    # Combined adjustment
    total_adjustment = face_adjustment + bg_adjustment
    
    # Scale adjustment to brightness percentage (0-100)
    new_brightness = current_brightness + (total_adjustment * SCALING_FACTOR)
    
    # Apply smoothing using exponential moving average
    smoothed_brightness = (SMOOTHING_FACTOR * new_brightness + 
                           (1 - SMOOTHING_FACTOR) * current_brightness)
    
    return max(MIN_BRIGHTNESS, min(MAX_BRIGHTNESS, smoothed_brightness))

def adjust_brightness(face_region, background_region):
    global current_brightness, last_adjustment_time
    
    # Check cooldown period
    if time.time() - last_adjustment_time < cooldown_period:
        return current_brightness, None, None
    
    try:
        # Process face and background regions
        face_features, face_lightness = preprocess_region(face_region)
        bg_features, bg_lightness = preprocess_region(background_region)
        
        # Calculate new brightness
        target_brightness = calculate_optimal_brightness(
            face_lightness, 
            bg_lightness, 
            current_brightness
        )
        
        # Only adjust if change is significant
        if abs(target_brightness - current_brightness) >= ADJUSTMENT_THRESHOLD:
            current_brightness = target_brightness
            sbc.set_brightness(int(current_brightness))
            last_adjustment_time = time.time()
        
        # Print lightness values for debugging
        print(f"Face Lightness: {face_lightness:.2f}, BG Lightness: {bg_lightness:.2f}, Brightness: {current_brightness}%")
        
        return current_brightness, face_lightness, bg_lightness
        
    except Exception as e:
        print(f"Error in brightness adjustment: {e}")
        return current_brightness, None, None

def preprocess_frame(frame):
    # Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive histogram equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized_frame = clahe.apply(gray_frame)
    
    # Calculate Sobel edge detection
    sobel_x = cv2.Sobel(equalized_frame, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(equalized_frame, cv2.CV_64F, 0, 1, ksize=5)
    sobel_frame = cv2.magnitude(sobel_x, sobel_y)
    sobel_frame = cv2.normalize(sobel_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return equalized_frame, sobel_frame

# Main loop
selected_key = None
button_pressed = False
press_start_time = 0
press_duration = 0.5  # Duration in seconds to consider a press

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Error: Could not read frame from camera.")
        break

    # Flip the image horizontally for a later selfie-view display
    image = cv2.flip(image, 1)

    # Preprocess the frame for equalization and Sobel edge detection
    equalized_frame, sobel_frame = preprocess_frame(image)

    # Convert the BGR image to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process hand tracking
    hand_results = hands.process(rgb_image)

    # Process face detection
    face_results = face_detection.process(rgb_image)

    # Draw virtual keyboard
    draw_keyboard(image, selected_key)

    # Handle hand tracking results
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get index finger tip and thumb tip coordinates
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            h, w, _ = image.shape
            index_x, index_y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)

            # Check if finger is pointing at a key
            key = get_key_at_position(index_x, index_y)
            if key:
                selected_key = key

            # Check if finger and thumb are closed (simulating a button press)
            distance = np.sqrt((index_x - thumb_x)**2 + (index_y - thumb_y)**2)
            if distance < 20:
                if not button_pressed:
                    press_start_time = time.time()
                    button_pressed = True
                elif time.time() - press_start_time >= press_duration:
                    if selected_key:
                        if selected_key == 'Switch':
                            mode = "Manual" if mode == "Auto" else "Auto"
                        elif selected_key == '10':
                            mode = "Manual"
                            current_brightness = 100
                            sbc.set_brightness(current_brightness)
                        elif selected_key.isdigit():
                            mode = "Manual"
                            current_brightness = int(selected_key) * 10
                            sbc.set_brightness(current_brightness)
                    button_pressed = False
            else:
                button_pressed = False

    # Handle face detection results
    face_brightness = None
    bg_brightness = None

    if face_results.detections:
        for detection in face_results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

            # Draw face bounding box
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Extract face and background regions
            face_region = image[y:y+h, x:x+w]
            background_region = np.copy(image)
            cv2.rectangle(background_region, (x, y), (x+w, y+h), (0, 0, 0), -1)

            # Adjust brightness if in Auto mode
            if mode == "Auto":
                current_brightness, face_brightness, bg_brightness = adjust_brightness(face_region, background_region)

            # Display face and background brightness values
            if face_brightness is not None and bg_brightness is not None:
                cv2.putText(image, f"Face: {face_brightness:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(image, f"BG: {bg_brightness:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display current brightness and mode
    cv2.putText(image, f"Brightness: {current_brightness}%", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(image, f"Mode: {mode}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show the images
    cv2.imshow('Original Frame', image)
    cv2.imshow("Equalized Frame", equalized_frame)
    cv2.imshow("Sobel Edge Detection", sobel_frame)

    # Check for 'q' key to exit
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()