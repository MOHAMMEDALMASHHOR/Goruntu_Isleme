import cv2
import mediapipe as mp
import numpy as np
import screen_brightness_control as sbc
import time

# Initialize MediaPipe solutions
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize hand tracking and face detection
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Virtual keyboard setup
keys = [
    ['1', '2', '3'],
    ['4', '5', '6'],
    ['7', '8', '9'],
    ['Static', '0', 'Auto']
]

# Initialize variables
mode = "Auto"
current_brightness = sbc.get_brightness()[0]
last_adjustment_time = time.time()
cooldown_period = 1  # 1 second cooldown between adjustments

def draw_keyboard(frame, selected_key=None):
    h, w, _ = frame.shape
    start_x = 10
    start_y = h - 200
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
    start_y = h - 200
    key_size = 60
    padding = 10

    for i, row in enumerate(keys):
        for j, key in enumerate(row):
            key_x = start_x + j * (key_size + padding)
            key_y = start_y + i * (key_size + padding)
            if key_x < x < key_x + key_size and key_y < y < key_y + key_size:
                return key
    return None

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    blurred = cv2.GaussianBlur(normalized, (5, 5), 0)
    
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel = np.uint8(sobel / np.max(sobel) * 255)
    
    return gray, sobel

def analyze_brightness(image, face_bbox):
    if face_bbox is None:
        return None, None

    x, y, w, h = face_bbox
    face_region = image[y:y+h, x:x+w]
    background_mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
    background_mask[y:y+h, x:x+w] = 0
    background_region = cv2.bitwise_and(image, image, mask=background_mask)

    face_gray, _ = preprocess_image(face_region)
    bg_gray, _ = preprocess_image(background_region)

    face_brightness = np.mean(face_gray)
    bg_brightness = np.mean(bg_gray[background_mask == 255])

    return face_brightness, bg_brightness

def adjust_brightness(face_brightness, bg_brightness):
    global current_brightness, last_adjustment_time
    
    if time.time() - last_adjustment_time < cooldown_period:
        return current_brightness

    face_optimal_range = (100, 180)
    bg_optimal_range = (80, 160)
    buffer = 10

    if face_brightness is None or bg_brightness is None:
        return current_brightness

    if bg_brightness < bg_optimal_range[0] - buffer:
        new_brightness = min(current_brightness + 5, 100)
    elif bg_brightness > bg_optimal_range[1] + buffer:
        new_brightness = max(current_brightness - 5, 0)
    elif face_brightness > face_optimal_range[1] + buffer:
        new_brightness = max(current_brightness - 5, 0)
    elif face_brightness < face_optimal_range[0] - buffer:
        new_brightness = min(current_brightness + 5, 100)
    else:
        new_brightness = current_brightness

    if abs(new_brightness - current_brightness) >= 5:
        current_brightness = new_brightness
        sbc.set_brightness(int(current_brightness))
        last_adjustment_time = time.time()

    return current_brightness

selected_key = None
button_pressed = False

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # Flip the image horizontally for a later selfie-view display
    image = cv2.flip(image, 1)

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
            if distance < 20 and not button_pressed:
                button_pressed = True
                if selected_key:
                    if selected_key == 'Static':
                        mode = "Manual"
                    elif selected_key == 'Auto':
                        mode = "Auto"
                    elif selected_key.isdigit():
                        mode = "Manual"
                        current_brightness = int(selected_key) * 10
                        sbc.set_brightness(current_brightness)
            elif distance >= 20:
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

            # Calculate face and background brightness
            face_brightness, bg_brightness = analyze_brightness(image, (x, y, w, h))

            # Display face and background brightness values
            cv2.putText(image, f"Face: {face_brightness:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, f"BG: {bg_brightness:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Adjust brightness if in Auto mode
    if mode == "Auto" and face_brightness is not None and bg_brightness is not None:
        current_brightness = adjust_brightness(face_brightness, bg_brightness)

    # Display current brightness and mode
    cv2.putText(image, f"Brightness: {current_brightness}%", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(image, f"Mode: {mode}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show the image
    cv2.imshow('Screen Brightness Control', image)

    # Check for 'q' key to exit
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()