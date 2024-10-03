import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance

# Initialize MediaPipe solutions
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize video capture
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Initialize face detection, face mesh, and hand tracking
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Function to count fingers
def count_fingers(hand_landmarks):
    finger_tips = [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                   mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]
    count = 0
    for tip in finger_tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            count += 1
    if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x:
        count += 1
    return count

# Function to estimate emotion based on facial landmarks
def estimate_emotion(face_landmarks):
    left_eye = face_landmarks.landmark[159]  # Left eye outer corner
    right_eye = face_landmarks.landmark[386]  # Right eye outer corner
    mouth_top = face_landmarks.landmark[13]  # Upper lip top
    mouth_bottom = face_landmarks.landmark[14]  # Lower lip bottom
    left_eyebrow = face_landmarks.landmark[70]  # Left eyebrow
    right_eyebrow = face_landmarks.landmark[300]  # Right eyebrow

    # Calculate features
    eye_openness = distance.euclidean((left_eye.x, left_eye.y), (right_eye.x, right_eye.y))
    mouth_openness = distance.euclidean((mouth_top.x, mouth_top.y), (mouth_bottom.x, mouth_bottom.y))
    eyebrow_distance = distance.euclidean((left_eyebrow.x, left_eyebrow.y), (right_eyebrow.x, right_eyebrow.y))

    # Determine emotion based on features
    if mouth_openness > 0.05 and eye_openness > 0.03:
        return "Surprised"
    elif mouth_openness < 0.02 and eyebrow_distance < 0.02:
        return "Angry"
    elif mouth_openness > 0.03 and eye_openness < 0.02:
        return "Happy"
    elif mouth_openness < 0.02 and eye_openness < 0.02:
        return "Sad"
    else:
        return "Neutral"

# Main loop for processing video feed
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the image horizontally
    image = cv2.flip(image, 1)

    # Convert the image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Process the image with MediaPipe
    face_results = face_detection.process(image)
    mesh_results = face_mesh.process(image)
    hand_results = hands.process(image)

    # Convert the image back to BGR for OpenCV
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Process face detection results
    if face_results.detections:
        for detection in face_results.detections:
            # Draw face bounding box
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(image, bbox, (0, 255, 0), 2)
            mp_drawing.draw_detection(image, detection)

    # Process face mesh results and estimate emotion
    if mesh_results.multi_face_landmarks:
        for face_landmarks in mesh_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)
            emotion = estimate_emotion(face_landmarks)
            cv2.putText(image, f"Emotion: {emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Process hand tracking results and count fingers
    total_fingers = 0
    if hand_results.multi_hand_landmarks:
        for hand_landmarks, hand_label in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Count fingers for each hand
            total_fingers += count_fingers(hand_landmarks)

            # Identify the hand (Left or Right)
            label = hand_label.classification[0].label  # 'Left' or 'Right'
            ih, iw, _ = image.shape
            x, y = int(hand_landmarks.landmark[0].x * iw), int(hand_landmarks.landmark[0].y * ih)
            cv2.putText(image, f"{label} Hand", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Display total fingers
        cv2.putText(image, f"Total Fingers: {total_fingers}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the image
    cv2.imshow('MediaPipe Face, Hand, and Emotion Detection', image)

    # Break the loop when 'q' is pressed or window is closed
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q') or cv2.getWindowProperty('MediaPipe Face, Hand, and Emotion Detection', cv2.WND_PROP_VISIBLE) < 1:
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
