import cv2
import mediapipe as mp

# Initialize Mediapipe solutions
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_face_detection = mp.solutions.face_detection
mp_pose = mp.solutions.pose

# Initialize webcam
cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic, \
     mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection, \
     mp_pose.Pose(min_detection_confidence=0.5) as pose_detection:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert the BGR image to RGB before processing.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Perform face detection
        results = face_detection.process(image)
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(image, detection)

        # Perform pose detection
        pose_results = pose_detection.process(image)
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Perform holistic detection
        holistic_results = holistic.process(image)
        if holistic_results.face_landmarks:
            mp_drawing.draw_landmarks(image, holistic_results.face_landmarks, mp_holistic.FACE_CONNECTIONS)
        if holistic_results.left_hand_landmarks:
            mp_drawing.draw_landmarks(image, holistic_results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if holistic_results.right_hand_landmarks:
            mp_drawing.draw_landmarks(image, holistic_results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if holistic_results.pose_landmarks:
            mp_drawing.draw_landmarks(image, holistic_results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        # Flip the image horizontally for a selfie-view display.
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow('MediaPipe Detection', cv2.flip(image, 1))

        # Exit by pressing 'q'
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
