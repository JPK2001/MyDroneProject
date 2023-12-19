import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# For webcam input:
cap = cv2.VideoCapture ( 0 )

with mp_holistic.Holistic (
        min_detection_confidence = 0.5,
        min_tracking_confidence = 0.5 ) as holistic, \
        mp.solutions.hands.Hands ( min_detection_confidence = 0.7, min_tracking_confidence = 0.5,
                                   max_num_hands = 2 ) as hands:
    while cap.isOpened ( ):
        success, image = cap.read ( )
        if not success:
            print ( "Ignoring empty camera frame." )
            continue

        # Convert the BGR image to RGB before processing.
        image = cv2.cvtColor ( image, cv2.COLOR_BGR2RGB )

        # Detect hands in the image
        results_hands = hands.process ( image )
        hand_landmarks = results_hands.multi_hand_landmarks

        # Extract the left and right hand landmarks.
        results_holistic = holistic.process ( image )
        left_hand_landmarks = results_holistic.left_hand_landmarks
        right_hand_landmarks = results_holistic.right_hand_landmarks

        # Draw hand landmarks
        if hand_landmarks:
            for hand in hand_landmarks:
                mp_drawing.draw_landmarks ( image, hand, mp_holistic.HAND_CONNECTIONS )
                # Get the palm center of the hand
                palm_center = (int ( hand.landmark[0].x * image.shape[1] ), int ( hand.landmark[0].y * image.shape[0] ))
                cv2.circle ( image, palm_center, 10, (0, 255, 0), -1 )

        # Draw the palm landmarks on the image.
        image = cv2.cvtColor ( image, cv2.COLOR_RGB2BGR )
        if left_hand_landmarks is not None:
            cv2.circle ( image,
                         (int ( left_hand_landmarks.landmark[mp_holistic.HandLandmark.WRIST].x * image.shape[1] ),
                          int ( left_hand_landmarks.landmark[mp_holistic.HandLandmark.WRIST].y * image.shape[0] )),
                         10, (0, 255, 0), -1 )
        if right_hand_landmarks is not None:
            cv2.circle ( image,
                         (int ( right_hand_landmarks.landmark[mp_holistic.HandLandmark.WRIST].x * image.shape[1] ),
                          int ( right_hand_landmarks.landmark[mp_holistic.HandLandmark.WRIST].y * image.shape[0] )),
                         10, (0, 255, 0), -1 )

        # Flip the image horizontally for a selfie-view display.
        cv2.imshow ( 'MediaPipe Holistic', cv2.flip ( image, 1 ) )
        if cv2.waitKey ( 5 ) & 0xFF == 27:
            break

cap.release ( )
