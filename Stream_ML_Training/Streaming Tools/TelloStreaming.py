import cv2
from djitellopy import Tello
import threading

# Initialize Tello drone
tello = Tello()
tello.connect()
#tello.takeoff()
tello.streamon()

# Define function to read video frames from Tello drone
def read_video():
    address = 'udp://@0.0.0.0:11111'
    cap = cv2.VideoCapture()
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Set buffer size to 1
    cap.set(cv2.CAP_PROP_FPS, 30)  # Set FPS to 30
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Set frame width to 640
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)  # Set frame height to 480
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Turn off autofocus
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # Set video codec
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Set buffer size to 1
    cap.set(cv2.CAP_PROP_CONTRAST, 50)  # Increase contrast
    cap.set(cv2.CAP_PROP_SATURATION, 50)  # Increase saturation
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 50)  # Increase brightness
    cap.set(cv2.CAP_PROP_SHARPNESS, 50)  # Increase sharpness
    cap.open(address)
    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow("Tello Stream", frame)
            cv2.waitKey(1)  # Remove waitKey() delay
        else:
            break

# Create a thread to read video frames from Tello drone
video_thread = threading.Thread(target=read_video)

# Start the thread to read video frames
video_thread.start()

# Wait for the user to exit
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cv2.destroyAllWindows()
tello.streamoff()
tello.land()
