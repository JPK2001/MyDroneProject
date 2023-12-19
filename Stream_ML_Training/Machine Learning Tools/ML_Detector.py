import cv2
import os
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model(open(r"C:\Users\2kjee\OneDrive\Documents\PYCHARM CODES\Programmation2\DJI Tello Project\MyDroneProject\Stream_ML_Training\model.h5py"))

cap = cv2.VideoCapture()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Preprocess the image
    img = preprocess_image(frame)

    # Make a prediction using the model
    prediction = model.predict(img)

    # Get the predicted class and confidence score
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class]

    # Draw a rectangle around the hand region
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the predicted class and confidence score
    cv2.putText(frame, f'{classes[predicted_class]} ({confidence:.2f})', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Hand Sign Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
