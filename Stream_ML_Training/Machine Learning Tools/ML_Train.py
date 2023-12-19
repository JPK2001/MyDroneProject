import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Define paths to dataset directories
path_train = r"C:\Users\2kjee\OneDrive\Documents\PYCHARM CODES\Programmation2\DJI Tello Project\MyDroneProject\American Sign Language Letters.v1-v1.tensorflow"
path_validation = r"C:\Users\2kjee\OneDrive\Documents\PYCHARM CODES\Programmation2\DJI Tello Project\MyDroneProject\American Sign Language Letters.v1-v1.tensorflow"
path_test = r"C:\Users\2kjee\OneDrive\Documents\PYCHARM CODES\Programmation2\DJI Tello Project\MyDroneProject\American Sign Language Letters.v1-v1.tensorflow"

# Configure train and validation datasets
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    path_train,
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=(416, 416),
    batch_size=16,
    shuffle=True  # Add shuffle to improve performance
)

val_data = tf.keras.preprocessing.image_dataset_from_directory(
    path_validation,
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=(416, 416),
    batch_size=16,
    shuffle=False  # Add shuffle to improve performance
)

# Get class names from the train data
class_names = train_data.class_names
print(class_names)

# Visualize a sample of the data
for images, labels in train_data.take(1):
    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i+1)
        plt.imshow(images[i].numpy().astype('uint8'))
        plt.title(class_names[labels[i]])
        plt.axis('off')

# Define and compile the model
model = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(class_names), activation='softmax')  # Set output layer to match number of classes
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # Use sparse categorical cross-entropy for integer labels
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10  # Increase number of epochs to improve performance
)

# Test the model
test_data = tf.keras.preprocessing.image_dataset_from_directory(
    path_test,
    seed=123,
    image_size=(416, 416),
    batch_size=16
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

test_loss, test_acc = model.evaluate(test_data)
print('Test accuracy:', test_acc)