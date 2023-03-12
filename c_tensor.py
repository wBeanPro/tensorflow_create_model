import tensorflow as tf
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

# Load your images
image_dir = './dataset/'
image_size = (100, 100)

images = []
labels = []
for class_name in os.listdir(image_dir):
    class_dir = os.path.join(image_dir, class_name)
    for filename in os.listdir(class_dir):
        img = tf.keras.preprocessing.image.load_img(
            os.path.join(class_dir, filename), target_size=image_size, color_mode="rgb")
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        images.append(img_array)
        labels.append(class_name)
images = np.array(images)
print(labels)
# Convert class names to numerical labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Define your model architecture
num_classes = len(label_encoder.classes_)
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compile your model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train your model
model.fit(images, labels, epochs=10, validation_split=0.2)

# Save your model
model.save('model.h5')

# Load your trained TensorFlow model
model = tf.keras.models.load_model('model.h5')

# Convert your model to a TFLite model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save your TFLite model to a file
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
