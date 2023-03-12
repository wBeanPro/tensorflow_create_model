import tensorflow as tf
import numpy as np
import PIL

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load and preprocess the input image.
image_size = (100, 100)
img = PIL.Image.open("Capture.PNG").resize(image_size)
img = img.convert("RGB")
input_data = np.expand_dims(img, axis=0)
input_data = (input_data.astype(np.float32) / 255.0)

# Run inference on the input image.
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)
# Print the predicted class and probability.
class_names = ["9x39mm-bp-gs", "9x39mm-pab-9-gs", "9x39mm-sp-6-gs","9x39mm-spp-gs"]  # replace with your own class names
print(np.argmax(output_data))
predicted_class = class_names[np.argmax(output_data)]
predicted_prob = np.max(output_data)
print(f"Predicted class: {predicted_class}, Probability: {predicted_prob}")
