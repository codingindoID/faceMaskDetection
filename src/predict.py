import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load model
model = tf.keras.models.load_model("models/mask_detector.h5")

# Load image
img = image.load_img(
    "dataset/with_mask/with_mask_1.jpg",
    target_size=(224, 224)
)

# Convert image
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# Predict
prediction = model.predict(img_array)

print("Raw prediction:", prediction)

# CASE 1: sigmoid (1 output neuron)
if prediction.shape[-1] == 1:
    if prediction[0][0] > 0.3:
        print("Tanpa masker")
    else:
        print("Pakai masker")

# CASE 2: softmax (2 output neuron)
else:
    class_names = ["pakai masker", "tanpa masker"]
    result = np.argmax(prediction)
    print(class_names[result])