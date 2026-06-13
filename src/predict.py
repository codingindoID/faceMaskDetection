import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model(
    "models/mask_detector.h5"
)

img = image.load_img(
    "testimage/test-withmask1.jpg",
    target_size=(224,224)
)

img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

prediction = model.predict(img_array)

if prediction[0][0] > 0.5:
    print("tanpa masker")
else:
    print("pakai masker")