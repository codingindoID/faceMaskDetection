import tensorflow as tf
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# Konfigurasi
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10

# Data Generator
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

# Training Data
train_generator = datagen.flow_from_directory(
    "dataset",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    shuffle=True
)

# Validation Data
val_generator = datagen.flow_from_directory(
    "dataset",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=False
)

# Model CNN
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3)),

    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )
])

# Compile
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Ringkasan model
model.summary()

# Training
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# Evaluasi
print("\n=== Evaluasi Model ===")

val_generator.reset()

predictions = model.predict(val_generator)

y_pred = (predictions > 0.5).astype(int).flatten()

y_true = val_generator.classes

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=[
            "With Mask",
            "Without Mask"
        ]
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_true,
        y_pred
    )
)

# Simpan model
model.save("models/mask_detector.h5")

print("\nModel berhasil disimpan:")
print("models/mask_detector.h5")