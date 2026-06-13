import cv2
import tensorflow as tf
import numpy as np

# Load Face Detector
face_cascade = cv2.CascadeClassifier(
    "models/haarcascade_frontalface_default.xml"
)

# Load Model
model = tf.keras.models.load_model(
    "models/mask_detector.h5"
)

# Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Gagal membaca webcam")
        break

    # Grayscale untuk deteksi wajah
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # Deteksi wajah
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(60, 60)
    )

    for (x, y, w, h) in faces:

        # Tambah area sekitar wajah
        padding = 30

        x1 = max(0, x - padding)
        y1 = max(0, y - padding)

        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)

        # Crop wajah
        face = frame[y1:y2, x1:x2]

        if face.size == 0:
            continue

        # Convert BGR -> RGB
        face = cv2.cvtColor(
            face,
            cv2.COLOR_BGR2RGB
        )

        # Resize sesuai training
        face = cv2.resize(
            face,
            (224, 224)
        )

        # Simpan debug image
        cv2.imwrite(
            "image/debug/debug_face.jpg",
            cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
        )

        # Preprocessing
        face = face.astype("float32") / 255.0
        face = np.expand_dims(
            face,
            axis=0
        )

        # Prediksi
        prediction = model.predict(
            face,
            verbose=0
        )[0][0]

        print(
            f"Prediction: {prediction:.4f}"
        )

        # Threshold
        if prediction > 0.5:
            label = f"TIDAK PAKAI MASKER ({prediction:.2f})"
            color = (0, 0, 255)
        else:
            label = f"PAKAI MASKER ({prediction:.2f})"
            color = (0, 255, 0)

        # Bounding Box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        # Label
        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    cv2.imshow(
        "Face Mask Detection",
        frame
    )

    key = cv2.waitKey(1)

    # ESC
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()