# Real-Time Face Mask Detection

File:

```python
src/realtime.py
```

## Tujuan

Script ini digunakan untuk:

1. Mengakses webcam.
2. Mendeteksi wajah menggunakan Haar Cascade.
3. Mengambil area wajah (crop).
4. Melakukan preprocessing gambar.
5. Mengklasifikasikan wajah menggunakan model CNN.
6. Menampilkan hasil prediksi secara real-time.

---

# Alur Kerja Program

```text
Webcam
   ↓
Ambil Frame
   ↓
Deteksi Wajah
   ↓
Crop Wajah
   ↓
Resize 224x224
   ↓
Normalisasi
   ↓
Prediksi CNN
   ↓
Tampilkan Bounding Box
   ↓
Tampilkan Label
```

---

# Import Library

```python
import cv2
import tensorflow as tf
import numpy as np
```

## OpenCV

```python
import cv2
```

Digunakan untuk:

- Mengakses webcam
- Membaca gambar
- Menampilkan window
- Menggambar bounding box
- Menampilkan text

---

## TensorFlow

```python
import tensorflow as tf
```

Digunakan untuk:

- Load model CNN
- Prediksi masker

---

## NumPy

```python
import numpy as np
```

Digunakan untuk:

- Manipulasi array gambar
- Menambahkan dimensi input model

---

# Load Face Detector

```python
face_cascade = cv2.CascadeClassifier(
    "models/haarcascade_frontalface_default.xml"
)
```

## Fungsi

Membaca model deteksi wajah.

File:

```text
models/
└── haarcascade_frontalface_default.xml
```

merupakan model bawaan OpenCV.

---

## Cara Kerja Haar Cascade

Haar Cascade mencari pola:

- Mata
- Hidung
- Mulut
- Bentuk wajah

Kemudian menghasilkan koordinat wajah:

```text
x
y
width
height
```

Contoh:

```python
(100, 50, 200, 200)
```

Artinya:

```text
x = 100
y = 50
lebar = 200
tinggi = 200
```

---

# Load Model CNN

```python
model = tf.keras.models.load_model(
    "models/mask_detector.h5"
)
```

## Fungsi

Membaca model hasil training.

File:

```text
models/
└── mask_detector.h5
```

Model ini telah mempelajari dua kelas:

```text
With Mask
Without Mask
```

---

# Membuka Webcam

```python
cap = cv2.VideoCapture(0)
```

## Fungsi

Mengakses webcam.

Parameter:

```python
0
```

menunjukkan kamera utama.

Jika terdapat lebih dari satu kamera:

```python
cv2.VideoCapture(1)
```

atau

```python
cv2.VideoCapture(2)
```

---

# Loop Utama

```python
while True:
```

Program berjalan terus menerus sampai pengguna menekan tombol ESC.

---

# Membaca Frame Webcam

```python
ret, frame = cap.read()
```

## ret

Menunjukkan apakah frame berhasil dibaca.

Contoh:

```python
True
```

atau

```python
False
```

---

## frame

Berisi gambar dari webcam.

Bentuk data:

```python
(height, width, channel)
```

Contoh:

```python
(720, 1280, 3)
```

---

# Validasi Webcam

```python
if not ret:
    print("Gagal membaca webcam")
    break
```

Jika webcam gagal dibaca:

- Program berhenti.
- Menghindari error pada proses berikutnya.

---

# Konversi Ke Grayscale

```python
gray = cv2.cvtColor(
    frame,
    cv2.COLOR_BGR2GRAY
)
```

## Tujuan

Haar Cascade bekerja lebih cepat pada gambar grayscale.

---

## Sebelum

```text
3 channel
BGR
```

---

## Sesudah

```text
1 channel
Gray
```

---

# Deteksi Wajah

```python
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.3,
    minNeighbors=5,
    minSize=(60, 60)
)
```

---

## scaleFactor

```python
1.3
```

Digunakan untuk pencarian wajah pada berbagai ukuran.

Semakin kecil:

```text
lebih akurat
lebih lambat
```

Semakin besar:

```text
lebih cepat
lebih sedikit deteksi
```

---

## minNeighbors

```python
5
```

Menentukan tingkat keyakinan deteksi wajah.

Semakin besar:

```text
False detection berkurang
```

---

## minSize

```python
(60,60)
```

Wajah yang lebih kecil dari ukuran ini akan diabaikan.

---

# Loop Semua Wajah

```python
for (x, y, w, h) in faces:
```

Jika terdapat:

```text
3 wajah
```

maka loop berjalan:

```text
3 kali
```

---

# Menambahkan Padding

```python
padding = 30
```

Tujuan:

Mengambil area sekitar wajah.

---

## Sebelum

```text
[WAJAH]
```

---

## Sesudah

```text
[ AREA WAJAH + SEDIKIT BACKGROUND ]
```

Hal ini membantu model memperoleh informasi lebih banyak.

---

# Menghitung Koordinat Baru

```python
x1 = max(0, x - padding)
y1 = max(0, y - padding)

x2 = min(frame.shape[1], x + w + padding)
y2 = min(frame.shape[0], y + h + padding)
```

---

## Fungsi max()

Mencegah koordinat negatif.

Contoh:

```python
x = 10
padding = 30
```

Tanpa max():

```python
-20
```

Dengan max():

```python
0
```

---

## Fungsi min()

Mencegah koordinat keluar dari ukuran frame.

---

# Crop Wajah

```python
face = frame[y1:y2, x1:x2]
```

Mengambil area wajah dari frame.

Contoh:

```text
Frame 1280x720
```

menjadi:

```text
Face 250x250
```

---

# Validasi Crop

```python
if face.size == 0:
    continue
```

Mencegah error jika area crop kosong.

---

# Konversi BGR Ke RGB

```python
face = cv2.cvtColor(
    face,
    cv2.COLOR_BGR2RGB
)
```

---

## Mengapa?

OpenCV:

```text
BGR
```

TensorFlow:

```text
RGB
```

Agar sesuai dengan data saat training.

---

# Resize Gambar

```python
face = cv2.resize(
    face,
    (224, 224)
)
```

Ukuran harus sama dengan:

```python
IMG_SIZE = 224
```

yang digunakan saat training.

---

# Menyimpan Debug Image

```python
cv2.imwrite(
    "image/debug/debug_face.jpg",
    cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
)
```

Tujuan:

Melihat wajah yang sebenarnya masuk ke model.

File:

```text
image/debug/debug_face.jpg
```

---

# Normalisasi

```python
face = face.astype("float32") / 255.0
```

Mengubah nilai pixel:

```text
0-255
```

menjadi:

```text
0-1
```

Sama seperti proses training.

---

# Menambah Dimensi

```python
face = np.expand_dims(
    face,
    axis=0
)
```

---

## Sebelum

```python
(224,224,3)
```

---

## Sesudah

```python
(1,224,224,3)
```

TensorFlow mengharuskan adanya dimensi batch.

---

# Prediksi

```python
prediction = model.predict(
    face,
    verbose=0
)[0][0]
```

Contoh hasil:

```python
0.08
```

atau

```python
0.91
```

---

# Menampilkan Nilai Prediksi

```python
print(
    f"Prediction: {prediction:.4f}"
)
```

Contoh:

```text
Prediction: 0.0823
```

---

# Threshold Klasifikasi

```python
if prediction > 0.5:
```

---

## Jika > 0.5

```python
label = "TIDAK PAKAI MASKER"
```

Warna:

```python
(0,0,255)
```

Merah.

---

## Jika <= 0.5

```python
label = "PAKAI MASKER"
```

Warna:

```python
(0,255,0)
```

Hijau.

---

# Membuat Bounding Box

```python
cv2.rectangle(
    frame,
    (x1, y1),
    (x2, y2),
    color,
    2
)
```

Menggambar kotak pada wajah.

---

# Menampilkan Label

```python
cv2.putText(
    frame,
    label,
    (x1, y1 - 10),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    color,
    2
)
```

Menampilkan teks:

```text
PAKAI MASKER
```

atau

```text
TIDAK PAKAI MASKER
```

di atas wajah.

---

# Menampilkan Window

```python
cv2.imshow(
    "Face Mask Detection",
    frame
)
```

Menampilkan hasil deteksi secara real-time.

---

# Membaca Keyboard

```python
key = cv2.waitKey(1)
```

Menunggu input keyboard selama:

```text
1 milidetik
```

---

# Tombol ESC

```python
if key == 27:
    break
```

Kode ASCII:

```text
27 = ESC
```

Digunakan untuk keluar dari aplikasi.

---

# Menutup Resource

```python
cap.release()
```

Melepaskan webcam.

---

```python
cv2.destroyAllWindows()
```

Menutup seluruh window OpenCV.

---

# Kesimpulan

Script `realtime.py` merupakan inti dari project Face Mask Detection karena menggabungkan:

1. OpenCV untuk webcam.
2. Haar Cascade untuk deteksi wajah.
3. TensorFlow untuk klasifikasi masker.
4. Bounding box dan label real-time.

Alur lengkap:

```text
Webcam
 ↓
Frame
 ↓
Grayscale
 ↓
Face Detection
 ↓
Crop Face
 ↓
Resize 224x224
 ↓
Normalization
 ↓
CNN Prediction
 ↓
Mask / No Mask
 ↓
Display Result
```
