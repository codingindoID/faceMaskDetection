# Face Mask Detection 😷

Sistem deteksi penggunaan masker berbasis **Deep Learning (CNN)** menggunakan **TensorFlow** dan **OpenCV**.

Project ini dapat:

- Melatih model klasifikasi masker
- Memprediksi gambar tunggal
- Deteksi wajah & masker secara real-time (webcam)
- Menggunakan Haar Cascade untuk face detection

---

# 📦 Dataset

Dataset berasal dari Kaggle:

https://www.kaggle.com/datasets/omkargurav/face-mask-dataset

---

## 📥 Cara Download Dataset

### Manual

1. Buka link Kaggle
2. Klik Download
3. File akan berupa `.zip`

---

## 📂 Struktur Dataset Setelah Extract

```text
Face Mask Dataset/
├── with_mask/
└── without_mask/
```

---

## 📁 Setup ke Project

Pindahkan ke:

```text
FaceMaskDetection/
└── dataset/
    ├── with_mask/
    └── without_mask/
```

⚠️ WAJIB sesuai nama folder di atas (digunakan sebagai label otomatis).

---

# 🧠 System Requirements

- Windows 10 / 11
- Python 3.12.x

Cek Python:

```bash
python --version
```

---

# 🐍 Instalasi Python

Download:
https://www.python.org/downloads/

Saat install:

✅ Centang:

```
Add Python to PATH
```

---

# 📁 Struktur Project

```text
FaceMaskDetection/
│
├── dataset/
│   ├── with_mask/
│   └── without_mask/
│
├── image/
│   ├── debug/
│   └── test/
│
├── models/
│   ├── haarcascade_frontalface_default.xml
│   └── mask_detector.h5
│
├── src/
│   ├── check_dataset.py
│   ├── train.py
│   ├── predict.py
│   └── realtime.py
│
├── venv/
├── requirements.txt
└── README.md
```

---

# ⚙️ Virtual Environment

## Buat venv

```bash
python -m venv venv
```

## Aktivasi

### CMD

```bash
venv\Scripts\activate
```

### PowerShell

```powershell
.\venv\Scripts\Activate.ps1
```

---

# 📦 Install Dependency

## Cara 1 (Recommended)

```bash
pip install -r requirements.txt
```

---

## Isi requirements.txt

```txt
tensorflow
numpy
opencv-python
scikit-learn
```

---

## Cara 2 (Manual)

```bash
pip install tensorflow numpy opencv-python scikit-learn
```

---

# 📊 Cek Dataset

```bash
python src/check_dataset.py
```

### Output:

```text
With Mask: 3725
Without Mask: 3828
```

---

# 🧠 Training Model

```bash
python src/train.py
```

---

## Arsitektur CNN

```text
Conv2D (32)
MaxPooling

Conv2D (64)
MaxPooling

Conv2D (128)
MaxPooling

Flatten
Dense (128)
Dropout (0.5)
Dense (1 - Sigmoid)
```

---

## Parameter Training

```python
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
```

---

## Output Model

```text
models/mask_detector.h5
```

---

# 🖼️ Predict Gambar

```bash
python src/predict.py
```

---

## Input Image

```text
image/test/test-withmask1.jpg
```

---

## Output

```text
pakai masker
atau
tanpa masker
```

---

# 🎥 Real-Time Detection

```bash
python src/realtime.py
```

---

## Cara Kerja

```text
Webcam → Frame → Face Detection → Crop Face → Resize → Normalize → CNN → Output
```

---

## Fitur Real-Time

- Face detection (Haar Cascade)
- Crop wajah otomatis
- Bounding box
- Label hasil prediksi
- Confidence score
- Debug image saving

---

## Output Label

```text
PAKAI MASKER (0.12)
TIDAK PAKAI MASKER (0.89)
```

---

## Keluar Program

Tekan:

```text
ESC
```

---

# 🧪 Debug Image

```text
image/debug/debug_face.jpg
```

Digunakan untuk melihat input yang masuk ke model.

---

# 🧰 Models

## Haar Cascade

```text
models/haarcascade_frontalface_default.xml
```

Untuk face detection.

---

## CNN Model

```text
models/mask_detector.h5
```

Model hasil training.

---

# 🔥 Alur Sistem

```text
1. Download dataset (Kaggle)
2. Extract dataset
3. Masukkan ke folder dataset/
4. Install requirements.txt
5. Jalankan check_dataset.py
6. Training model (train.py)
7. Model tersimpan (.h5)
8. Testing predict.py
9. Real-time detection (realtime.py)
```

---

# 🚀 Troubleshooting

## ❌ TensorFlow error install

```bash
python --version
```

Harus:

```text
Python 3.12
```

---

## ❌ Webcam tidak muncul

Ubah:

```python
cv2.VideoCapture(0)
```

menjadi:

```python
cv2.VideoCapture(1)
```

---

## ❌ Model tidak ditemukan

Jalankan ulang:

```bash
python src/train.py
```

---

## ❌ Dataset error

Pastikan:

```text
dataset/with_mask
dataset/without_mask
```

---

# 📌 Teknologi

- Python 3.12
- TensorFlow
- Keras
- OpenCV
- NumPy
- Scikit-Learn

---

# 👨‍💻 Author

Face Mask Detection Project
Computer Vision + Deep Learning Implementation

---
