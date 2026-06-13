# Prediksi Gambar Tunggal

File:

```python
src/predict.py
```

## Tujuan

Melakukan pengujian model menggunakan satu gambar.

---

# Load Model

```python
model = tf.keras.models.load_model(
    "models/mask_detector.h5"
)
```

Membaca model hasil training.

---

# Membaca Gambar

```python
img = image.load_img(
    "image/test/test-withmask1.jpg",
    target_size=(224,224)
)
```

Gambar diubah menjadi:

```text
224 x 224
```

agar sesuai dengan ukuran training.

---

# Konversi Menjadi Array

```python
img_array = image.img_to_array(img)
```

TensorFlow tidak bisa memproses gambar secara langsung.

Karena itu gambar diubah menjadi array numerik.

---

# Menambah Dimensi

```python
img_array = np.expand_dims(
    img_array,
    axis=0
)
```

Dari:

```text
(224,224,3)
```

menjadi:

```text
(1,224,224,3)
```

Angka 1 menunjukkan jumlah gambar.

---

# Normalisasi

```python
img_array /= 255.0
```

Mengubah pixel:

```text
0-255
```

menjadi:

```text
0-1
```

---

# Prediksi

```python
prediction = model.predict(
    img_array
)
```

Contoh hasil:

```python
[[0.12]]
```

atau

```python
[[0.91]]
```

---

# Klasifikasi

```python
if prediction[0][0] > 0.5:
```

Threshold:

```text
0.5
```

Jika:

```text
> 0.5
```

maka:

```text
Tanpa Masker
```

Jika:

```text
<= 0.5
```

maka:

```text
Pakai Masker
```

---

# Kesimpulan

Script ini digunakan untuk menguji model sebelum digunakan pada webcam real-time.
