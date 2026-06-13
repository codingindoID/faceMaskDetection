# Training Model CNN

File:

```python
src/train.py
```

## Tujuan

Melatih model Deep Learning untuk membedakan:

- With Mask
- Without Mask

---

# Konfigurasi

```python
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
```

## IMG_SIZE

Ukuran gambar yang digunakan model.

```python
224 x 224 pixel
```

Semua gambar akan diubah ke ukuran ini.

---

## BATCH_SIZE

```python
BATCH_SIZE = 32
```

Artinya model membaca:

```text
32 gambar sekaligus
```

dalam satu iterasi training.

---

## EPOCHS

```python
EPOCHS = 10
```

Artinya seluruh dataset akan dipelajari sebanyak:

```text
10 kali putaran
```

---

# ImageDataGenerator

```python
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)
```

## rescale

Mengubah nilai pixel:

```text
0 - 255
```

menjadi:

```text
0 - 1
```

agar training lebih stabil.

---

## validation_split

```python
validation_split=0.2
```

Membagi dataset:

```text
80% Training
20% Validation
```

---

# Training Generator

```python
train_generator = datagen.flow_from_directory(...)
```

Fungsi ini:

1. Membaca dataset.
2. Memberi label otomatis.
3. Resize gambar.
4. Mengirim gambar ke model.

---

# Validation Generator

```python
val_generator = datagen.flow_from_directory(...)
```

Digunakan untuk mengukur performa model pada data yang belum pernah dipelajari.

---

# Arsitektur CNN

## Conv2D

```python
Conv2D(
    32,
    (3,3),
    activation="relu"
)
```

Tugas:

- Mengenali tepi wajah.
- Mengenali pola masker.
- Mengenali bentuk objek.

---

## MaxPooling

```python
MaxPooling2D()
```

Tugas:

- Mengurangi ukuran data.
- Mempercepat training.

---

## Flatten

```python
Flatten()
```

Mengubah matriks menjadi vektor.

Contoh:

```text
7x7x128
```

menjadi:

```text
6272
```

---

## Dense Layer

```python
Dense(128)
```

Berfungsi sebagai proses klasifikasi.

---

## Dropout

```python
Dropout(0.5)
```

Mematikan 50% neuron secara acak saat training.

Tujuan:

- Mengurangi overfitting.
- Membuat model lebih general.

---

## Output Layer

```python
Dense(
    1,
    activation="sigmoid"
)
```

Menghasilkan nilai:

```text
0.0 - 1.0
```

Contoh:

```text
0.12
```

Masker.

```text
0.92
```

Tidak memakai masker.

---

# Menyimpan Model

```python
model.save(
    "models/mask_detector.h5"
)
```

Model hasil training disimpan agar dapat digunakan kembali tanpa perlu training ulang.
