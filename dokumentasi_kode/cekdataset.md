# Check Dataset

File:

```python
src/check_dataset.py
```

## Tujuan

Script ini digunakan untuk menghitung jumlah gambar pada masing-masing kelas dataset.

Struktur dataset:

```text
dataset/
├── with_mask/
└── without_mask/
```

---

## Import Library

```python
import os
```

Library `os` digunakan untuk membaca isi folder pada sistem operasi.

---

## Menghitung Jumlah Gambar Dengan Masker

```python
with_mask = len(os.listdir("dataset/with_mask"))
```

Penjelasan:

1. `os.listdir()` membaca seluruh file dalam folder.
2. Menghasilkan list file.
3. `len()` menghitung jumlah file.

Contoh:

```python
os.listdir("dataset/with_mask")
```

Output:

```python
[
    "img1.jpg",
    "img2.jpg",
    "img3.jpg"
]
```

Maka:

```python
len(...)
```

menghasilkan:

```python
3
```

---

## Menghitung Jumlah Gambar Tanpa Masker

```python
without_mask = len(os.listdir("dataset/without_mask"))
```

Prosesnya sama seperti sebelumnya.

---

## Menampilkan Hasil

```python
print("With Mask:", with_mask)
print("Without Mask:", without_mask)
```

Contoh output:

```text
With Mask: 3725
Without Mask: 3828
```

---

## Kesimpulan

Script ini digunakan sebagai validasi awal untuk memastikan dataset berhasil disiapkan sebelum proses training dimulai.
