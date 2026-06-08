# 🚶 Pedestrian Detection dengan HOG + SVM

Proyek ini mengimplementasikan **deteksi pejalan kaki** menggunakan metode **HOG (Histogram of Oriented Gradients)** dikombinasikan dengan **SVM (Support Vector Machine)** bawaan OpenCV. Tersedia dua versi: deteksi pada gambar statis dan deteksi pada video/live stream.

---

## 📸 Hasil Demo

### Deteksi pada Gambar

> *Ganti bagian ini dengan screenshot hasil deteksi pada gambar*

![Hasil Deteksi Gambar](hasil_gambar.png)

---

### Deteksi pada Video

> *Ganti bagian ini dengan GIF atau screenshot hasil deteksi pada video*

![Hasil Deteksi Video](hasil_video.gif)

---

## 📁 Struktur Proyek

```
pedestrian-detection/
│
├── img.png                  # Gambar input untuk deteksi statis
├── vid.mp4                  # Video input untuk deteksi video
│
├── detect_image.py          # Script deteksi pada gambar
├── detect_video.py          # Script deteksi pada video
│
└── README.md
```

---

## ⚙️ Instalasi

### Prasyarat

- Python 3.7+
- OpenCV
- imutils

### Install Dependensi

```bash
pip install opencv-python imutils
```

---

## 🧠 Penjelasan Kode

### 1. Deteksi pada Gambar (`detect_image.py`)

Script ini mendeteksi pejalan kaki dari sebuah file gambar statis.

```python
import cv2
import imutils

# Menginisialisasi HOG descriptor dan menggunakan detektor orang bawaan
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Membaca gambar dari file
image = cv2.imread('img.png')

# Resize gambar agar proses lebih cepat (maks lebar 400px)
image = imutils.resize(image, width=min(400, image.shape[1]))

# Mendeteksi wilayah yang mengandung pejalan kaki
(regions, _) = hog.detectMultiScale(image,
    winStride=(4, 4),
    padding=(4, 4),
    scale=1.05)

# Menggambar bounding box merah pada setiap pejalan kaki yang terdeteksi
for (x, y, w, h) in regions:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

# Menampilkan hasil
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

#### Alur Kerja

| Langkah | Keterangan |
|--------|------------|
| 1 | Inisialisasi HOG descriptor dengan model SVM default OpenCV |
| 2 | Baca gambar dari file `img.png` |
| 3 | Resize gambar untuk efisiensi komputasi |
| 4 | Jalankan `detectMultiScale()` untuk mencari wilayah pejalan kaki |
| 5 | Gambar kotak merah (bounding box) pada area yang terdeteksi |
| 6 | Tampilkan hasil ke layar |

---

### 2. Deteksi pada Video (`detect_video.py`)

Script ini mendeteksi pejalan kaki secara real-time frame per frame dari file video.

```python
import cv2
import imutils

# Inisialisasi HOG descriptor
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Membuka file video
cap = cv2.VideoCapture('vid.mp4')

while cap.isOpened():
    ret, image = cap.read()
    if ret:
        # Resize frame agar proses lebih cepat
        image = imutils.resize(image, width=min(400, image.shape[1]))

        # Deteksi pejalan kaki pada frame saat ini
        (regions, _) = hog.detectMultiScale(image,
            winStride=(4, 4),
            padding=(4, 4),
            scale=1.05)

        # Gambar bounding box pada tiap pejalan kaki
        for (x, y, w, h) in regions:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Tampilkan frame
        cv2.imshow("Image", image)

        # Tekan 'q' untuk keluar
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
```

#### Alur Kerja

| Langkah | Keterangan |
|--------|------------|
| 1 | Inisialisasi HOG descriptor dengan model SVM default OpenCV |
| 2 | Buka video menggunakan `cv2.VideoCapture()` |
| 3 | Loop membaca frame satu per satu |
| 4 | Resize tiap frame untuk efisiensi |
| 5 | Jalankan deteksi HOG pada setiap frame |
| 6 | Gambar bounding box merah pada pejalan kaki yang terdeteksi |
| 7 | Tekan `q` untuk menghentikan program |

---

## 🔍 Parameter `detectMultiScale`

| Parameter | Nilai | Keterangan |
|-----------|-------|------------|
| `winStride` | `(4, 4)` | Langkah pergeseran window dalam piksel (horizontal, vertikal). Makin kecil = lebih akurat tapi lebih lambat |
| `padding` | `(4, 4)` | Padding tambahan di sekitar window deteksi |
| `scale` | `1.05` | Faktor skala image pyramid. Makin kecil mendekati 1.0 = lebih banyak skala yang diperiksa, lebih akurat tapi lebih lambat |

---

## ▶️ Cara Menjalankan

### Deteksi Gambar

```bash
python detect_image.py
```

Pastikan file `img.png` ada di direktori yang sama.

### Deteksi Video

```bash
python detect_video.py
```

Pastikan file `vid.mp4` ada di direktori yang sama. Tekan **`q`** untuk menghentikan.

---

## 📦 Dependensi

| Library | Fungsi |
|---------|--------|
| `opencv-python` | Pemrosesan gambar, deteksi HOG, rendering bounding box |
| `imutils` | Utilitas resize gambar yang praktis |

---

## 📝 Catatan

- Metode HOG + SVM adalah pendekatan **klasik (non-deep learning)** yang ringan dan tidak membutuhkan GPU.
- Cocok digunakan untuk lingkungan dengan resource terbatas.
- Untuk akurasi yang lebih tinggi, dapat dipertimbangkan model berbasis deep learning seperti YOLO atau MobileNet SSD.

---

## 📄 Lisensi

Proyek ini dibuat untuk tujuan edukasi dan pembelajaran computer vision.
