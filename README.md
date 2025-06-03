# Face-Recognition-with-SVM
Proyek ini merupakan implementasi face recognition menggunakan metode **Support Vector Machine (SVM)**, dikembangkan untuk tugas mata kuliah **Computer Vision**.

## 🧠 Deskripsi Singkat

Face recognition dilakukan dengan alur sebagai berikut:
1. Ekstraksi wajah dari dataset menggunakan deteksi wajah.
2. Ekstraksi fitur menggunakan `face_recognition` (FaceNet encoding).
3. Pelatihan model klasifikasi menggunakan Support Vector Machine (SVM).
4. Evaluasi performa model menggunakan metrik klasifikasi dan confusion matrix.
5. Implementasi real-time face recognition melalui webcam.

---

## 📁 Struktur Folder
face-recognition-svm/
│
├── preprocess_encode.py # Preprocessing data wajah dan menyimpan face encoding
├── train.py # Melatih model SVM
├── evaluated.py # Mengevaluasi model dengan data uji
├── real_time.py # Real-time face recognition dengan webcam
├── requirements.txt # Dependensi Python
├── README.md # Deskripsi proyek
│
├── dataset/ # Dataset wajah (format: folder per orang)
│ ├── Person1/
│ ├── Person2/
│ └── ...
│
├── encoded_faces/
│ └── svm_model.pkl # Model SVM hasil pelatihan
│
└── evaluation_results/
└── confusion_matrix.png # Hasil evaluasi visual berupa confusion matrix

---

## 📂 Dataset

Dataset wajah digunakan dalam format:
- Setiap subfolder merepresentasikan satu orang.
- Setiap folder berisi beberapa gambar wajah orang tersebut (.jpg/.png).

📥 **Download dataset di sini**: [Google Drive - Dataset Wajah(https://drive.google.com/drive/folders/1OkNLJHocP_5kcjQxYAAroH6Fl1-YLJ8n?usp=sharing)]


---

## 💾 Model

Model yang digunakan adalah **Support Vector Machine (SVM)** dengan kernel linear atau RBF (dapat disesuaikan).  
Model yang telah dilatih disimpan dalam direktori `encoded_faces/svm_model.pkl` dan dapat langsung digunakan untuk real-time prediction.

---

## ⚙️ Cara Menjalankan

1. Pastikan semua dependensi sudah terinstall:
   ```bash
   pip install -r requirements.txt
2. Jalankan proses preprocessing data data:
   ```bash
   python preprocess_encode.py
3. Latih model SVM menggunakan data encoding (.pkl):
   ```bash
   python train.py
4. Evaluasi model:
   ```bash
   python evaluated.py
5. Jalankan sistem face recognition secara real-time:
   ```bash
   python realtime.py

## 📈 Hasil Evaluasi

Evaluasi dilakukan dengan menghitung akurasi dan menampilkan confusion matrix.  
Berikut adalah hasil confusion matrix yang menunjukkan performa model pada data uji:
Akurasi : 83%
![Confusion Matrix](results/confusion_matrix.png)

---

## 📦 Instalasi

1. Clone repositori:
   ```bash
   git clone https://github.com/username/nama-repo.git
   cd nama-repo
atau
```bash
pip install -r requirements.txt
