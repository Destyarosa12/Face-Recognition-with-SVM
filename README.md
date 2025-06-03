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
