# preprocess_encode.py (versi dengan crop wajah dan resize + CLAHE)
import os
import cv2
import face_recognition
import pickle
import numpy as np

def preprocess_and_encode(dataset_path='dataset_augmented/train', output_path='encoded_faces/encodings2.pkl'):
    known_encodings = []
    known_labels = []

    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_dir):
            continue

        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            image_bgr = cv2.imread(img_path)
            if image_bgr is None:
                print(f"[WARNING] Gagal membaca {img_path}")
                continue

            # Convert ke RGB (dibutuhkan face_recognition)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            # Deteksi wajah
            boxes = face_recognition.face_locations(image_rgb)
            if not boxes:
                print(f"[WARNING] No face found in {img_path}")
                continue

            for box in boxes:
                top, right, bottom, left = box
                face = image_bgr[top:bottom, left:right]

                # Resize dan CLAHE
                face = cv2.resize(face, (160, 160))
                lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                cl = clahe.apply(l)
                limg = cv2.merge((cl, a, b))
                face_clahe = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

                # Encode wajah
                rgb_face = cv2.cvtColor(face_clahe, cv2.COLOR_BGR2RGB)
                encoding = face_recognition.face_encodings(rgb_face)
                if encoding:
                    known_encodings.append(encoding[0])
                    known_labels.append(person_name)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump((known_encodings, known_labels), f)

    print(f"[INFO] Saved {len(known_encodings)} encodings to {output_path}")

if __name__ == '__main__':
    preprocess_and_encode()