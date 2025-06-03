import os
import pickle
import face_recognition
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def evaluate(dataset_path='dataset/test', encoding_path='encoded_faces/encodings2.pkl', model_path='encoded_faces/svm_model2.pkl'):
    with open(encoding_path, 'rb') as f:
        known_encodings, known_labels = pickle.load(f)

    with open(model_path, 'rb') as f:
        clf = pickle.load(f)

    y_true = []
    y_pred = []

    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            image = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(image)

            if encodings:
                prediction = clf.predict([encodings[0]])[0]
                y_true.append(person_name)
                y_pred.append(prediction)
            else:
                print(f"[WARNING] No face found in {img_path}")

    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred))

    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_true, y_pred))

if __name__ == '__main__':
    evaluate()
