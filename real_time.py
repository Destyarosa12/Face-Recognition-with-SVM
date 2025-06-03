import cv2
import face_recognition
import pickle

def load_data(encoding_path='encoded_faces/encodings2.pkl', model_path='encoded_faces/svm_model2.pkl'):
    with open(encoding_path, 'rb') as f:
        _, known_labels = pickle.load(f)
    with open(model_path, 'rb') as f:
        clf = pickle.load(f)
    return clf

def main(threshold=0.6):  # Ambang batas probabilitas
    clf = load_data()

    cap = cv2.VideoCapture(0)
    print("[INFO] Starting camera...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            probs = clf.predict_proba([face_encoding])[0]
            best_class_index = probs.argmax()
            best_prob = probs[best_class_index]

            if best_prob >= threshold:
                name = clf.classes_[best_class_index]
            else:
                name = "Unknown"

            # Scale back up face locations
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw box + label
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (left + 5, bottom - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

        cv2.imshow('Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
