# train.py (dengan evaluasi & GridSearchCV untuk SVM)
import pickle
import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Load encodings
with open('encoded_faces/encodings2.pkl', 'rb') as f:
    encodings, labels = pickle.load(f)

X = np.array(encodings)
y = np.array(labels)

# Split train-test
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# SVM dengan GridSearchCV
print("[INFO] Melakukan training dan pencarian hyperparameter...")
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

grid = GridSearchCV(SVC(probability=True), param_grid, cv=3)
grid.fit(X_train, y_train)

print(f"[INFO] Best parameters: {grid.best_params_}")

# Evaluasi
y_pred = grid.predict(X_val)
acc = accuracy_score(y_val, y_pred)
print(f"[INFO] Validation Accuracy: {acc:.2f}")
print("[INFO] Classification Report:")
print(classification_report(y_val, y_pred))

# Simpan model
os.makedirs('encoded_faces', exist_ok=True)
with open('encoded_faces/svm_model2.pkl', 'wb') as f:
    pickle.dump(grid.best_estimator_, f)
print("[INFO] Saved model to encoded_faces/svm_model2.pkl")
