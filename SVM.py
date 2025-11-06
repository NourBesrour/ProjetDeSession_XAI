import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from skimage.feature import hog
from tqdm import tqdm

path = 'brisc2025_balanced_aug/train'
class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
img_size = (128, 128)  
features = []
labels = []

print("Extracting HOG features...")
for class_index, class_name in enumerate(class_names):
    class_folder = os.path.join(path, class_name)
    for image_file in tqdm(os.listdir(class_folder), desc=f"Processing {class_name}"):
        img_path = os.path.join(class_folder, image_file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, img_size)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        hog_features = hog(
            gray,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm='L2-Hys'
        )
        features.append(hog_features)
        labels.append(class_index)

features = np.array(features)
labels = np.array(labels)
print("Feature extraction done.")
print("Feature shape:", features.shape)

X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.15, shuffle=True, stratify=labels
)

print("Training SVM...")
svm_clf = SVC(kernel='linear', C=1.0, probability=True)
svm_clf.fit(X_train, y_train)

y_pred = svm_clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

import pickle
with open("MRI_SVM_model.pkl", "wb") as f:
    pickle.dump(svm_clf, f)
print("SVM model saved as MRI_SVM_model.pkl")
