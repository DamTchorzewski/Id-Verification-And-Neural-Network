
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import os
import joblib

# Funkcja do przetwarzania obrazu
def preprocess_image(image):
    if len(image.shape) == 3:
        image = cv2.GaussianBlur(image, (3, 3), 0)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image = cv2.GaussianBlur(image, (3, 3), 0)

    image = cv2.resize(image, (32, 32))
    image = (image * 255).astype(np.uint8)
    image = cv2.equalizeHist(image)
    return image

# Wczytywanie danych
dataset_path = "./data/GTSRB/Final_Training/Images/"
image_data = []
labels = []

for class_id in range(0, 43):
    class_folder = f"{class_id:05d}"
    class_path = os.path.join(dataset_path, class_folder)
    annotations_file = os.path.join(class_path, f"GT-{class_folder}.csv")
    
    if not os.path.exists(annotations_file):
        print(f"Annotations file not found for class {class_id}. Skipping...")
        continue
    
    annotations = pd.read_csv(annotations_file, delimiter=";")
    
    for index, row in annotations.iterrows():
        filename = row['Filename']
        image_file = os.path.join(class_path, filename)
        image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        roi_x1, roi_y1, roi_x2, roi_y2 = row['Roi.X1'], row['Roi.Y1'], row['Roi.X2'], row['Roi.Y2']
        sign_image = image[roi_y1:roi_y2, roi_x1:roi_x2]
        resized_image = cv2.resize(sign_image, (32, 32))
        hist_eq_image = cv2.equalizeHist(resized_image)
        image_data.append(hist_eq_image)
        labels.append(class_id)

image_data = np.array(image_data)
labels = np.array(labels)

# Przetwarzanie obrazu
image_data_normalized = cv2.normalize(image_data.astype("float"), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
image_data_processed = np.stack([preprocess_image(image) for image in image_data_normalized])

# Ekstrakcja cech i klasyfikacja
image_data_flat = image_data_processed.reshape(image_data_processed.shape[0], -1)
X_train, X_test, y_train, y_test = train_test_split(image_data_flat, labels, test_size=0.2, random_state=42)

# Trenowanie klasyfikatora
classifier = SVC()
classifier.fit(X_train, y_train)

# Zapisz wytrenowany model
joblib.dump(classifier, 'traffic_sign_classifier.pkl')

# Testowanie i ocena modelu
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average=None)
recall = recall_score(y_test, y_pred, average=None)
f1 = f1_score(y_test, y_pred, average=None)

# Wyświetlenie wyników
plt.figure(figsize=(10, 6))
plt.bar(np.arange(len(precision))-0.2, precision, width=0.2, label="Precision")
plt.bar(np.arange(len(recall)), recall, width=0.2, label="Recall")
plt.bar(np.arange(len(f1))+0.2, f1, width=0.2, label="F1-score")
plt.xticks(np.arange(len(precision)), np.unique(y_test))
plt.ylabel("Score")
plt.xlabel("Classes")
plt.title("Class-wise Metrics")
plt.legend()
plt.tight_layout()
plt.show()

print(f"Accuracy: {accuracy:.4f}")
print(f"Overall Precision: {np.mean(precision):.4f}")

print("Class-wise Precision:")
for i, p in enumerate(precision):
    print(f"Class {i}: {p:.4f}")
