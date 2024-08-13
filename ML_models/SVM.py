import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Define the directory where your dataset is located
dataset_dir = "C:/Users/Nithya Sri/Downloads/PCOS Detect/Model_2/Automatic_PCOS_Detection_Using_Ultrasound_images/processed_images"

# Initialize lists to store image data and corresponding labels
X = []  # Image data
y = []  # Labels

# Supported image file extensions
valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']

# Load and preprocess images
for class_folder in os.listdir(dataset_dir):
    class_path = os.path.join(dataset_dir, class_folder)
    if os.path.isdir(class_path):
        label = class_folder  # Use the folder name as the label
        for image_file in os.listdir(class_path):
            file_extension = os.path.splitext(image_file)[-1].lower()
            if file_extension in valid_extensions:
                image_path = os.path.join(class_path, image_file)
                image = cv2.imread(image_path)
                # Preprocess the image (e.g., resize to a fixed size)
                # You can also perform other preprocessing steps here
                image = cv2.resize(image, (64, 64))  # Adjust the size as needed
                X.append(image.flatten())  # Flatten the image to create a feature vector
                y.append(label)

# Convert lists to NumPy arrays
X = np.array(X)
y = np.array(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform PCA dimensionality reduction
num_components = 100  # Adjust this based on your dataset and requirements
pca = PCA(n_components=num_components)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Create and train the SVM classifier on the reduced data
svm_classifier = SVC(kernel='rbf', C=1.0)
svm_classifier.fit(X_train_pca, y_train)

# Make predictions on the test set using the reduced data
y_pred = svm_classifier.predict(X_test_pca)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot confusion matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()