import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Function to load and preprocess images
def load_images(image_paths):
    images = []
    for img_path in image_paths:
        img = image.load_img(img_path, target_size=(224, 224))  # Resize image to 224x224
        img_array = image.img_to_array(img)
        images.append(img_array)
    return np.array(images)

# Preprocess the images (normalize to [0, 1])
def preprocess_images(image_paths):
    images = load_images(image_paths)
    images = images.astype('float32') / 255.0  # Normalize pixel values to [0, 1]
    return images

# Load the dataset CSV (assumes you've already created the 'architectural_styles.csv')
df = pd.read_csv("architectural_styles.csv")

# Stratified split to maintain class balance
train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["architectural_style"], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["architectural_style"], random_state=42)

# Encode the labels into integers
label_encoder = LabelEncoder()
train_df['encoded_labels'] = label_encoder.fit_transform(train_df['architectural_style'])
val_df['encoded_labels'] = label_encoder.transform(val_df['architectural_style'])
test_df['encoded_labels'] = label_encoder.transform(test_df['architectural_style'])

# Define the target (y) for training
y_train = train_df['encoded_labels']
y_val = val_df['encoded_labels']
y_test = test_df['encoded_labels']

# Preprocess images for training and validation sets
X_train = preprocess_images(train_df["image_path"].values)
X_val = preprocess_images(val_df["image_path"].values)
X_test = preprocess_images(test_df["image_path"].values)

# Flatten the image arrays (for models like KNN)
X_train_flattened = X_train.reshape(X_train.shape[0], -1)
X_val_flattened = X_val.reshape(X_val.shape[0], -1)
X_test_flattened = X_test.reshape(X_test.shape[0], -1)

# Optional: Standardize the data for KNN
scaler = StandardScaler()
X_train_flattened = scaler.fit_transform(X_train_flattened)
X_val_flattened = scaler.transform(X_val_flattened)
X_test_flattened = scaler.transform(X_test_flattened)

# Train a K-Nearest Neighbors (KNN) model
knn = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors (k)
knn.fit(X_train_flattened, y_train)

# Predict on the test set
y_pred = knn.predict(X_test_flattened)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {accuracy:.4f}")

# Optionally, save the model for future use (with joblib)
import joblib
joblib.dump(knn, 'architectural_style_knn_model.pkl')

# Optionally, plot training & validation accuracy/loss (requires matplotlib)
import matplotlib.pyplot as plt

# KNN doesn't provide training loss/accuracy, but we can visualize prediction vs. actual for some images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(y_test[:10], label='Actual')
plt.plot(y_pred[:10], label='Predicted')
plt.title('Prediction vs Actual (KNN)')
plt.legend()

plt.show()
