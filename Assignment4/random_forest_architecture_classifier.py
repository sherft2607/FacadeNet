# Assuming data preprocessing has been done as discussed above.

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from tensorflow.keras.preprocessing import image  # Add this import

# Load the images
def load_images(image_paths):
    images = []
    for path in image_paths:
        img = image.load_img(path, target_size=(224, 224))  # Resize image to 224x224
        img_array = image.img_to_array(img) / 255.0  # Normalize
        images.append(img_array)
    return np.array(images)

# Load the dataset
train_df = pd.read_csv("train_data.csv")
val_df = pd.read_csv("val_data.csv")

# Preprocess images
X_train = load_images(train_df["image_path"].values)
X_val = load_images(val_df["image_path"].values)

# Flatten images for Random Forest (each image is flattened to a 1D vector)
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat = X_val.reshape(X_val.shape[0], -1)

# Label encoding
le = LabelEncoder()
y_train = le.fit_transform(train_df["architectural_style"])
y_val = le.transform(val_df["architectural_style"])

# Train Random Forest
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train_flat, y_train)

# Evaluate on validation set
y_pred = rf_clf.predict(X_val_flat)
print(f"Random Forest Accuracy: {accuracy_score(y_val, y_pred) * 100:.2f}%")
print(classification_report(y_val, y_pred, target_names=le.classes_))
