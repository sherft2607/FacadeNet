import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder  # Import LabelEncoder

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

# Convert labels to categorical (one-hot encoding)
y_train = to_categorical(train_df['encoded_labels'])
y_val = to_categorical(val_df['encoded_labels'])
y_test = to_categorical(test_df['encoded_labels'])

# Define the model using VGG16 pre-trained base
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze base model layers
base_model.trainable = False

# Create a Sequential model and add layers
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(25, activation='softmax')  # 25 classes for architectural styles
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    preprocess_images(train_df["image_path"].values), y_train,
    epochs=10,
    validation_data=(preprocess_images(val_df["image_path"].values), y_val),
    batch_size=32
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(
    preprocess_images(test_df["image_path"].values), y_test
)
print(f"Test accuracy: {test_accuracy}")

# Save the model for future use
model.save('architectural_style_model.h5')

# Optionally, plot training and validation accuracy/loss (requires matplotlib)
import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()
