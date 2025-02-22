import os
import numpy as np
import pandas as pd
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import matplotlib.pyplot as plt  # <-- ADDED FOR PLOTTING

# ========================
# Settings
# ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
sample_ratio = 0.2

# ========================
# 1. Load Data (Subsampled)
# ========================
dataset_path = r"C:\Users\shand\Downloads\archive\architectural-styles-dataset"
csv_path = r"H:\My Drive\Uni\2024-25\Spring\17-634 Applied Machine Learning\assignment2\arch_style_eras.csv"

label_df = pd.read_csv(csv_path)
style_to_era = dict(zip(label_df["arch_style"], label_df["era"]))

image_paths, y = [], []
for root, _, files in os.walk(dataset_path):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            full_path = os.path.join(root, file)
            style = os.path.basename(os.path.dirname(full_path))
            if style in style_to_era:
                image_paths.append(full_path)
                y.append(style_to_era[style])

image_paths, y = resample(image_paths, y, n_samples=int(len(image_paths)*sample_ratio), random_state=42)
y = np.array(y)

# ========================
# 2. Feature Extraction with Flattening
# ========================
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
model = torch.nn.Sequential(*list(model.children())[:-1]).to(device)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(112),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features_fast(image_paths):
    features = []
    for i in range(0, len(image_paths), batch_size):
        batch = [preprocess(Image.open(p).convert("RGB")) for p in image_paths[i:i+batch_size]]
        batch = torch.stack(batch).to(device)
        with torch.no_grad():
            batch_features = model(batch)
            batch_features = batch_features.view(batch_features.size(0), -1).cpu().numpy()
        features.append(batch_features)
    return np.concatenate(features, axis=0)

print("Extracting features...")
X = extract_features_fast(image_paths)

# ========================
# 3. Dimensionality Reduction
# ========================
pca = PCA(n_components=50)
X = pca.fit_transform(X)

# ========================
# 4. Train Model & Evaluate
# ========================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
regressor = SGDRegressor(max_iter=1000, tol=1e-3)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.2f} (Subsampled Data)")

# ========================
# 5. Plot Predicted vs Observed
# ========================
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, c='blue', label='Predictions')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2, label='Perfect Prediction')
plt.text(0.05, 0.9, f'R² = {r2:.2f}', transform=plt.gca().transAxes, 
         fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
plt.xlabel("Observed Era Year", fontsize=12)
plt.ylabel("Predicted Era Year", fontsize=12)
plt.title("Architectural Era Regression Results", fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ========================
# 6. Plot Accuracy vs. Model Complexity
# ========================
n_features = range(1, 51)  # From 1 to 50 features
r2_scores = []

for n in n_features:
    # Use first `n` PCA components
    X_train_n = X_train[:, :n]
    X_test_n = X_test[:, :n]
    
    # Train model
    regressor = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
    regressor.fit(X_train_n, y_train)
    
    # Predict and score
    y_pred_n = regressor.predict(X_test_n)
    r2 = r2_score(y_test, y_pred_n)
    r2_scores.append(r2)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(n_features, r2_scores, marker='o', linestyle='--', color='blue')
plt.xlabel("Number of Features (Model Complexity)", fontsize=12)
plt.ylabel("R² Score (Accuracy)", fontsize=12)
plt.title("Bias-Variance Tradeoff: Accuracy vs. Number of Features", fontsize=14)
plt.grid(alpha=0.3)
plt.xticks(np.arange(0, 51, 5))
plt.show()