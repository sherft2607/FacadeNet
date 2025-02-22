import os
import numpy as np
import pandas as pd
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# ================================================
# 1. Load Era Labels from CSV (UPDATED PATH)
# ================================================
dataset_path = r"C:\Users\shand\Downloads\archive\architectural-styles-dataset"
csv_path = r"H:\My Drive\Uni\2024-25\Spring\17-634 Applied Machine Learning\assignment2\arch_style_eras.csv"  # UPDATED

label_df = pd.read_csv(csv_path)
style_to_era = dict(zip(label_df["arch_style"], label_df["era"]))

# ================================================
# 2. Prepare Image Paths and Labels (WITH ERROR HANDLING)
# ================================================
image_paths = []
y = []
skipped_files = []

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            full_path = os.path.join(root, file)
            style = os.path.basename(os.path.dirname(full_path))
            
            if style in style_to_era:
                image_paths.append(full_path)
                y.append(style_to_era[style])
            else:
                skipped_files.append((style, file))

y = np.array(y)

# Report mismatches
if skipped_files:
    print(f"\nWarning: {len(skipped_files)} files skipped due to style mismatch.")
    print("First 5 mismatches:")
    for style, file in skipped_files[:5]:
        print(f"- Folder: '{style}' | File: '{file}'")

# ================================================
# 3. Extract ResNet Features (WITH BATCH PROCESSING)
# ================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model = torch.nn.Sequential(*(list(model.children())[:-1])).to(device)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(image_paths, batch_size=32):
    features = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        
        for path in batch_paths:
            img = Image.open(path).convert("RGB")
            batch_images.append(preprocess(img))
            
        batch_tensor = torch.stack(batch_images).to(device)
        with torch.no_grad():
            batch_features = model(batch_tensor).squeeze().cpu().numpy()
        features.extend(batch_features)
        
    return np.array(features)

print("\nExtracting features...")
X = extract_features(image_paths)  # Shape: (num_images, 2048)
print(f"Feature extraction complete. Shape: {X.shape}")

# ================================================
# 4. Train Regression Model (WITH DATA CHECK)
# ================================================
if len(image_paths) == 0:
    raise ValueError("No images found! Check your dataset path and CSV mappings.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"\nR² Score: {r2:.2f}")

# ================================================
# 5. Enhanced Visualization (WITH ERA CONTEXT)
# ================================================
plt.figure(figsize=(10, 7))
plt.scatter(y_test, y_pred, alpha=0.6, 
            c=np.abs(y_test - y_pred), 
            cmap='viridis', 
            label='Predictions')

# Add historical era reference lines
era_reference = {
    'Ancient': (-3000, 500),
    'Medieval': (500, 1500),
    'Modern': (1500, 2000)
}

for era, (start, end) in era_reference.items():
    plt.axhspan(start, end, alpha=0.1, label=f"{era} Era")

plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--", lw=1.5, label="Perfect Prediction")
plt.colorbar(label='Absolute Error (Years)')
plt.xlabel("Observed Era Year", fontsize=12)
plt.ylabel("Predicted Era Year", fontsize=12)
plt.title(f"Architectural Era Regression\n(R² = {r2:.2f}, {len(image_paths)} images)", fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()