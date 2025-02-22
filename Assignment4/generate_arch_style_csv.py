import os
import pandas as pd

# UPDATE THIS: Set your dataset directory
dataset_path = r"C:\Users\shand\Downloads\archive\g-images-dataset"  # Use raw string (r"...") for Windows paths

# Create a list to store file paths and labels
data = []

# Scan through each folder (representing architectural styles)
for style in os.listdir(dataset_path):
    style_path = os.path.join(dataset_path, style)
    
    if os.path.isdir(style_path):  # Ensure it's a folder
        images = [img for img in os.listdir(style_path) if img.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        for img in images:
            image_path = os.path.join(style_path, img)  # Full path to image
            data.append([image_path, style])  # Store image path and style

# Convert to DataFrame
df = pd.DataFrame(data, columns=["image_path", "architectural_style"])

# Save CSV file
csv_path = "architectural_styles.csv"
df.to_csv(csv_path, index=False)

print(f"CSV file created successfully! Saved as {csv_path}.")
print(f"Total images processed: {len(df)}")
