import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

# Settings
n_samples = 5  # Number of samples per class
n_cols = 5     # Columns in grid

# Get list of classes
classes = list(class_counts.keys())

# Create figure
plt.figure(figsize=(20, 25))
plt.suptitle("Sample Images per Architectural Style", y=1.02, fontsize=16)

for idx, style in enumerate(classes):
    # Get random images from class
    class_dir = os.path.join(dataset_path, style)
    images = os.listdir(class_dir)
    selected_images = random.sample(images, min(n_samples, len(images)))
    
    # Plot samples
    for i in range(n_samples):
        plt.subplot(len(classes), n_cols, idx * n_cols + i + 1)
        img = Image.open(os.path.join(class_dir, selected_images[i]))
        plt.imshow(img)
        plt.axis('off')
        if i == 0:
            plt.title(style, fontsize=10, pad=4)
plt.tight_layout()
plt.show()