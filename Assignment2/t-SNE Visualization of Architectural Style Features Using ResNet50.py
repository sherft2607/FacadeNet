import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.manifold import TSNE
from PIL import Image

# 1. Define dataset path and classes
dataset_path = r"C:\Users\shand\Downloads\archive\architectural-styles-dataset"
classes = [cls for cls in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, cls))]
classes = sorted(classes)

# 2. Load pretrained ResNet
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model = torch.nn.Sequential(*(list(model.children())[:-1]))  # Remove last layer
model.eval()

# 3. Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 4. Extract features (first 10 classes, 50 images each)
features = []
labels = []
for style_idx, style in enumerate(classes[:10]):
    class_dir = os.path.join(dataset_path, style)
    images = os.listdir(class_dir)[:50]
    for img_name in images:
        img_path = os.path.join(class_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        img_tensor = preprocess(img).unsqueeze(0)
        with torch.no_grad():
            feature = model(img_tensor).squeeze().numpy()
        features.append(feature)
        labels.append(style_idx)

# 5. Apply t-SNE and plot
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
embeddings = tsne.fit_transform(np.array(features))

plt.figure(figsize=(15, 10))
scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap='tab20', alpha=0.6)
plt.title("t-SNE Visualization of Architectural Style Features", fontsize=14)
plt.xlabel("t-SNE Dimension 1", fontsize=12)
plt.ylabel("t-SNE Dimension 2", fontsize=12)
plt.legend(handles=scatter.legend_elements()[0], 
           labels=classes[:10], 
           bbox_to_anchor=(1.05, 1), 
           loc='upper left')
plt.tight_layout()
plt.show()