import os
import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ========== STEP 1: Preprocessing Functions ==========

# CLAHE only
def apply_clahe_only(img):
    img = np.array(img)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(img)

# Unsharp masking only
def apply_unsharp_only(img):
    img = np.array(img)
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
    return Image.fromarray(sharpened)

# CLAHE + Unsharp
def apply_clahe_then_unsharp(img):
    img = apply_clahe_only(img)
    img = apply_unsharp_only(img)
    return img

# ========== STEP 2: Custom Dataset Class ==========

class CustomImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, apply='none'):
        super().__init__(root, transform=transform)
        self.apply = apply

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert('RGB')

        # Apply preprocessing
        if self.apply == 'clahe':
            image = apply_clahe_only(image)
        elif self.apply == 'unsharp':
            image = apply_unsharp_only(image)
        elif self.apply == 'both':
            image = apply_clahe_then_unsharp(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label

# ========== STEP 3: Transforms & DataLoader ==========

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Set your dataset path
base_path = '/kaggle/input/datasets/niharmnit/sorghum-weed-dataset/SorghumWeedDataset_Classification'

# Example: CLAHE-only loader
train_dataset = CustomImageFolder(os.path.join(base_path, 'Train'), transform=transform, apply='both')
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# ========== STEP 4: Visualization Function ==========

def denormalize(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    return img_tensor * std + mean

def show_batch(loader, class_names):
    images, labels = next(iter(loader))
    images = images[:9]
    labels = labels[:9]
    images = denormalize(images)

    plt.figure(figsize=(8, 8))
    for i in range(len(images)):
        ax = plt.subplot(3, 3, i + 1)
        img = images[i].permute(1, 2, 0).numpy()
        plt.imshow(np.clip(img, 0, 1))
        plt.axis("off")
        plt.text(0.5, -0.1, class_names[labels[i]],
                 size=12, ha="center", transform=ax.transAxes)
    plt.tight_layout()
    plt.show()

# ========== STEP 5: Show Preprocessed Images ==========

show_batch(train_loader, train_dataset.classes)
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# =============================
# Load and preprocess functions
# =============================

def load_image(path):
    return Image.open(path).convert('RGB')

def apply_clahe_only(img):
    img = np.array(img)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(img)

def apply_unsharp_only(img):
    img = np.array(img)
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
    return Image.fromarray(sharpened)

def apply_clahe_then_unsharp(img):
    img = apply_clahe_only(img)
    img = apply_unsharp_only(img)
    return img

# =============================
# Plot comparison
# =============================

def show_preprocessing_comparison(image_paths, max_images=5):
    num_images = min(len(image_paths), max_images)

    fig, axes = plt.subplots(num_images, 4, figsize=(16, 4 * num_images))
    if num_images == 1:
        axes = np.expand_dims(axes, 0)

    for i in range(num_images):
        img = load_image(image_paths[i])
        clahe_img = apply_clahe_only(img)
        unsharp_img = apply_unsharp_only(img)
        both_img = apply_clahe_then_unsharp(img)

        images = [img, clahe_img, unsharp_img, both_img]
        titles = ['Original', 'CLAHE', 'Unsharp', 'CLAHE + Unsharp']

        for j in range(4):
            axes[i][j].imshow(images[j])
            axes[i][j].axis('off')
            axes[i][j].set_title(titles[j], fontsize=12)

    plt.tight_layout()
    plt.show()

# =============================
# Example Usage
# =============================

# 📁 Path to one of your class folders
image_dir = '/kaggle/input/datasets/niharmnit/sorghum-weed-dataset/SorghumWeedDataset_Classification/Train/Class0_Sorghum'

# 📄 Get image paths
image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir)
               if fname.lower().endswith(('.jpg', '.png', '.jpeg'))]

# 🔍 Show comparison
show_preprocessing_comparison(image_paths, max_images=5)
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# =============================
# Step 1: Load and Resize Image
# =============================
def load_and_resize_image(path, size=(224, 224)):
    img = Image.open(path).convert('RGB')
    img = img.resize(size)
    return img

# =============================
# Step 2: Apply CLAHE
# =============================
def apply_clahe(img):
    img_np = np.array(img)
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    merged = cv2.merge((cl, a, b))
    clahe_img = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    return Image.fromarray(clahe_img)

# =============================
# Step 3: Apply Unsharp Masking
# =============================
def apply_unsharp_mask(img):
    img_np = np.array(img)
    blurred = cv2.GaussianBlur(img_np, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(img_np, 1.5, blurred, -0.5, 0)
    return Image.fromarray(sharpened)

# =============================
# Step 4: Normalize Image (0 to 1)
# =============================
def normalize_image(img):
    img_np = np.array(img).astype(np.float32) / 255.0
    return img_np

# =============================
# Step 5: Full Preprocessing Function
# =============================
def full_preprocess_pipeline(path):
    resized = load_and_resize_image(path)
    clahe_img = apply_clahe(resized)
    unsharp_img = apply_unsharp_mask(clahe_img)
    normalized = normalize_image(unsharp_img)
    return resized, clahe_img, unsharp_img, normalized

# =============================
# Step 6: Visualize Images
# =============================
def visualize_preprocessing(image_paths, max_images=5):
    num_images = min(len(image_paths), max_images)
    fig, axes = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))

    if num_images == 1:
        axes = np.expand_dims(axes, 0)

    for i in range(num_images):
        orig, clahe, final, _ = full_preprocess_pipeline(image_paths[i])

        images = [orig, clahe, final]
        titles = ['Original (Resized)', 'CLAHE', 'CLAHE + Unsharp']

        for j in range(3):
            axes[i][j].imshow(images[j])
            axes[i][j].axis('off')
            axes[i][j].set_title(titles[j], fontsize=12)

    plt.tight_layout()
    plt.show()

# =============================
# Example Usage
# =============================

# Replace this path with your dataset folder path
image_dir = '/kaggle/input/datasets/niharmnit/sorghum-weed-dataset/SorghumWeedDataset_Classification/Train/Class0_Sorghum'

# Load image paths
image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir)
               if fname.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Visualize preprocessing pipeline
visualize_preprocessing(image_paths, max_images=5)

# Optional: To use the normalized image as model input
# _, _, _, normalized_img = full_preprocess_pipeline(image_paths[0])
# Example shape: (224, 224, 3) float32
