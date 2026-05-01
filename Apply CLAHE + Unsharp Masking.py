import os
import cv2
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch

# 🔧 Apply CLAHE + Unsharp Masking
def apply_clahe_unsharp(img):
    img = np.array(img)

    # Convert to LAB and apply CLAHE to L-channel
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Unsharp masking
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)

    return Image.fromarray(sharpened)

# 📦 Custom Dataset using ImageFolder + preprocessing
class CustomImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert('RGB')
        image = apply_clahe_unsharp(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

# 🔁 Transformations (resize, to tensor, normalize)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 📂 Dataset paths
base_path = '/kaggle/input/datasets/niharmnit/sorghum-weed-dataset/SorghumWeedDataset_Classification'
train_dataset = CustomImageFolder(os.path.join(base_path, 'Train'), transform=transform)
val_dataset = CustomImageFolder(os.path.join(base_path, 'Validate'), transform=transform)
test_dataset = CustomImageFolder(os.path.join(base_path, 'Test'), transform=transform)

# 🔃 Data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
import matplotlib.pyplot as plt

def show_batch(loader, class_names):
    images, labels = next(iter(loader))
    images = images[:9]
    labels = labels[:9]

    # Undo normalization for display
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    images = images * std + mean  # denormalize

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

# Show preprocessed training batch
show_batch(train_loader, train_dataset.classes)
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image

class CustomImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, apply='both'):
        super().__init__(root, transform=transform)
        self.apply = apply

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert('RGB')

        # Apply selected preprocessing
        if self.apply == 'clahe':
            image = apply_clahe_only(image)
        elif self.apply == 'unsharp':
            image = apply_unsharp_only(image)
        elif self.apply == 'both':
            image = apply_clahe_then_unsharp(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label
# Preprocessing transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load dataset with only CLAHE
train_dataset = CustomImageFolder(os.path.join(base_path, 'Train'), transform=transform, apply='clahe')
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
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
train_dataset = CustomImageFolder(os.path.join(base_path, 'Train'), transform=transform, apply='clahe')
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
train_dataset = CustomImageFolder(os.path.join(base_path, 'Train'), transform=transform, apply='unsharp')
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
