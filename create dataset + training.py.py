import pandas as pd 
import numpy as np
import seaborn as sns
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize all images to 224x224
    transforms.ToTensor(),          # Convert images to tensors
])

# Base dataset path
base_path = '/kaggle/input/datasets/niharmnit/sorghum-weed-dataset/SorghumWeedDataset_Classification'

# Load datasets
train_dataset = datasets.ImageFolder(os.path.join(base_path, 'Train'), transform=transform)
val_dataset = datasets.ImageFolder(os.path.join(base_path, 'Validate'), transform=transform)
test_dataset = datasets.ImageFolder(os.path.join(base_path, 'Test'), transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Check class names
print("Class names:", train_dataset.classes)
import matplotlib.pyplot as plt
import torchvision
import numpy as np

# Function to display images with class names below
def show_batch(loader, class_names):
    images, labels = next(iter(loader))
    images = images[:9]
    labels = labels[:9]

    plt.figure(figsize=(8, 8))
    for i in range(len(images)):
        ax = plt.subplot(3, 3, i + 1)
        img = images[i].permute(1, 2, 0).numpy()
        plt.imshow(img)
        plt.axis("off")
        plt.text(0.5, -0.1, class_names[labels[i]],
                 size=12, ha="center", transform=ax.transAxes)
    plt.tight_layout()
    plt.show()

# Show batch from training data
show_batch(train_loader, train_dataset.classes)

