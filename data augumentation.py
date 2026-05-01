#data agumentation
import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# --- Preprocessing functions ---
def load_and_resize_image(path, size=(224, 224)):
    img = Image.open(path).convert('RGB')
    img = img.resize(size)
    return img

def apply_clahe(img):
    img_np = np.array(img)
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    clahe_img = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    return Image.fromarray(clahe_img)

def apply_unsharp_mask(img):
    img_np = np.array(img)
    blurred = cv2.GaussianBlur(img_np, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(img_np, 1.5, blurred, -0.5, 0)
    return Image.fromarray(sharpened)

def normalize_image(img):
    return img   # ❗ DO NOTHING

# --- Combined pipeline ---
def preprocess_and_save(input_root, output_root):
    os.makedirs(output_root, exist_ok=True)
    for class_name in os.listdir(input_root):
        class_input_path = os.path.join(input_root, class_name)
        class_output_path = os.path.join(output_root, class_name)
        os.makedirs(class_output_path, exist_ok=True)

        for fname in tqdm(os.listdir(class_input_path), desc=f"Processing {class_name}"):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            fpath = os.path.join(class_input_path, fname)
            img = load_and_resize_image(fpath)
            img = apply_clahe(img)
            img = apply_unsharp_mask(img)
            img = normalize_image(img)
            img.save(os.path.join(class_output_path, fname))

# Paths
train_input = '/kaggle/input/datasets/niharmnit/sorghum-weed-dataset/SorghumWeedDataset_Classification/Train'
valid_input = '/kaggle/input/datasets/niharmnit/sorghum-weed-dataset/SorghumWeedDataset_Classification/Validate'
train_output = '/kaggle/working/preprocessed_dataset/Train'
valid_output = '/kaggle/working/preprocessed_dataset/Validate'
test_input='/kaggle/input/datasets/niharmnit/sorghum-weed-dataset/SorghumWeedDataset_Classification/Test'
test_output='/kaggle/working/preprocessed_dataset/test'

# Preprocess and save
preprocess_and_save(train_input, train_output)
preprocess_and_save(valid_input, valid_output)
preprocess_and_save(test_input,test_output)
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# TRAIN TRANSFORMS (augmentation)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.RandomRotation(20),   # 🔥 reduced
    transforms.RandomHorizontalFlip(),
    
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1),   # 🔥 reduced
        shear=10                # 🔥 reduced
    ),

    transforms.ColorJitter(brightness=0.3),  # 🔥 safer
    
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_dataset = CustomImageFolder(
    train_output,
    transform=train_transform,
    apply='both'   # keep your preprocessing
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)
print(train_loader)
print(val_loader)
print(test_loader)
val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_dataset = CustomImageFolder(
    valid_output,
    transform=val_test_transform,
    apply='both'
)

test_dataset = CustomImageFolder(
    test_output,
    transform=val_test_transform,
    apply='both'
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

class_counts = [983, 1027, 1009]
classes = np.array([0, 1, 2])

# Compute class weights
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=np.repeat(classes, class_counts))
class_weight_dict = dict(zip(classes, class_weights))

print("Class Weights:", class_weight_dict)
