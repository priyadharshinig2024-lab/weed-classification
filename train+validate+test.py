import os

def count_images_in_dir(directory):
    class_counts = {}
    for class_name in os.listdir(directory):
        class_path = os.path.join(directory, class_name)
        if os.path.isdir(class_path):
            # Count files with image extensions
            count = len([f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))])
            class_counts[class_name] = count
    return class_counts

train_dir = '/kaggle/working/preprocessed_dataset/Train'
valid_dir = '/kaggle/working/preprocessed_dataset/Validate'
test_dir  = '/kaggle/working/preprocessed_dataset/test'

print("Train images per class:")
print(count_images_in_dir(train_dir))

print("\nValidation images per class:")
print(count_images_in_dir(valid_dir))

print("\nTest images per class:")
print(count_images_in_dir(test_dir))
import os
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import numpy as np

# =============================
# UPDATED AUGMENTATION (MATCH TRAINING)
# =============================
viz_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.3)
])

# =============================
# Visualization
# =============================
train_output = '/kaggle/working/preprocessed_dataset/Train'
classes = ['Class1_Grass', 'Class2_BroadLeafWeed', 'Class0_Sorghum']
num_images = 3

plt.figure(figsize=(12, 8))

for class_idx, class_name in enumerate(classes):
    class_dir = os.path.join(train_output, class_name)

    image_paths = [os.path.join(class_dir, fname)
                   for fname in os.listdir(class_dir)
                   if fname.lower().endswith(('png', 'jpg', 'jpeg'))][:num_images]

    for i, img_path in enumerate(image_paths):
        img = Image.open(img_path).convert('RGB')

        # CLAHE + Unsharp
        img_proc = apply_clahe_then_unsharp(img)

        # Augmentation
        aug_img = viz_transform(img_proc)

        # Plot original
        plt.subplot(len(classes)*2, num_images, class_idx*num_images*2 + i + 1)
        plt.imshow(img_proc)
        if i == 0:
            plt.ylabel(class_name + "\nOriginal", fontsize=20)
        plt.xticks([])
        plt.yticks([])

        # Plot augmented
        plt.subplot(len(classes)*2, num_images, class_idx*num_images*2 + num_images + i + 1)
        plt.imshow(aug_img)
        if i == 0:
            plt.ylabel(class_name + "\nAugmented", fontsize=20)
        plt.xticks([])
        plt.yticks([])

plt.tight_layout()
plt.show()
import os
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import numpy as np

# =============================
# SAME AUGMENTATION AS TRAINING
# =============================
viz_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.3)
])

# =============================
# Visualization
# =============================
train_output = '/kaggle/working/preprocessed_dataset/Train'
classes = ['Class1_Grass', 'Class2_BroadLeafWeed', 'Class0_Sorghum']
num_images = 3

plt.figure(figsize=(12, 8))

for class_idx, class_name in enumerate(classes):
    class_dir = os.path.join(train_output, class_name)

    image_paths = [
        os.path.join(class_dir, fname)
        for fname in os.listdir(class_dir)
        if fname.lower().endswith(('png', 'jpg', 'jpeg'))
    ][:num_images]

    for i, img_path in enumerate(image_paths):
        img = Image.open(img_path).convert('RGB')

        # Apply preprocessing
        img_proc = apply_clahe_then_unsharp(img)

        # Apply augmentation
        aug_img = viz_transform(img_proc)

        # Plot original
        plt.subplot(len(classes)*2, num_images, class_idx*num_images*2 + i + 1)
        plt.imshow(img_proc)
        if i == 0:
            plt.ylabel(class_name + "\nOriginal", fontsize=20)
        plt.xticks([])
        plt.yticks([])

        # Plot augmented
        plt.subplot(len(classes)*2, num_images, class_idx*num_images*2 + num_images + i + 1)
        plt.imshow(aug_img)
        if i == 0:
            plt.ylabel(class_name + "\nAugmented", fontsize=20)
        plt.xticks([])
        plt.yticks([])

plt.tight_layout()
plt.show()
x_batch, y_batch = next(train_generator)

print("Min:", x_batch.min())
print("Max:", x_batch.max())
from torchvision import transforms
from torch.utils.data import DataLoader

# =============================
# TRAIN TRANSFORM (WITH AUGMENTATION)
# =============================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =============================
# VALIDATION TRANSFORM (NO AUGMENTATION)
# =============================
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =============================
# DATASETS
# =============================
train_dataset = CustomImageFolder(
    train_output,
    transform=train_transform,
    apply='both'
)

val_dataset = CustomImageFolder(
    valid_output,
    transform=val_transform,
    apply='both'
)

# =============================
# DATALOADERS
# =============================
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)
import tensorflow as tf
from tensorflow.keras import layers, models

# =========================
# Residual Conv Block
# =========================
def conv_block(x, filters):
    shortcut = x

    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)

    # Match dimensions if needed
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, padding='same')(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x


# =========================
# Squeeze-and-Excitation Block
# =========================
def se_block(input_tensor, ratio=8):
    filters = input_tensor.shape[-1]

    se = layers.GlobalAveragePooling2D()(input_tensor)
    se = layers.Dense(filters // ratio, activation='relu')(se)
    se = layers.Dense(filters, activation='sigmoid')(se)
    se = layers.Reshape((1, 1, filters))(se)

    return layers.Multiply()([input_tensor, se])


# =========================
# Positional Encoding (Learnable)
# =========================
class PositionalEncoding(layers.Layer):
    def __init__(self, max_len=1000):
        super().__init__()
        self.max_len = max_len

    def build(self, input_shape):
        self.pos_embedding = self.add_weight(
            shape=(1, input_shape[1], input_shape[2]),
            initializer='random_normal',
            trainable=True
        )

    def call(self, x):
        return x + self.pos_embedding


# =========================
# Transformer Encoder
# =========================
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)

    x = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=head_size,
        dropout=dropout
    )(x, x)

    x = layers.Add()([x, inputs])

    x_ff = layers.LayerNormalization(epsilon=1e-6)(x)
    x_ff = layers.Dense(ff_dim, activation='relu')(x_ff)
    x_ff = layers.Dropout(dropout)(x_ff)
    x_ff = layers.Dense(inputs.shape[-1])(x_ff)

    return layers.Add()([x, x_ff])


# =========================
# FINAL MODEL
# =========================
def build_custom_cnn_transformer(input_shape=(224,224,3), num_classes=3):
    inputs = layers.Input(shape=input_shape)

    # ---- Stage 1 ----
    x = conv_block(inputs, 32)
    x = layers.MaxPooling2D()(x)
    x = se_block(x)

    # ---- Stage 2 ----
    x = conv_block(x, 64)
    x = layers.MaxPooling2D()(x)
    x = se_block(x)

    # ---- Stage 3 ----
    x = conv_block(x, 128)
    x = layers.MaxPooling2D()(x)
    x = se_block(x)

    # ---- Patch Embedding (better than flatten) ----
    x = layers.Conv2D(128, kernel_size=3, strides=2, padding='same')(x)

    # Shape → (B, H*W, C)
    h, w, c = x.shape[1], x.shape[2], x.shape[3]
    x = layers.Reshape((h*w, c))(x)

    # ---- Positional Encoding ----
    x = PositionalEncoding()(x)

    # ---- Transformer ----
    x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=128)
    x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=128)

    # ---- Classification ----
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs, outputs)


model = build_custom_cnn_transformer()
model.summary()
