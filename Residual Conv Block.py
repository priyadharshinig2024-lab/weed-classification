import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# Residual Conv Block
# =========================
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        x = x + shortcut
        return F.relu(x)


# =========================
# SE Block
# =========================
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()

        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        b, c, h, w = x.size()

        se = F.adaptive_avg_pool2d(x, 1).view(b, c)
        se = F.relu(self.fc1(se))
        se = torch.sigmoid(self.fc2(se)).view(b, c, 1, 1)

        return x * se


# =========================
# Positional Encoding
# =========================
class PositionalEncoding(nn.Module):
    def __init__(self, max_len=1000, d_model=128):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x):
        return x + self.pos_embedding[:, :x.size(1), :]


# =========================
# Transformer Encoder
# =========================
class TransformerEncoder(nn.Module):
    def __init__(self, dim=128, heads=4, ff_dim=128, dropout=0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)

        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim)
        )

    def forward(self, x):
        # Attention
        x_attn = self.norm1(x)
        attn_out, _ = self.attn(x_attn, x_attn, x_attn)
        x = x + attn_out

        # FFN
        x_ff = self.norm2(x)
        x = x + self.ff(x_ff)

        return x


# =========================
# FINAL MODEL
# =========================
class CNN_SE_Transformer(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        # CNN + SE
        self.stage1 = nn.Sequential(
            ConvBlock(3, 32),
            nn.MaxPool2d(2),
            SEBlock(32)
        )

        self.stage2 = nn.Sequential(
            ConvBlock(32, 64),
            nn.MaxPool2d(2),
            SEBlock(64)
        )

        self.stage3 = nn.Sequential(
            ConvBlock(64, 128),
            nn.MaxPool2d(2),
            SEBlock(128)
        )

        # Patch embedding
        self.patch_embed = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)

        # Transformer
        self.pos_encoding = PositionalEncoding(max_len=1000, d_model=128)

        self.transformer1 = TransformerEncoder(128, 4, 128)
        self.transformer2 = TransformerEncoder(128, 4, 128)

        # Classification
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # CNN
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        # Patch embedding
        x = self.patch_embed(x)  # (B, C, H, W)

        b, c, h, w = x.size()
        x = x.view(b, c, h*w).permute(0, 2, 1)  # (B, N, C)

        # Transformer
        x = self.pos_encoding(x)
        x = self.transformer1(x)
        x = self.transformer2(x)

        # Classification
        x = x.permute(0, 2, 1)  # (B, C, N)
        x = self.pool(x).squeeze(-1)

        x = self.dropout(x)
        x = self.fc(x)

        return x
      import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model = CNN_SE_Transformer(num_classes=3)
 model = model.to(device)

print(model)
!pip install torchsummary
from torchsummary import summary

summary(model, (3, 224, 224))
import pandas as pd

def analyze_model_layers_pytorch(model):
    layer_data = []

    total_params = 0

    for name, module in model.named_modules():
        # Skip the main model container
        if len(list(module.children())) > 0:
            continue

        params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        total_params += params

        layer_data.append({
            "Layer Name": name,
            "Type": module.__class__.__name__,
            "Params": params
        })

    df = pd.DataFrame(layer_data)

    print("\n📊 Layer-wise Analysis:")
    print(df)

    print("\n🔢 Total Trainable Params:", total_params)

    return df
df = analyze_model_layers_pytorch(model)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)
