import os
import torch
import numpy as np
import pandas as pd

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# SAVE DIRECTORY
# =========================
base_dir = "/kaggle/working/experiments"
os.makedirs(base_dir, exist_ok=True)

# =========================
# TRAIN FUNCTION
# =========================
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


# =========================
# EVALUATION FUNCTION
# =========================
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), correct / total, np.array(all_preds), np.array(all_labels)


# =========================
# MAIN MULTI-RUN FUNCTION
# =========================
def run_experiments(runs=3, epochs=20):

    all_results = []

    for run in range(runs):

        print("\n" + "="*60)
        print(f"🔁 RUN {run+1}/{runs}")
        print("="*60)

        # Create folder
        run_dir = os.path.join(base_dir, f"run_{run+1}")
        os.makedirs(run_dir, exist_ok=True)

        # Model
        model = CNN_SE_Transformer(num_classes=3).to(device)

        # Loss + Optimizer
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        best_val_loss = float('inf')
        patience_counter = 0
        history = []

        # =========================
        # TRAIN LOOP
        # =========================
        for epoch in range(epochs):

            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)

            # 🔥 FULL LOG (THIS YOU WANTED)
            print(f"RUN {run+1} | Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

            # Save history
            history.append({
                "epoch": epoch+1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc
            })

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= 5:
                print("⏹ Early stopping triggered")
                break

        # Load best model
        model.load_state_dict(best_weights)

        # =========================
        # TEST
        # =========================
        test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion)

        print(f"\n📌 RUN {run+1} FINAL RESULTS")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        print("Unique Predictions:", np.unique(preds))

        # =========================
        # SAVE RESULTS
        # =========================
        torch.save(model.state_dict(), os.path.join(run_dir, "model.pth"))

        pd.DataFrame(history).to_csv(os.path.join(run_dir, "history.csv"), index=False)

        np.save(os.path.join(run_dir, "predictions.npy"), preds)
        np.save(os.path.join(run_dir, "labels.npy"), labels)

        with open(os.path.join(run_dir, "results.txt"), "w") as f:
            f.write(f"Test Accuracy: {test_acc}\n")
            f.write(f"Test Loss: {test_loss}\n")

        all_results.append(test_acc)

    # =========================
    # FINAL SUMMARY
    # =========================
    all_results = np.array(all_results)

    print("\n" + "="*60)
    print("📊 FINAL RESULTS (ALL RUNS)")
    print("="*60)
    print(f"Mean Accuracy: {all_results.mean():.4f}")
    print(f"Std Accuracy: {all_results.std():.4f}")

    return all_results


import os
import torch
import numpy as np
import pandas as pd
import random
import copy

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# SAVE DIRECTORY
# =========================
base_dir = "/kaggle/working/experiments2"
os.makedirs(base_dir, exist_ok=True)

# =========================
# SEED FUNCTION (NO LEAKAGE)
# =========================
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

# =========================
# TRAIN FUNCTION
# =========================
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total

# =========================
# EVALUATION FUNCTION
# =========================
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), correct / total, np.array(all_preds), np.array(all_labels)

# =========================
# MAIN MULTI-RUN FUNCTION
# =========================
def run_experiments(runs=3, epochs=20):

    all_results = []

    for run in range(runs):

        print("\n" + "="*60)
        print(f"🔁 RUN {run+1}/{runs}")
        print("="*60)

        # 🔥 SET SEED (NO LEAKAGE)
        set_seed(42 + run)

        # Create run folder
        run_dir = os.path.join(base_dir, f"run_{run+1}")
        os.makedirs(run_dir, exist_ok=True)

        # Model
        model = CNN_SE_Transformer(num_classes=3).to(device)

        # Loss + Optimizer
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # LR Scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.3,
            patience=3,
        )

        best_val_loss = float('inf')
        patience_counter = 0
        history = []

        # =========================
        # TRAIN LOOP
        # =========================
        for epoch in range(epochs):

            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)

            print(f"RUN {run+1} | Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

            # Update LR
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Current LR: {current_lr:.6f}")

            # Save history
            history.append({
                "epoch": epoch+1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": optimizer.param_groups[0]['lr']
            })

            # Save BEST model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = copy.deepcopy(model.state_dict())

                # 🔥 SAVE BEST MODEL IMMEDIATELY
                torch.save(best_weights, os.path.join(run_dir, "best_model.pth"))

                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= 5:
                print("⏹ Early stopping triggered")
                break

        # Load best model
        model.load_state_dict(best_weights)

        # =========================
        # TEST
        # =========================
        test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion)

        print(f"\n📌 RUN {run+1} FINAL RESULTS")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        print("Unique Predictions:", np.unique(preds))

        # =========================
        # SAVE RESULTS
        # =========================

        # Save history
        pd.DataFrame(history).to_csv(os.path.join(run_dir, "history.csv"), index=False)

        # Save predictions
        np.save(os.path.join(run_dir, "predictions.npy"), preds)
        np.save(os.path.join(run_dir, "labels.npy"), labels)

        # Save summary
        with open(os.path.join(run_dir, "results.txt"), "w") as f:
            f.write(f"Test Accuracy: {test_acc}\n")
            f.write(f"Test Loss: {test_loss}\n")

        all_results.append(test_acc)

    # =========================
    # FINAL SUMMARY
    # =========================
    all_results = np.array(all_results)

    print("\n" + "="*60)
    print("📊 FINAL RESULTS (ALL RUNS)")
    print("="*60)
    print(f"Mean Accuracy: {all_results.mean():.4f}")
    print(f"Std Accuracy: {all_results.std():.4f}")

    return all_results
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 20,
    'axes.labelsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18
})
# Load history
run_path = "/kaggle/working/experiments/run_1/history.csv"
df = pd.read_csv(run_path)

# ===== Accuracy =====
plt.figure()
plt.plot(df['epoch'], df['train_acc'], label='Train Accuracy')
plt.plot(df['epoch'], df['val_acc'], label='Validation Accuracy')

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid()
plt.show()

# ===== Loss =====
plt.figure()
plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
plt.plot(df['epoch'], df['val_loss'], label='Validation Loss')

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid()
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 20,
    'axes.labelsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18
})


# Load predictions
preds = np.load("/kaggle/working/experiments/run_1/predictions.npy")
labels = np.load("/kaggle/working/experiments/run_1/labels.npy")

# Class names (IMPORTANT: same order as dataset)
class_names = ['Sorghum', 'Grass', 'BroadLeafWeed']

cm = confusion_matrix(labels, preds)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot()

plt.title("Confusion Matrix")
plt.show()
from sklearn.metrics import classification_report

print("Classification Report:\n")

report = classification_report(
    labels,
    preds,
    target_names=class_names,
    digits=4
)

print(report)
import pandas as pd
import matplotlib.pyplot as plt
import os

# =========================
# GLOBAL FONT SETTINGS
# =========================
plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 20,
    'axes.labelsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 14
})

# =========================
# BASE DIRECTORY
# =========================
base_dir = "/kaggle/working/experiments"
num_runs = 3

# =========================
# PLOT ACCURACY (ALL RUNS)
# =========================
plt.figure()

for i in range(1, num_runs + 1):
    df = pd.read_csv(os.path.join(base_dir, f"run_{i}", "history.csv"))
    
    plt.plot(df['epoch'], df['train_acc'], linestyle='--', label=f'Run {i} Train')
    plt.plot(df['epoch'], df['val_acc'], linestyle='-', label=f'Run {i} Val')

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training & Validation Accuracy (All Runs)")
plt.legend()
plt.grid()

plt.savefig("all_runs_accuracy.png", dpi=300, bbox_inches='tight')
plt.show()


# =========================
# PLOT LOSS (ALL RUNS)
# =========================
plt.figure()

for i in range(1, num_runs + 1):
    df = pd.read_csv(os.path.join(base_dir, f"run_{i}", "history.csv"))
    
    plt.plot(df['epoch'], df['train_loss'], linestyle='--', label=f'Run {i} Train')
    plt.plot(df['epoch'], df['val_loss'], linestyle='-', label=f'Run {i} Val')

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss (All Runs)")
plt.legend()
plt.grid()

plt.savefig("all_runs_loss.png", dpi=300, bbox_inches='tight')
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

# =========================
# SETTINGS
# =========================
base_dir = "/kaggle/working/experiments"
num_runs = 3
class_names = ['Sorghum', 'Grass', 'BroadLeafWeed']

plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 20,
    'axes.labelsize': 20
})

# =========================
# FIXED PLOT
# =========================
fig, axes = plt.subplots(1, num_runs, figsize=(24, 7))  # 🔥 wider

for i in range(1, num_runs + 1):
    preds = np.load(os.path.join(base_dir, f"run_{i}", "predictions.npy"))
    labels = np.load(os.path.join(base_dir, f"run_{i}", "labels.npy"))

    cm = confusion_matrix(labels, preds)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=axes[i-1], colorbar=False)

    axes[i-1].set_title(f"Run {i}")

    # 🔥 FIX: rotate x labels
    axes[i-1].set_xticklabels(class_names, rotation=30, ha='right')

# 🔥 IMPORTANT: spacing fix
plt.subplots_adjust(wspace=0.4)  

plt.savefig("confusion_all_runs_fixed.png", dpi=300, bbox_inches='tight')
plt.show()
from sklearn.metrics import classification_report
import numpy as np
import os

base_dir = "/kaggle/working/experiments"
num_runs = 3
class_names = ['Sorghum', 'Grass', 'BroadLeafWeed']

for i in range(1, num_runs + 1):
    preds = np.load(os.path.join(base_dir, f"run_{i}", "predictions.npy"))
    labels = np.load(os.path.join(base_dir, f"run_{i}", "labels.npy"))

    print("\n" + "="*50)
    print(f"📊 CLASSIFICATION REPORT - RUN {i}")
    print("="*50)

    report = classification_report(
        labels,
        preds,
        target_names=class_names,
        digits=4
    )

    print(report)
  import numpy as np
from sklearn.metrics import precision_recall_fscore_support

all_preds = []
all_labels = []

for i in range(1, num_runs + 1):
    preds = np.load(os.path.join(base_dir, f"run_{i}", "predictions.npy"))
    labels = np.load(os.path.join(base_dir, f"run_{i}", "labels.npy"))

    all_preds.extend(preds)
    all_labels.extend(labels)

precision, recall, f1, _ = precision_recall_fscore_support(
    all_labels, all_preds, average=None
)

print("\n📊 AVERAGED METRICS ACROSS RUNS")
for i, cls in enumerate(class_names):
    print(f"{cls}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1[i]:.4f}")
  import torch
import numpy as np
import os

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# LOAD MODEL
# =========================
run_dir = "/kaggle/working/experiments/run_3"  # change run_2/run_3

model = CNN_SE_Transformer(num_classes=3)  # MUST match training model
model.load_state_dict(torch.load(os.path.join(run_dir, "model.pth"), map_location=device))
model = model.to(device)
model.eval()

# =========================
# GENERATE PROBABILITIES
# =========================
all_probs = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:   # your existing loader
        images = images.to(device)

        outputs = model(images)

        probs = torch.softmax(outputs, dim=1)  # 🔥 KEY LINE

        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())

# =========================
# SAVE
# =========================
all_probs = np.vstack(all_probs)
all_labels = np.concatenate(all_labels)

np.save(os.path.join(run_dir, "probs.npy"), all_probs)

print("✅ probs.npy saved successfully!")
print("Shape:", all_probs.shape)
