import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import os

# =========================
# SETTINGS
# =========================
base_dir = "/kaggle/working/experiments"
num_runs = 3
n_classes = 3
class_names = ['Sorghum', 'Grass', 'BroadLeafWeed']

plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 20,
    'axes.labelsize': 20,
    'legend.fontsize': 18
})

# =========================
# PLOT
# =========================
fig, axes = plt.subplots(1, num_runs, figsize=(24, 7))

for run in range(1, num_runs + 1):

    # Load data
    y_true = np.load(os.path.join(base_dir, f"run_{run}", "labels.npy"))
    y_prob = np.load(os.path.join(base_dir, f"run_{run}", "probs.npy"))

    # Binarize labels
    y_true_bin = label_binarize(y_true, classes=[0,1,2])

    # Plot per class
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)

        axes[run-1].plot(fpr, tpr, label=f'{class_names[i]} (AUC={roc_auc:.3f})')

    # Diagonal
    axes[run-1].plot([0,1], [0,1], 'k--')

    axes[run-1].set_title(f"Run {run}")
    axes[run-1].set_xlabel("False Positive Rate")
    axes[run-1].set_ylabel("True Positive Rate")
    axes[run-1].grid()
    axes[run-1].legend()

plt.tight_layout()
plt.savefig("roc_all_runs.png", dpi=300, bbox_inches='tight')
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import os

# =========================
# SETTINGS
# =========================
base_dir = "/kaggle/working/experiments"
num_runs = 3
n_classes = 3
class_names = ['Sorghum', 'Grass', 'BroadLeafWeed']

plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 20,
    'axes.labelsize': 20,
    'legend.fontsize': 18
})

# =========================
# PLOT PR CURVES
# =========================
fig, axes = plt.subplots(1, num_runs, figsize=(24, 7))

for run in range(1, num_runs + 1):

    # Load data
    y_true = np.load(os.path.join(base_dir, f"run_{run}", "labels.npy"))
    y_prob = np.load(os.path.join(base_dir, f"run_{run}", "probs.npy"))

    # Binarize labels (one-vs-rest)
    y_true_bin = label_binarize(y_true, classes=[0,1,2])

    # Plot per class
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
        ap_score = average_precision_score(y_true_bin[:, i], y_prob[:, i])

        axes[run-1].plot(recall, precision,
                         label=f'{class_names[i]} (AP={ap_score:.3f})')

    axes[run-1].set_title(f"Run {run}")
    axes[run-1].set_xlabel("Recall")
    axes[run-1].set_ylabel("Precision")
    axes[run-1].grid()
    axes[run-1].legend()

plt.tight_layout()
plt.savefig("pr_all_runs.png", dpi=300, bbox_inches='tight')
plt.show()
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
run_dir = "/kaggle/working/experiments2/run_3"  # change run_2/run_3

model = CNN_SE_Transformer(num_classes=3)  # MUST match training model
model.load_state_dict(torch.load(os.path.join(run_dir, "best_model.pth"), map_location=device))
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
base_dir = "/kaggle/working/experiments2"
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
base_dir = "/kaggle/working/experiments2"
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

base_dir = "/kaggle/working/experiments2"
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
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import os

# =========================
# SETTINGS
# =========================
base_dir = "/kaggle/working/experiments2"
num_runs = 3
n_classes = 3
class_names = ['Sorghum', 'Grass', 'BroadLeafWeed']

plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 20,
    'axes.labelsize': 20,
    'legend.fontsize': 18
})

# =========================
# PLOT
# =========================
fig, axes = plt.subplots(1, num_runs, figsize=(24, 7))

for run in range(1, num_runs + 1):

    # Load data
    y_true = np.load(os.path.join(base_dir, f"run_{run}", "labels.npy"))
    y_prob = np.load(os.path.join(base_dir, f"run_{run}", "probs.npy"))

    # Binarize labels
    y_true_bin = label_binarize(y_true, classes=[0,1,2])

    # Plot per class
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)

        axes[run-1].plot(fpr, tpr, label=f'{class_names[i]} (AUC={roc_auc:.3f})')

    # Diagonal
    axes[run-1].plot([0,1], [0,1], 'k--')

    axes[run-1].set_title(f"Run {run}")
    axes[run-1].set_xlabel("False Positive Rate")
    axes[run-1].set_ylabel("True Positive Rate")
    axes[run-1].grid()
    axes[run-1].legend()

plt.tight_layout()
plt.savefig("roc_all_runs.png", dpi=300, bbox_inches='tight')
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import os

# =========================
# SETTINGS
# =========================
base_dir = "/kaggle/working/experiments2"
num_runs = 3
n_classes = 3
class_names = ['Sorghum', 'Grass', 'BroadLeafWeed']

plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 20,
    'axes.labelsize': 20,
    'legend.fontsize': 18
})

# =========================
# PLOT PR CURVES
# =========================
fig, axes = plt.subplots(1, num_runs, figsize=(24, 7))

for run in range(1, num_runs + 1):

    # Load data
    y_true = np.load(os.path.join(base_dir, f"run_{run}", "labels.npy"))
    y_prob = np.load(os.path.join(base_dir, f"run_{run}", "probs.npy"))

    # Binarize labels (one-vs-rest)
    y_true_bin = label_binarize(y_true, classes=[0,1,2])

    # Plot per class
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
        ap_score = average_precision_score(y_true_bin[:, i], y_prob[:, i])

        axes[run-1].plot(recall, precision,
                         label=f'{class_names[i]} (AP={ap_score:.3f})')

    axes[run-1].set_title(f"Run {run}")
    axes[run-1].set_xlabel("Recall")
    axes[run-1].set_ylabel("Precision")
    axes[run-1].grid()
    axes[run-1].legend()

plt.tight_layout()
plt.savefig("pr_all_runs.png", dpi=300, bbox_inches='tight')
plt.show()
import os
import torch
import numpy as np
import pandas as pd
import random
import copy
from scipy.stats import ttest_rel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# SEED
# =========================
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

# =========================
# TRAIN
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

    return total_loss/len(loader), correct/total


# =========================
# EVALUATE + SAVE PREDICTIONS
# =========================
def evaluate_full(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            probs = torch.softmax(outputs, dim=1)  # 🔥 for ROC/PR

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return (
        total_loss/len(loader),
        correct/total,
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_probs)
    )

# =========================
# MAIN EXPERIMENT
# =========================
def run_experiment(name, optimizer_type, label_smoothing, runs=3, epochs=30):

    base_dir = f"/kaggle/working/experiments3/{name}"
    os.makedirs(base_dir, exist_ok=True)

    results = []

    for run in range(1, runs+1):

        print("\n" + "="*60)
        print(f"{name} | RUN {run}")
        print("="*60)

        set_seed(42 + run)

        run_dir = os.path.join(base_dir, f"run_{run}")
        os.makedirs(run_dir, exist_ok=True)

        model = CNN_SE_Transformer(num_classes=3).to(device)

        # LOSS
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # OPTIMIZER
        if optimizer_type == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        else:
            optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4)

        # LR Scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.3, patience=3
        )

        best_val_loss = float('inf')
        history = []
        patience_counter = 0

        for epoch in range(epochs):

            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_acc, _, _, _ = evaluate_full(model, val_loader, criterion)

            current_lr = optimizer.param_groups[0]['lr']

            print(f"Epoch {epoch+1}: "
                  f"Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | "
                  f"Val Loss={val_loss:.4f}, Acc={val_acc:.4f} | "
                  f"LR={current_lr:.6f}")

            scheduler.step(val_loss)

            history.append({
                "epoch": epoch+1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc
            })

            # SAVE BEST MODEL
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = copy.deepcopy(model.state_dict())
                torch.save(best_weights, os.path.join(run_dir, "model.pth"))
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= 5:
                print("⏹ Early stopping")
                break

        # LOAD BEST MODEL
        model.load_state_dict(best_weights)

        # =========================
        # TEST + SAVE EVERYTHING
        # =========================
        test_loss, test_acc, preds, labels, probs = evaluate_full(model, test_loader, criterion)

        print(f"✅ Test Accuracy: {test_acc:.4f}")

        results.append(test_acc)

        # SAVE FILES
        pd.DataFrame(history).to_csv(os.path.join(run_dir, "history.csv"), index=False)
        np.save(os.path.join(run_dir, "predictions.npy"), preds)
        np.save(os.path.join(run_dir, "labels.npy"), labels)
        np.save(os.path.join(run_dir, "probs.npy"), probs)

        with open(os.path.join(run_dir, "results.txt"), "w") as f:
            f.write(f"Test Accuracy: {test_acc}\n")
            f.write(f"Test Loss: {test_loss}\n")

    return np.array(results)
  rms_results = run_experiment("RMSprop_LR", "rmsprop", label_smoothing=0.0)
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
base_dir = "/kaggle/working/experiments/RMSprop_LR"
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
base_dir = "/kaggle/working/experiments/RMSprop_LR"
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
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import os

# =========================
# SETTINGS
# =========================
base_dir = "/kaggle/working/experiments/RMSprop_LR"
num_runs = 3
n_classes = 3
class_names = ['Sorghum', 'Grass', 'BroadLeafWeed']

plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 20,
    'axes.labelsize': 20,
    'legend.fontsize': 18
})

# =========================
# PLOT
# =========================
fig, axes = plt.subplots(1, num_runs, figsize=(24, 7))

for run in range(1, num_runs + 1):

    # Load data
    y_true = np.load(os.path.join(base_dir, f"run_{run}", "labels.npy"))
    y_prob = np.load(os.path.join(base_dir, f"run_{run}", "probs.npy"))

    # Binarize labels
    y_true_bin = label_binarize(y_true, classes=[0,1,2])

    # Plot per class
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)

        axes[run-1].plot(fpr, tpr, label=f'{class_names[i]} (AUC={roc_auc:.3f})')

    # Diagonal
    axes[run-1].plot([0,1], [0,1], 'k--')

    axes[run-1].set_title(f"Run {run}")
    axes[run-1].set_xlabel("False Positive Rate")
    axes[run-1].set_ylabel("True Positive Rate")
    axes[run-1].grid()
    axes[run-1].legend()

plt.tight_layout()
plt.savefig("roc_all_runs.png", dpi=300, bbox_inches='tight')
plt.show()
adam_results = run_experiment("Adam_LS_LR", "adam", label_smoothing=0.05)
print("\n📊 FINAL STATISTICS")

print(f"RMSprop → Mean={rms_results.mean():.4f}, Std={rms_results.std():.4f}")
print(f"Adam → Mean={adam_results.mean():.4f}, Std={adam_results.std():.4f}")

# T-test
t_stat, p_value = ttest_rel(adam_results, rms_results)

print("\nT-test:")
print(f"T-statistic = {t_stat:.4f}")
print(f"P-value = {p_value:.6f}")
