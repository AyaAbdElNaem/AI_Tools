"""
====================================================================
Garbage Classification - Training with ConvNeXt (torchvision)
====================================================================

WHAT THIS SCRIPT DOES
----------------------
1. Loads images ONLY from:  DATASET_ROOT / "original" / <class_name> / *.jpg
   (10 classes, already balanced by the augmentation script).
2. Splits the data into train (80%) / validation (10%) / test (10%),
   STRATIFIED so every split keeps the same class proportions.
3. Builds a ConvNeXt (tiny or small) model pretrained on ImageNet,
   with its classifier head replaced to output 10 classes.
4. Trains with:
     - CrossEntropyLoss (optional label smoothing)
     - AdamW optimizer
     - Linear warm-up followed by Cosine Annealing LR schedule
     - tqdm progress bars per epoch (train + validation)
     - Best checkpoint saved automatically (highest validation accuracy)
5. After training, loads the best checkpoint and evaluates on the
   held-out TEST set:
     - Accuracy, Precision/Recall/F1 (macro & weighted)
     - Full per-class classification report (printed + saved as .csv)
6. Saves 3 PNG charts:
     - training_curves.png   (loss & accuracy vs. epoch)
     - confusion_matrix.png  (normalized heatmap)
     - per_class_f1.png      (bar chart of per-class F1 scores)

HOW TO RUN
----------
1. Activate your virtual environment (see README_AR.md).
2. pip install -r requirements.txt
3. Edit the CONFIGURATION block below (DATASET_ROOT, batch size, etc.)
4. python train_convnext.py
"""

import os
import copy
import random
import time
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
from torchvision.datasets import ImageFolder

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ============================================================
# CONFIGURATION - edit these values to match your setup
# ============================================================

# Root folder that contains original/, standardized_256/, standardized_384/
DATASET_ROOT =r"C:\Users\Admin\Downloads\archive"

# We only train on the "original" subfolder, as required
DATA_DIR = os.path.join(DATASET_ROOT, "original")

# Where to save checkpoints, charts, and reports
OUTPUT_DIR = "./training_outputs"

# Model choice: "convnext_tiny" or "convnext_small" (torchvision)
MODEL_NAME = "convnext_tiny"

NUM_CLASSES = 10
IMAGE_SIZE = 224          # standard ConvNeXt input size
BATCH_SIZE = 32
NUM_WORKERS = 4           # set to 0 on Windows if you hit DataLoader issues

NUM_EPOCHS = 30
WARMUP_EPOCHS = 3         # linear warm-up epochs before cosine annealing kicks in
BASE_LR = 3e-4
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.1     # set to 0.0 to disable

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1          # must sum to 1.0 with the above

RANDOM_SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)


# ============================================================
# DATA: transforms, stratified split, DataLoaders
# ============================================================

# ImageNet normalization stats (required for pretrained ConvNeXt weights)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Training transform: light augmentation on top of the already-augmented
# dataset (mostly geometric/color jitter, kept mild since heavy augmentation
# was already applied when the dataset was balanced).
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Validation / test transform: deterministic, no augmentation
eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class TransformSubset(Dataset):
    """
    Wraps a Subset of an ImageFolder so we can apply a DIFFERENT transform
    to train vs. validation vs. test, even though they all come from the
    same underlying ImageFolder object (which only has one `.transform`).
    """

    def __init__(self, subset: Subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def build_datasets():
    """
    Loads the ImageFolder (no transform yet - PIL images only), then performs
    a STRATIFIED split into train/val/test indices so class proportions are
    preserved across all three splits.
    """
    assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6, \
        "TRAIN_RATIO + VAL_RATIO + TEST_RATIO must sum to 1.0"

    base_dataset = ImageFolder(DATA_DIR, transform=None)
    class_names = base_dataset.classes
    targets = np.array(base_dataset.targets)

    assert len(class_names) == NUM_CLASSES, (
        f"Found {len(class_names)} classes in '{DATA_DIR}', "
        f"but NUM_CLASSES is set to {NUM_CLASSES}. Update NUM_CLASSES "
        f"or check your dataset folder."
    )

    all_indices = np.arange(len(base_dataset))

    # First split: train vs. (val + test)
    train_idx, temp_idx = train_test_split(
        all_indices,
        train_size=TRAIN_RATIO,
        stratify=targets,
        random_state=RANDOM_SEED,
    )

    # Second split: val vs. test, taken proportionally out of temp_idx
    val_share_of_temp = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    temp_targets = targets[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx,
        train_size=val_share_of_temp,
        stratify=temp_targets,
        random_state=RANDOM_SEED,
    )

    train_subset = Subset(base_dataset, train_idx)
    val_subset = Subset(base_dataset, val_idx)
    test_subset = Subset(base_dataset, test_idx)

    train_dataset = TransformSubset(train_subset, train_transform)
    val_dataset = TransformSubset(val_subset, eval_transform)
    test_dataset = TransformSubset(test_subset, eval_transform)

    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Total images: {len(base_dataset)}")
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset, class_names


# ============================================================
# MODEL: ConvNeXt with a custom 10-class head
# ============================================================

def build_model(model_name: str, num_classes: int):
    """
    Loads a torchvision ConvNeXt model pretrained on ImageNet and replaces
    the final classifier layer to output `num_classes` logits.
    """
    if model_name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        model = models.convnext_tiny(weights=weights)
    elif model_name == "convnext_small":
        weights = models.ConvNeXt_Small_Weights.IMAGENET1K_V1
        model = models.convnext_small(weights=weights)
    else:
        raise ValueError(f"Unsupported MODEL_NAME: {model_name}")

    # torchvision ConvNeXt classifier is nn.Sequential:
    #   [0] LayerNorm2d, [1] Flatten, [2] Linear(in_features, 1000)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)

    return model


# ============================================================
# LR SCHEDULE: linear warm-up + cosine annealing
# ============================================================

def build_scheduler(optimizer, warmup_epochs, total_epochs, steps_per_epoch):
    """
    Builds a scheduler that linearly warms up the LR for `warmup_epochs`,
    then follows a cosine annealing curve down to ~0 for the remaining
    epochs. Stepped once per training batch (not per epoch) for smoothness.
    """
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(progress, 1.0)
        cosine_factor = 0.5 * (1.0 + np.cos(np.pi * progress))
        return cosine_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================
# TRAIN / VALIDATION LOOPS
# ============================================================

def run_one_epoch(model, dataloader, criterion, optimizer, scheduler, device, train: bool):
    """
    Runs one epoch of training (train=True) or validation (train=False).
    Returns (average_loss, accuracy) for the epoch.
    """
    model.train() if train else model.eval()

    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    phase_name = "Train" if train else "Val  "
    progress_bar = tqdm(dataloader, desc=phase_name, leave=False)

    with torch.set_grad_enabled(train):
        for images, labels in progress_bar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            preds = outputs.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            total_samples += batch_size

            progress_bar.set_postfix(
                loss=f"{running_loss / total_samples:.4f}",
                acc=f"{running_correct / total_samples:.4f}",
            )

    avg_loss = running_loss / total_samples
    accuracy = running_correct / total_samples
    return avg_loss, accuracy


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device,
                 num_epochs, output_dir):
    """
    Full training loop across all epochs. Tracks history for plotting and
    saves the best model checkpoint (highest validation accuracy).
    """
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
    }

    best_val_acc = 0.0
    best_model_weights = copy.deepcopy(model.state_dict())
    best_checkpoint_path = os.path.join(output_dir, "best_convnext_model.pth")

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        print(f"\nEpoch {epoch}/{num_epochs}")

        train_loss, train_acc = run_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, train=True
        )
        val_loss, val_acc = run_one_epoch(
            model, val_loader, criterion, optimizer=None, scheduler=None, device=device, train=False
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - start_time
        print(
            f"  Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}  ||  "
            f"Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f}  ({elapsed:.1f}s)"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_weights = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": best_model_weights,
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                },
                best_checkpoint_path,
            )
            print(f"  -> New best model saved (val_acc={val_acc:.4f}) -> {best_checkpoint_path}")

    model.load_state_dict(best_model_weights)
    print(f"\nBest validation accuracy achieved: {best_val_acc:.4f}")
    return model, history


# ============================================================
# EVALUATION ON TEST SET
# ============================================================

def evaluate_on_test_set(model, test_loader, device, class_names, output_dir):
    """
    Runs the final trained model on the held-out test set and computes
    all classification metrics + saves the confusion matrix / F1 chart.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing", leave=False):
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # ---- Overall metrics ----
    overall_acc = accuracy_score(all_labels, all_preds)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )

    print("\n" + "=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    print(f"Overall Accuracy       : {overall_acc:.4f}")
    print(f"Precision (macro)      : {precision_macro:.4f}")
    print(f"Recall    (macro)      : {recall_macro:.4f}")
    print(f"F1-score  (macro)      : {f1_macro:.4f}")
    print(f"Precision (weighted)   : {precision_weighted:.4f}")
    print(f"Recall    (weighted)   : {recall_weighted:.4f}")
    print(f"F1-score  (weighted)   : {f1_weighted:.4f}")

    # ---- Per-class classification report ----
    report_dict = classification_report(
        all_labels, all_preds, target_names=class_names, output_dict=True, zero_division=0
    )
    report_str = classification_report(
        all_labels, all_preds, target_names=class_names, zero_division=0
    )
    print("\nDetailed Classification Report:")
    print(report_str)

    # Save report as CSV for later reference
    import csv
    report_csv_path = os.path.join(output_dir, "classification_report.csv")
    with open(report_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "precision", "recall", "f1-score", "support"])
        for class_name in class_names:
            row = report_dict[class_name]
            writer.writerow([
                class_name, row["precision"], row["recall"], row["f1-score"], row["support"]
            ])
        writer.writerow([
            "macro avg",
            report_dict["macro avg"]["precision"],
            report_dict["macro avg"]["recall"],
            report_dict["macro avg"]["f1-score"],
            report_dict["macro avg"]["support"],
        ])
        writer.writerow([
            "weighted avg",
            report_dict["weighted avg"]["precision"],
            report_dict["weighted avg"]["recall"],
            report_dict["weighted avg"]["f1-score"],
            report_dict["weighted avg"]["support"],
        ])
    print(f"\nSaved detailed report -> {report_csv_path}")

    # ---- Confusion matrix (normalized) ----
    cm = confusion_matrix(all_labels, all_preds, normalize="true")
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        cbar_kws={"label": "Proportion"},
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Normalized Confusion Matrix - Test Set")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"Saved confusion matrix chart -> {cm_path}")

    # ---- Per-class F1 bar chart ----
    per_class_f1 = [report_dict[name]["f1-score"] for name in class_names]
    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_names, per_class_f1, color="steelblue")
    plt.ylim(0, 1.0)
    plt.ylabel("F1-score")
    plt.title("Per-Class F1-Score - Test Set")
    plt.xticks(rotation=45, ha="right")
    for bar, score in zip(bars, per_class_f1):
        plt.text(
            bar.get_x() + bar.get_width() / 2, score + 0.01,
            f"{score:.2f}", ha="center", va="bottom", fontsize=9,
        )
    plt.tight_layout()
    f1_path = os.path.join(output_dir, "per_class_f1.png")
    plt.savefig(f1_path, dpi=200)
    plt.close()
    print(f"Saved per-class F1 chart -> {f1_path}")

    return {
        "overall_accuracy": overall_acc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
    }


# ============================================================
# PLOTTING: training curves
# ============================================================

def plot_training_curves(history, output_dir):
    epochs_range = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs_range, history["train_loss"], label="Train Loss", marker="o")
    axes[0].plot(epochs_range, history["val_loss"], label="Validation Loss", marker="o")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training vs. Validation Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs_range, history["train_acc"], label="Train Accuracy", marker="o")
    axes[1].plot(epochs_range, history["val_acc"], label="Validation Accuracy", marker="o")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training vs. Validation Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    curves_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(curves_path, dpi=200)
    plt.close()
    print(f"Saved training curves chart -> {curves_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Dataset directory (original only): {os.path.abspath(DATA_DIR)}")

    # ---- Data ----
    train_dataset, val_dataset, test_dataset, class_names = build_datasets()

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    # ---- Model ----
    model = build_model(MODEL_NAME, NUM_CLASSES)
    model = model.to(DEVICE)

    # ---- Loss, optimizer, scheduler ----
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    scheduler = build_scheduler(
        optimizer,
        warmup_epochs=WARMUP_EPOCHS,
        total_epochs=NUM_EPOCHS,
        steps_per_epoch=len(train_loader),
    )

    # ---- Train ----
    model, history = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        DEVICE, NUM_EPOCHS, OUTPUT_DIR,
    )

    # Save training history as JSON for reference
    history_path = os.path.join(OUTPUT_DIR, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history -> {history_path}")

    # ---- Plot training curves ----
    plot_training_curves(history, OUTPUT_DIR)

    # ---- Evaluate on test set with the BEST checkpoint ----
    test_metrics = evaluate_on_test_set(model, test_loader, DEVICE, class_names, OUTPUT_DIR)

    metrics_path = os.path.join(OUTPUT_DIR, "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"Saved test metrics -> {metrics_path}")

    print("\nAll done. Check the output folder for checkpoints, reports, and charts:")
    print(f"  {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
