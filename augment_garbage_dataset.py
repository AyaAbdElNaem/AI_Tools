"""
====================================================================
Garbage Classification Dataset - Class Balancing Data Augmentation
====================================================================

WHAT THIS SCRIPT DOES
----------------------
Your dataset root contains 3 subfolders, each holding the SAME set of
class folders (e.g. paper, plastic, glass, metal, cardboard, trash...):

    dataset_root/
        original/
            paper/
            plastic/
            glass/
            ...
        standardized_256/
            paper/
            plastic/
            ...
        standardized_384/
            paper/
            plastic/
            ...

For EACH of these 3 subfolders, independently:
    1. Count images in every class folder.
    2. Find classes with FEWER than SMALL_CLASS_THRESHOLD images
       (default: 1400) -> these are the "target" classes to fix.
    3. For each target class, generate new augmented images (random
       flips, rotations, brightness/contrast/color jitter, etc.) by
       sampling from the class's own existing images, until the class
       reaches a total image count somewhere in TARGET_RANGE
       (default: 1400-1500).
    4. New images are saved directly inside the same class folder with
       unique, non-colliding filenames like aug_0001.jpg, aug_0002.jpg
       so nothing that already exists is ever overwritten.
    5. Print a clear BEFORE / AFTER report per subfolder so you can
       confirm every target class now falls inside the target range.

HOW TO RUN
----------
1. Create/activate a virtual environment (see the Arabic guide,
   شرح.md, for VS Code steps).
2. Install dependencies:
       pip install -r requirements.txt
3. Edit the CONFIGURATION block below (mainly DATASET_ROOT) to match
   your machine.
4. Run:
       python augment_garbage_dataset.py

The script is idempotent-ish: if you run it again after it already
reached the target range, it will simply see the classes are no
longer "small" and skip them. If a class is still below the
threshold (e.g. you added a threshold change), it will keep adding
images with new unique filenames on top of previous augmented ones.
"""

import os
import random
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm
import albumentations as A

# ============================================================
# CONFIGURATION - edit these values to match your setup
# ============================================================

# Root folder that directly contains: original/, standardized_256/, standardized_384/
DATASET_ROOT = r"C:\Users\Admin\Downloads\archive"

# The 3 subfolders to process (must exist inside DATASET_ROOT)
SUBFOLDERS = ["original", "standardized_256", "standardized_384"]

# Any class folder with FEWER images than this is considered "small"
# and will be targeted for augmentation.
SMALL_CLASS_THRESHOLD = 1300

# After augmentation, each targeted class should end up with a total
# image count somewhere in this (inclusive) range.
TARGET_RANGE = (1400, 1500)

# Valid image extensions to look for / generate
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Prefix used for newly generated augmented images
AUG_PREFIX = "aug_"

# JPEG save quality for augmented images
JPEG_QUALITY = 95

# Reproducibility
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ============================================================
# AUGMENTATION PIPELINE
# ============================================================
# A mix of geometric + color transformations. Each is applied with
# its own probability so every generated image looks a bit different,
# even when sampled from the same source image.

augmentation_pipeline = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=25, border_mode=cv2.BORDER_REFLECT_101, p=0.6),
    A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.7),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
    A.GaussNoise(var_limit=(10.0, 40.0), p=0.2),
    A.Blur(blur_limit=3, p=0.15),
    A.Affine(scale=(0.9, 1.1), translate_percent=(0.0, 0.05), shear=(-5, 5), p=0.3),
])


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def list_images(folder: Path):
    """Return a sorted list of image file paths inside a folder (non-recursive)."""
    if not folder.exists():
        return []
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS]
    )


def count_classes(subfolder_path: Path):
    """Return {class_name: image_count} for every class folder inside subfolder_path."""
    counts = {}
    if not subfolder_path.exists():
        return counts
    for class_dir in sorted(subfolder_path.iterdir()):
        if class_dir.is_dir():
            counts[class_dir.name] = len(list_images(class_dir))
    return counts


def next_available_aug_index(class_dir: Path):
    """
    Look at existing files named aug_####.* in class_dir and return the next
    free integer index, so re-running the script never overwrites files.
    """
    max_index = 0
    for f in class_dir.iterdir():
        if f.is_file() and f.stem.startswith(AUG_PREFIX):
            suffix = f.stem[len(AUG_PREFIX):]
            if suffix.isdigit():
                max_index = max(max_index, int(suffix))
    return max_index + 1


def load_image_bgr(path: Path):
    """Load an image with OpenCV (BGR). Falls back gracefully on unreadable files."""
    img = cv2.imread(str(path))
    return img


def augment_class(class_dir: Path, current_count: int, target_count: int):
    """
    Generate (target_count - current_count) new augmented images inside
    class_dir, sampling randomly (with replacement) from its existing images.
    Returns the number of images actually generated.
    """
    source_images = list_images(class_dir)
    if not source_images:
        print(f"  [!] Skipping '{class_dir.name}': no source images found to augment from.")
        return 0

    needed = target_count - current_count
    if needed <= 0:
        return 0

    start_index = next_available_aug_index(class_dir)
    generated = 0

    with tqdm(total=needed, desc=f"Augmenting '{class_dir.name}'", unit="img") as pbar:
        for i in range(needed):
            src_path = random.choice(source_images)
            image = load_image_bgr(src_path)

            if image is None:
                # Corrupt/unreadable file - try another source instead of failing the run
                continue

            augmented = augmentation_pipeline(image=image)["image"]

            new_index = start_index + i
            new_filename = f"{AUG_PREFIX}{new_index:04d}.jpg"
            new_path = class_dir / new_filename

            cv2.imwrite(
                str(new_path),
                augmented,
                [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
            )
            generated += 1
            pbar.update(1)

    return generated


def print_report(title: str, report_data: dict):
    """
    report_data: {subfolder_name: {class_name: count}}
    Prints a clean aligned table.
    """
    print(f"\n{title}")
    print("=" * len(title))
    for subfolder, class_counts in report_data.items():
        print(f"\n[{subfolder}]")
        if not class_counts:
            print("  (folder not found or empty)")
            continue
        name_width = max(len(name) for name in class_counts) + 2
        for class_name, count in sorted(class_counts.items()):
            print(f"  {class_name:<{name_width}} {count}")


# ============================================================
# MAIN SCRIPT
# ============================================================

def main():
    dataset_root = Path(DATASET_ROOT)

    if not dataset_root.exists():
        raise FileNotFoundError(
            f"DATASET_ROOT '{dataset_root}' does not exist. "
            f"Please edit DATASET_ROOT at the top of the script."
        )

    print(f"Dataset root: {dataset_root.resolve()}")
    print(f"Small-class threshold: < {SMALL_CLASS_THRESHOLD} images")
    print(f"Target range after augmentation: {TARGET_RANGE[0]}-{TARGET_RANGE[1]} images\n")

    before_report = {}
    after_report = {}
    targeted_classes_summary = defaultdict(list)  # subfolder -> list of class names targeted

    # ---- Step 1: BEFORE counts ----
    for subfolder in SUBFOLDERS:
        subfolder_path = dataset_root / subfolder
        before_report[subfolder] = count_classes(subfolder_path)

    print_report("BEFORE AUGMENTATION - Image counts per class", before_report)

    # ---- Step 2: Identify target (small) classes PER SUBFOLDER ----
    # A class might be small in one subfolder but fine in another, so each
    # subfolder is evaluated independently.
    for subfolder in SUBFOLDERS:
        counts = before_report[subfolder]
        small_classes = [c for c, n in counts.items() if n < SMALL_CLASS_THRESHOLD]
        targeted_classes_summary[subfolder] = small_classes

    print("\nClasses identified for augmentation (per subfolder):")
    for subfolder in SUBFOLDERS:
        classes = targeted_classes_summary[subfolder]
        if classes:
            print(f"  [{subfolder}] -> {', '.join(classes)}")
        else:
            print(f"  [{subfolder}] -> none (all classes already >= {SMALL_CLASS_THRESHOLD})")

    # ---- Step 3: Run augmentation ----
    for subfolder in SUBFOLDERS:
        subfolder_path = dataset_root / subfolder
        small_classes = targeted_classes_summary[subfolder]

        if not small_classes:
            continue

        print(f"\n>>> Processing subfolder: {subfolder}")
        for class_name in small_classes:
            class_dir = subfolder_path / class_name
            current_count = before_report[subfolder][class_name]

            # Pick a random target within TARGET_RANGE for natural variation
            target_count = random.randint(TARGET_RANGE[0], TARGET_RANGE[1])
            # Never target below what we already have
            target_count = max(target_count, current_count)

            generated = augment_class(class_dir, current_count, target_count)
            print(
                f"  '{class_name}': {current_count} -> "
                f"{current_count + generated} images "
                f"(+{generated} generated, target was {target_count})"
            )

    # ---- Step 4: AFTER counts ----
    for subfolder in SUBFOLDERS:
        subfolder_path = dataset_root / subfolder
        after_report[subfolder] = count_classes(subfolder_path)

    print_report("AFTER AUGMENTATION - Image counts per class", after_report)

    # ---- Step 5: Verification summary ----
    print("\nVERIFICATION SUMMARY")
    print("=====================")
    all_ok = True
    for subfolder in SUBFOLDERS:
        for class_name in targeted_classes_summary[subfolder]:
            final_count = after_report[subfolder].get(class_name, 0)
            in_range = TARGET_RANGE[0] <= final_count <= TARGET_RANGE[1]
            status = "OK" if in_range else "OUT OF RANGE"
            if not in_range:
                all_ok = False
            print(f"  [{subfolder}] {class_name}: {final_count} images -> {status}")

    if all_ok:
        print("\nAll targeted classes are now within the target range. Done!")
    else:
        print(
            "\nSome classes are still out of range - re-run the script, "
            "or check for unreadable source images in that class folder."
        )


if __name__ == "__main__":
    main()
