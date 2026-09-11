"""
Dataset preparation script.

This script does NOT download anything. You must first download the
Kaggle dataset yourself:

    import kagglehub
    path = kagglehub.dataset_download(
        "programmerrdai/road-issues-detection-dataset"
    )
    print("Path to dataset files:", path)

Then set RAW_DATASET_PATH below (or pass --raw_path on the command line)
to that folder.

WHAT THIS SCRIPT DOES
----------------------
1. Scans the raw dataset folder structure and lists the category folders
   it finds (e.g. "Damaged Road Issues", "Pothole Issues", etc).
2. Maps ONLY the categories that genuinely represent a damaged road
   surface into the DAMAGED class (see DAMAGED_CATEGORY_KEYWORDS below).
   It does NOT dump every non-road category (garbage, illegal parking,
   broken signs) into "damaged".
3. Looks for a genuine NORMAL-road source. The Kaggle "Road Issues
   Detection" dataset is built around ISSUES, so it may not contain a
   clean "normal / healthy road" category. This script:
       - checks for any folder whose name suggests normal/clear roads
       - if none is found, it STOPS and clearly reports that you must
         supply normal-road images yourself (see add_normal_images()).
   It will NEVER silently relabel arbitrary images as "normal".
4. Splits the resulting normal/damaged images 70% train / 15% validation
   / 15% test using a fixed random seed for reproducibility.
5. Copies (does not move) the images into:
       dataset/train/{normal,damaged}
       dataset/validation/{normal,damaged}
       dataset/test/{normal,damaged}
6. Prints class counts and warns about class imbalance.
"""

import os
import shutil
import random
import argparse

random.seed(42)

# ---------------------------------------------------------------------
# CONFIGURATION -- EDIT THESE TO MATCH YOUR LOCAL DOWNLOAD
# ---------------------------------------------------------------------

# Set this to the folder kagglehub printed out, e.g.
# "/home/you/.cache/kagglehub/datasets/programmerrdai/road-issues-detection-dataset/versions/1"
RAW_DATASET_PATH = r"C:\Users\Atharv\Downloads\archive\data"

# Folder (created by you) containing ONLY clean, undamaged road images,
# used if the Kaggle dataset itself has no normal-road category.
MANUAL_NORMAL_DIR = os.path.join(os.path.dirname(__file__), "..", "manual_normal_images")

OUTPUT_ROOT = os.path.join(os.path.dirname(__file__), "..", "dataset")

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")

# Only folders whose lowercase name contains one of these keywords are
# treated as genuine ROAD-SURFACE damage (pavement/road damage or
# potholes). Everything else (signs, litter, parking) is explicitly
# EXCLUDED from the damaged class.
DAMAGED_CATEGORY_KEYWORDS = ["damaged road", "pothole"]

# Optional: if a "Mixed Issues" folder exists, only include it if you
# have manually verified it contains road-surface damage. Left disabled
# by default to avoid mislabeling.
INCLUDE_MIXED_ISSUES = False

# Keywords that would indicate a genuine "normal road" folder, in case
# the dataset ever ships one.
NORMAL_CATEGORY_KEYWORDS = ["normal road", "clear road", "good road", "healthy road"]

SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}


def list_categories(raw_path):
    """
    Recursively lists EVERY sub-folder under raw_path, at any depth
    (e.g. 'data/Road Issues/Pothole Issues'), since the real Kaggle
    download may nest category folders inside a top-level grouping
    folder like 'Road Issues' or 'Public Cleanliness + Environmental
    Issues' rather than putting them directly under raw_path.
    """
    if not os.path.isdir(raw_path):
        raise FileNotFoundError(
            f"RAW_DATASET_PATH does not exist: '{raw_path}'.\n"
            "Download the dataset with kagglehub first and set RAW_DATASET_PATH."
        )
    categories = []
    for root, dirs, _files in os.walk(raw_path):
        for d in dirs:
            full = os.path.join(root, d)
            categories.append(full)
    print("Found category folders in raw dataset (recursive scan):")
    for c in categories:
        rel = os.path.relpath(c, raw_path)
        print(f"  - {rel}")
    return categories


def collect_images(folder):
    images = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(VALID_EXTENSIONS):
                images.append(os.path.join(root, f))
    return images


def _basename_lower(path):
    return os.path.basename(os.path.normpath(path)).lower()


def find_damaged_images(raw_path, categories):
    """
    categories is a list of FULL folder paths (from the recursive
    list_categories scan). We match on each folder's own basename
    (e.g. 'Pothole Issues'), not the full path, and only collect
    images directly for that matched folder (its own os.walk already
    picks up any images nested inside it too).

    To avoid double-counting, we skip a folder if one of its parent
    folders (up to raw_path) was already matched.
    """
    damaged_images = []
    matched_categories = []
    matched_full_paths = []

    for cat_path in categories:
        cat_lower = _basename_lower(cat_path)
        is_damaged = any(k in cat_lower for k in DAMAGED_CATEGORY_KEYWORDS)
        is_mixed = "mixed" in cat_lower
        if is_damaged or (is_mixed and INCLUDE_MIXED_ISSUES):
            # skip if a parent of this folder was already matched
            if any(os.path.commonpath([cat_path, m]) == m for m in matched_full_paths):
                continue
            matched_categories.append(os.path.relpath(cat_path, raw_path))
            matched_full_paths.append(cat_path)
            damaged_images.extend(collect_images(cat_path))

    print(f"\nDamaged-road categories used: {matched_categories}")
    print(f"Total damaged-road images found: {len(damaged_images)}")
    return damaged_images


def find_normal_images_in_dataset(raw_path, categories):
    normal_images = []
    matched_categories = []
    matched_full_paths = []

    for cat_path in categories:
        cat_lower = _basename_lower(cat_path)
        if any(k in cat_lower for k in NORMAL_CATEGORY_KEYWORDS):
            if any(os.path.commonpath([cat_path, m]) == m for m in matched_full_paths):
                continue
            matched_categories.append(os.path.relpath(cat_path, raw_path))
            matched_full_paths.append(cat_path)
            normal_images.extend(collect_images(cat_path))

    if matched_categories:
        print(f"Normal-road categories found in Kaggle dataset: {matched_categories}")
    return normal_images


def find_manual_normal_images():
    if os.path.isdir(MANUAL_NORMAL_DIR):
        imgs = collect_images(MANUAL_NORMAL_DIR)
        print(f"Manual normal-road images found in '{MANUAL_NORMAL_DIR}': {len(imgs)}")
        return imgs
    return []


def add_normal_images_instructions():
    print("\n" + "=" * 60)
    print("NO VALID NORMAL-ROAD IMAGES WERE FOUND.")
    print("=" * 60)
    print(
        "The Kaggle 'road-issues-detection-dataset' is built around ROAD\n"
        "ISSUES (potholes, damage, litter, signs, parking) and may not\n"
        "contain a clean 'normal / undamaged road' category.\n\n"
        "RoadGuard will NOT invent normal-road labels from unrelated images.\n\n"
        "TO FIX THIS:\n"
        f"  1. Create the folder: {os.path.abspath(MANUAL_NORMAL_DIR)}\n"
        "  2. Place clear photos of NORMAL, undamaged roads inside it\n"
        "     (you can photograph roads yourself, or use a separate,\n"
        "     clearly-licensed 'normal road' image source).\n"
        "  3. Re-run this script.\n"
    )


def split_images(images, ratios=SPLIT_RATIOS):
    images = images[:]  # copy
    random.shuffle(images)
    n = len(images)
    n_train = int(n * ratios["train"])
    n_val = int(n * ratios["validation"])
    return {
        "train": images[:n_train],
        "validation": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:],
    }


def copy_split(split_dict, class_name):
    for split_name, paths in split_dict.items():
        out_dir = os.path.join(OUTPUT_ROOT, split_name, class_name)
        os.makedirs(out_dir, exist_ok=True)
        for src in paths:
            base = os.path.basename(src)
            dst = os.path.join(out_dir, base)
            # avoid name collisions across categories
            if os.path.exists(dst):
                name, ext = os.path.splitext(base)
                dst = os.path.join(out_dir, f"{name}_{random.randint(0,999999)}{ext}")
            shutil.copyfile(src, dst)
        print(f"  {split_name}/{class_name}: {len(paths)} images")


def main(raw_path):
    print(f"Scanning raw dataset at: {raw_path}")
    categories = list_categories(raw_path)

    damaged_images = find_damaged_images(raw_path, categories)

    normal_images = find_normal_images_in_dataset(raw_path, categories)
    normal_images += find_manual_normal_images()

    print(f"\nTotal normal-road images available: {len(normal_images)}")
    print(f"Total damaged-road images available: {len(damaged_images)}")

    if len(normal_images) == 0:
        add_normal_images_instructions()
        print("STOPPING: cannot build the dataset without normal-road images.")
        return

    if len(damaged_images) == 0:
        print("STOPPING: no damaged-road images matched the configured keywords.")
        return

    # Class imbalance check
    ratio = len(normal_images) / max(len(damaged_images), 1)
    if ratio < 0.5 or ratio > 2.0:
        print(
            f"\nWARNING: class imbalance detected "
            f"(normal={len(normal_images)}, damaged={len(damaged_images)}). "
            "Consider balancing the classes before training."
        )

    print("\nSplitting into train/validation/test (70/15/15, seed=42)...")
    normal_split = split_images(normal_images)
    damaged_split = split_images(damaged_images)

    print("\nCopying files...")
    print("NORMAL class:")
    copy_split(normal_split, "normal")
    print("DAMAGED class:")
    copy_split(damaged_split, "damaged")

    print("\nDataset preparation complete.")
    print(f"Output written to: {os.path.abspath(OUTPUT_ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_path", type=str, default=RAW_DATASET_PATH,
                         help="Path to the downloaded Kaggle dataset folder")
    args = parser.parse_args()

    if not args.raw_path:
        print(
            "ERROR: RAW_DATASET_PATH is not set.\n"
            "Edit RAW_DATASET_PATH at the top of prepare_dataset.py, or run:\n"
            "  python prepare_dataset.py --raw_path /path/to/kaggle/dataset\n"
        )
    else:
        main(args.raw_path)
