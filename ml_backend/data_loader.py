"""
Memory-efficient dataset loading utilities.

Images are NOT all loaded into RAM at once. File paths are listed first,
then mini-batches of images are read + preprocessed from disk on demand
by batch_generator(), using the SAME preprocessing pipeline as prediction
(see preprocessing.py).
"""

import os
import numpy as np
from preprocessing import preprocess_image

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")


def list_split(dataset_root, split):
    """
    Returns (file_paths, labels) for a given split ("train", "validation",
    "test"). Label 0 = normal, 1 = damaged.
    """
    paths, labels = [], []
    for class_name, label in [("normal", 0), ("damaged", 1)]:
        class_dir = os.path.join(dataset_root, split, class_name)
        if not os.path.isdir(class_dir):
            continue
        for f in os.listdir(class_dir):
            if f.lower().endswith(VALID_EXTENSIONS):
                paths.append(os.path.join(class_dir, f))
                labels.append(label)
    return paths, np.array(labels, dtype=np.float32).reshape(-1, 1)


def batch_generator(paths, labels, batch_size=32, shuffle=True, seed=42):
    """
    Yields (X_batch, y_batch) pairs. Only `batch_size` images are decoded
    and held in memory at any one time.
    """
    n = len(paths)
    indices = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    for start in range(0, n, batch_size):
        batch_idx = indices[start:start + batch_size]
        X_batch = []
        y_batch = []
        for i in batch_idx:
            try:
                features = preprocess_image(paths[i])
            except ValueError:
                continue  # skip unreadable image
            X_batch.append(features)
            y_batch.append(labels[i])
        if not X_batch:
            continue
        yield np.array(X_batch, dtype=np.float32), np.array(y_batch, dtype=np.float32)


def num_batches(n_samples, batch_size):
    return int(np.ceil(n_samples / batch_size))
