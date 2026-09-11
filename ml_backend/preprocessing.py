"""
Image preprocessing pipeline used consistently across:
    - dataset preparation / training
    - validation
    - testing
    - live web prediction (Flask /predict)

Only ONE pipeline (preprocess_image) is actually fed into the neural
network. The other OpenCV operations (grayscale, blur, edges, histogram
equalization) are implemented and exposed separately so they can be
demonstrated in the viva / "How It Works" page, but they are NOT silently
stacked on top of the real training pipeline.
"""

import cv2
import numpy as np

IMG_SIZE = 32  # 32 x 32, matches the 3072-feature input layer


def load_image(path):
    """Read an image from disk using OpenCV (BGR by default)."""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def resize_image(img, size=IMG_SIZE):
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def normalize_image(img):
    """Scale pixel values from [0,255] to [0,1]."""
    return img.astype(np.float32) / 255.0


def to_grayscale(img_bgr):
    """Optional demo operation -- not used in the main pipeline."""
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def gaussian_blur(img, ksize=5):
    """Optional demo operation -- not used in the main pipeline."""
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def edge_detection(img_gray):
    """Optional demo operation -- not used in the main pipeline."""
    return cv2.Canny(img_gray, 100, 200)


def histogram_equalization(img_gray):
    """Optional demo operation -- not used in the main pipeline."""
    return cv2.equalizeHist(img_gray)


def preprocess_image(path_or_array, size=IMG_SIZE):
    """
    THE single official RoadGuard preprocessing pipeline.
    Used identically during training, validation, testing and prediction.

    Steps:
        1. Read image (if a path is given)
        2. Convert BGR -> RGB
        3. Resize to size x size
        4. Normalize pixel values to [0,1]
        5. Flatten to a 1D feature vector (size*size*3,)

    Returns:
        1D numpy float32 array of length size*size*3
    """
    if isinstance(path_or_array, str):
        img = load_image(path_or_array)
    else:
        img = path_or_array  # already a BGR numpy array (e.g. from Flask upload)

    img_rgb = bgr_to_rgb(img)
    img_resized = resize_image(img_rgb, size)
    img_norm = normalize_image(img_resized)
    features = img_norm.flatten()  # size*size*3 features
    return features


def preprocess_bytes(image_bytes, size=IMG_SIZE):
    """Preprocess an image supplied as raw bytes (used by the Flask API)."""
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not decode uploaded image bytes.")
    return preprocess_image(img_bgr, size)


if __name__ == "__main__":
    print("This module provides the shared RoadGuard preprocessing pipeline.")
    print(f"Output feature vector length for {IMG_SIZE}x{IMG_SIZE} RGB: {IMG_SIZE*IMG_SIZE*3}")
