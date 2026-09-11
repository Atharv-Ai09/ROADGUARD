"""
Single-image prediction using the trained RoadGuard model.

Used both as a CLI tool and imported by app.py (the Flask API).
"""

import os
import numpy as np

from neural_network import NeuralNetwork
from preprocessing import preprocess_image, preprocess_bytes

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "model.npz")

_model_cache = None


def get_model():
    """Loads model.npz once and caches it in memory."""
    global _model_cache
    if _model_cache is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                "model/model.npz not found. Train the model first: python train.py"
            )
        _model_cache = NeuralNetwork.load(MODEL_PATH)
    return _model_cache


def predict_image_path(path):
    features = preprocess_image(path)
    return _predict_from_features(features)


def predict_image_bytes(image_bytes):
    features = preprocess_bytes(image_bytes)
    return _predict_from_features(features)


def _predict_from_features(features):
    model = get_model()
    X = features.reshape(1, -1)
    probability = float(model.forward(X)[0, 0])
    label = "DAMAGED ROAD" if probability >= 0.5 else "NORMAL ROAD"
    return {"prediction": label, "probability": round(probability, 4)}


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>")
    else:
        result = predict_image_path(sys.argv[1])
        print(result)
