"""
Evaluation script. Computes Accuracy, Precision, Recall, F1 Score and a
Confusion Matrix from the ACTUAL trained model on the ACTUAL test set.

No sklearn.metrics is used -- everything is computed manually with NumPy.

Run:
    python evaluation.py
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

from neural_network import NeuralNetwork
from data_loader import list_split, batch_generator

DATASET_ROOT = os.path.join(os.path.dirname(__file__), "..", "dataset")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "model.npz")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

BATCH_SIZE = 32


def confusion_matrix_manual(y_true, y_pred):
    """
    Returns [[TN, FP], [FN, TP]] computed manually with NumPy,
    for binary labels 0 (normal) / 1 (damaged).
    """
    y_true = y_true.flatten().astype(int)
    y_pred = y_pred.flatten().astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    return np.array([[tn, fp], [fn, tp]]), tp, tn, fp, fn


def compute_metrics(tp, tn, fp, fn):
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2 * precision * recall / max(precision + recall, 1e-8)) if (precision + recall) > 0 else 0.0
    return accuracy, precision, recall, f1


def plot_confusion_matrix(cm, out_path):
    labels = ["NORMAL", "DAMAGED"]
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xticks([0, 1], labels)
    plt.yticks([0, 1], labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix (Test Set)")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                      color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def main():
    if not os.path.exists(MODEL_PATH):
        print("ERROR: model/model.npz not found. Run train.py first.")
        return

    test_paths, test_labels = list_split(DATASET_ROOT, "test")
    if len(test_paths) == 0:
        print("ERROR: dataset/test is empty. Run prepare_dataset.py first.")
        return

    model = NeuralNetwork.load(MODEL_PATH)

    all_true, all_pred = [], []
    for X_batch, y_batch in batch_generator(test_paths, test_labels,
                                             batch_size=BATCH_SIZE, shuffle=False):
        probs = model.forward(X_batch)
        preds = (probs >= 0.5).astype(int)
        all_true.append(y_batch)
        all_pred.append(preds)

    y_true = np.vstack(all_true)
    y_pred = np.vstack(all_pred)

    cm, tp, tn, fp, fn = confusion_matrix_manual(y_true, y_pred)
    accuracy, precision, recall, f1 = compute_metrics(tp, tn, fp, fn)

    print(f"Test samples: {len(test_paths)}")
    print(f"Accuracy:  {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall:    {recall*100:.2f}%")
    print(f"F1 Score:  {f1*100:.2f}%")
    print("Confusion Matrix [[TN, FP], [FN, TP]]:")
    print(cm)

    plot_confusion_matrix(cm, os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    print("Saved results/confusion_matrix.png")

    # Persist metrics as JSON so the Flask API / React dashboard can
    # display the REAL numbers instead of anything fabricated.
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "test_samples": len(test_paths),
        "confusion_matrix": cm.tolist(),
    }
    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved {metrics_path}")


if __name__ == "__main__":
    main()
