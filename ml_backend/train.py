"""
Training script for the RoadGuard NumPy neural network.

Trains on dataset/train, monitors dataset/validation, saves:
    model/model.npz
    results/loss_curve.png
    results/optimizer_comparison.png
    results/loss_comparison.png

Run:
    python train.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from neural_network import NeuralNetwork, binary_cross_entropy, mse_loss
from data_loader import list_split, batch_generator, num_batches

DATASET_ROOT = os.path.join(os.path.dirname(__file__), "..", "dataset")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 30          # main training run
LR = 0.001
OPTIMIZER = "adam"   # "sgd" or "adam" -- change here to switch the main run

COMPARISON_EPOCHS = 10  # shorter runs used only for the SGD-vs-Adam and
                         # MSE-vs-BCE comparison charts (kept short so the
                         # project stays lightweight on a student laptop)


def run_epoch(model, paths, labels, batch_size, lr, optimizer, loss_type="bce", train=True):
    """Runs one pass over the data. Returns the average loss for the epoch."""
    losses = []
    for X_batch, y_batch in batch_generator(paths, labels, batch_size=batch_size, shuffle=train):
        if train:
            loss = model.step(X_batch, y_batch, lr=lr, optimizer=optimizer, loss_type=loss_type)
        else:
            y_pred = model.forward(X_batch)
            loss = binary_cross_entropy(y_batch, y_pred) if loss_type == "bce" \
                else mse_loss(y_batch, y_pred)
        losses.append(loss)
    return float(np.mean(losses)) if losses else float("nan")


def train_model(train_paths, train_labels, val_paths, val_labels,
                 epochs, lr, optimizer, loss_type="bce", verbose=True):
    model = NeuralNetwork()
    train_history, val_history = [], []

    for epoch in range(epochs):
        train_loss = run_epoch(model, train_paths, train_labels, BATCH_SIZE, lr,
                                optimizer, loss_type, train=True)
        val_loss = run_epoch(model, val_paths, val_labels, BATCH_SIZE, lr,
                              optimizer, loss_type, train=False)
        train_history.append(train_loss)
        val_history.append(val_loss)
        if verbose:
            print(f"[{optimizer}/{loss_type}] Epoch {epoch+1}/{epochs} "
                  f"- train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

    return model, train_history, val_history


def main():
    train_paths, train_labels = list_split(DATASET_ROOT, "train")
    val_paths, val_labels = list_split(DATASET_ROOT, "validation")

    if len(train_paths) == 0 or len(val_paths) == 0:
        print(
            "ERROR: dataset/train or dataset/validation is empty.\n"
            "Run prepare_dataset.py first to build the dataset from your "
            "downloaded Kaggle data."
        )
        return

    print(f"Train samples: {len(train_paths)} | Validation samples: {len(val_paths)}")

    # ------------------------------------------------------------
    # 1. MAIN TRAINING RUN (this becomes the model served by Flask)
    # ------------------------------------------------------------
    print(f"\n=== Main training run ({OPTIMIZER}, BCE, {EPOCHS} epochs) ===")
    model, train_hist, val_hist = train_model(
        train_paths, train_labels, val_paths, val_labels,
        epochs=EPOCHS, lr=LR, optimizer=OPTIMIZER, loss_type="bce"
    )
    model.save(os.path.join(MODEL_DIR, "model.npz"))

    plt.figure(figsize=(7, 5))
    plt.plot(train_hist, label="Train loss")
    plt.plot(val_hist, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross-Entropy Loss")
    plt.title(f"RoadGuard Training Loss ({OPTIMIZER})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(RESULTS_DIR, "loss_curve.png"), bbox_inches="tight")
    plt.close()
    print("Saved results/loss_curve.png")

    # ------------------------------------------------------------
    # 2. OPTIMIZER COMPARISON: SGD vs Adam (shorter runs)
    # ------------------------------------------------------------
    print(f"\n=== Optimizer comparison: SGD vs Adam ({COMPARISON_EPOCHS} epochs each) ===")
    _, sgd_train_hist, _ = train_model(
        train_paths, train_labels, val_paths, val_labels,
        epochs=COMPARISON_EPOCHS, lr=LR, optimizer="sgd", loss_type="bce"
    )
    _, adam_train_hist, _ = train_model(
        train_paths, train_labels, val_paths, val_labels,
        epochs=COMPARISON_EPOCHS, lr=LR, optimizer="adam", loss_type="bce"
    )

    plt.figure(figsize=(7, 5))
    plt.plot(sgd_train_hist, label="SGD")
    plt.plot(adam_train_hist, label="Adam")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss (BCE)")
    plt.title("Optimizer Comparison: SGD vs Adam")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(RESULTS_DIR, "optimizer_comparison.png"), bbox_inches="tight")
    plt.close()
    print("Saved results/optimizer_comparison.png")

    # ------------------------------------------------------------
    # 3. LOSS FUNCTION COMPARISON: MSE vs Binary Cross-Entropy
    # ------------------------------------------------------------
    print(f"\n=== Loss comparison: MSE vs BCE ({COMPARISON_EPOCHS} epochs each) ===")
    _, mse_train_hist, _ = train_model(
        train_paths, train_labels, val_paths, val_labels,
        epochs=COMPARISON_EPOCHS, lr=LR, optimizer=OPTIMIZER, loss_type="mse"
    )
    _, bce_train_hist, _ = train_model(
        train_paths, train_labels, val_paths, val_labels,
        epochs=COMPARISON_EPOCHS, lr=LR, optimizer=OPTIMIZER, loss_type="bce"
    )

    plt.figure(figsize=(7, 5))
    plt.plot(mse_train_hist, label="MSE")
    plt.plot(bce_train_hist, label="Binary Cross-Entropy")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss (own scale)")
    plt.title("Loss Function Comparison: MSE vs BCE")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(RESULTS_DIR, "loss_comparison.png"), bbox_inches="tight")
    plt.close()
    print("Saved results/loss_comparison.png")

    print("\nTraining complete. Main model saved to model/model.npz")
    print("Run evaluation.py next to compute accuracy/precision/recall/F1 "
          "and generate the confusion matrix on the test set.")


if __name__ == "__main__":
    main()
