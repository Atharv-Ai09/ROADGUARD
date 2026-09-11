"""
==================================================
SYLLABUS TOPIC 2: PERCEPTRON AND ACTIVATION FUNCTIONS (part 2)
==================================================
Manual NumPy implementations of Sigmoid, Tanh and ReLU, plus their
derivatives, plotted from real calculated values (no fabricated curves).

These exact sigmoid()/relu() functions are reused by neural_network.py.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def sigmoid(z):
    z = np.clip(z, -500, 500)  # numerical stability
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)


def tanh(z):
    return np.tanh(z)


def tanh_derivative(z):
    return 1 - np.tanh(z) ** 2


def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return (z > 0).astype(float)


def generate_plot():
    z = np.linspace(-10, 10, 400)

    sig = sigmoid(z)
    tan = tanh(z)
    rel = relu(z)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(z, sig, color="royalblue")
    axes[0].set_title("Sigmoid: 1 / (1 + e^-z)")
    axes[0].axhline(0, color="gray", linewidth=0.5)
    axes[0].axvline(0, color="gray", linewidth=0.5)
    axes[0].grid(alpha=0.3)

    axes[1].plot(z, tan, color="seagreen")
    axes[1].set_title("Tanh: (e^z - e^-z)/(e^z + e^-z)")
    axes[1].axhline(0, color="gray", linewidth=0.5)
    axes[1].axvline(0, color="gray", linewidth=0.5)
    axes[1].grid(alpha=0.3)

    axes[2].plot(z, rel, color="firebrick")
    axes[2].set_title("ReLU: max(0, z)")
    axes[2].axhline(0, color="gray", linewidth=0.5)
    axes[2].axvline(0, color="gray", linewidth=0.5)
    axes[2].grid(alpha=0.3)

    for ax in axes:
        ax.set_xlabel("z")
        ax.set_ylabel("activation(z)")

    plt.suptitle("Activation Functions (computed with NumPy)")
    out_path = os.path.join(RESULTS_DIR, "activation_functions.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved activation function plot to {out_path}")


if __name__ == "__main__":
    generate_plot()
    print("Sigmoid(0) =", sigmoid(np.array([0.0])))
    print("Tanh(0)    =", tanh(np.array([0.0])))
    print("ReLU(-3)   =", relu(np.array([-3.0])))
    print("ReLU(3)    =", relu(np.array([3.0])))
