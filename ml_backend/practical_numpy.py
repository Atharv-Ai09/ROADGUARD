"""
==================================================
SYLLABUS TOPIC 1: NUMPY FOR DEEP LEARNING
==================================================
This script is a standalone teaching demo (Practical 1). It is NOT part of
the training pipeline -- it exists purely so the concepts of NumPy that
power the RoadGuard neural network can be demonstrated and explained
independently in a viva.

Run:
    python practical_numpy.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def demo_array_creation():
    print("\n--- 1. Array creation ---")
    a = np.array([1, 2, 3, 4, 5])
    zeros = np.zeros((2, 3))
    ones = np.ones((2, 3))
    rand = np.random.randn(2, 3)
    print("1D array:", a)
    print("Zeros:\n", zeros)
    print("Ones:\n", ones)
    print("Random normal:\n", rand)
    return a


def demo_reshaping(a):
    print("\n--- 2. Reshaping ---")
    # A flat image-like vector reshaped into a small "image"
    flat = np.arange(12)
    reshaped = flat.reshape(3, 4)
    print("Flat:", flat)
    print("Reshaped to (3,4):\n", reshaped)
    return reshaped


def demo_slicing(mat):
    print("\n--- 3. Slicing ---")
    print("First row:", mat[0, :])
    print("First column:", mat[:, 0])
    print("Sub-block (rows 0-1, cols 1-2):\n", mat[0:2, 1:3])


def demo_matrix_operations(mat):
    print("\n--- 4. Matrix operations ---")
    print("Transpose:\n", mat.T)
    print("Element-wise +1:\n", mat + 1)
    print("Element-wise *2:\n", mat * 2)
    print("Sum of all elements:", mat.sum())
    print("Mean of all elements:", mat.mean())


def demo_dot_product():
    print("\n--- 5 & 6. Dot product: weights x input + bias ---")
    # Pretend X is 4 preprocessed image features (a tiny "mini image")
    X = np.array([0.2, 0.4, 0.1, 0.9])          # input features
    weights = np.array([0.5, -0.3, 0.8, 0.1])   # learned weights
    bias = 0.05

    # This exact operation (np.dot(X, weights) + bias) is the same
    # computation used inside neural_network.py during forward propagation.
    z = np.dot(X, weights) + bias
    print("X (input features):", X)
    print("weights:", weights)
    print("bias:", bias)
    print("z = dot(X, weights) + bias =", z)
    return X, weights, z


def visualize_matrix(mat):
    print("\n--- 7. Matrix visualization ---")
    plt.figure(figsize=(5, 4))
    plt.imshow(mat, cmap="viridis")
    plt.colorbar(label="value")
    plt.title("NumPy Matrix Visualization (Practical 1)")
    for (i, j), val in np.ndenumerate(mat):
        plt.text(j, i, f"{val}", ha="center", va="center", color="white")
    out_path = os.path.join(RESULTS_DIR, "numpy_matrix.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved matrix visualization to {out_path}")


if __name__ == "__main__":
    a = demo_array_creation()
    mat = demo_reshaping(a)
    demo_slicing(mat)
    demo_matrix_operations(mat)
    demo_dot_product()
    visualize_matrix(mat)
    print("\nPractical 1 (NumPy for Deep Learning) complete.")
