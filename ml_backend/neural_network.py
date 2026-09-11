"""
==================================================
SYLLABUS TOPIC 3: FORWARD AND BACKPROPAGATION
SYLLABUS TOPIC 4: LOSS FUNCTIONS AND OPTIMIZERS
==================================================
A 2-layer neural network implemented entirely from scratch with NumPy.

Architecture:
    Input (3072) -> Dense -> ReLU -> Dense (32->1) -> Sigmoid -> Probability

No TensorFlow / PyTorch / CNN / sklearn is used anywhere in this file.
"""

import numpy as np

from activation_visualization import sigmoid, sigmoid_derivative, relu, relu_derivative

INPUT_SIZE = 32 * 32 * 3   # 3072
HIDDEN_SIZE = 32
OUTPUT_SIZE = 1


# --------------------------------------------------------------------------
# LOSS FUNCTIONS (Topic 4)
# --------------------------------------------------------------------------
def mse_loss(y_true, y_pred):
    """Mean Squared Error."""
    return np.mean((y_true - y_pred) ** 2)


def mse_loss_derivative(y_true, y_pred):
    n = y_true.shape[0]
    return (2.0 / n) * (y_pred - y_true)


def binary_cross_entropy(y_true, y_pred):
    """Binary Cross-Entropy -- the MAIN loss used for training."""
    eps = 1e-8  # avoid log(0)
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def bce_loss_derivative(y_true, y_pred):
    eps = 1e-8
    y_pred = np.clip(y_pred, eps, 1 - eps)
    n = y_true.shape[0]
    return (1.0 / n) * (-(y_true / y_pred) + (1 - y_true) / (1 - y_pred))


class NeuralNetwork:
    """
    2-layer fully connected network:
        Layer 1: Input (3072) -> Hidden (32), ReLU
        Layer 2: Hidden (32)  -> Output (1),  Sigmoid
    """

    def __init__(self, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE,
                 output_size=OUTPUT_SIZE, seed=42):
        rng = np.random.default_rng(seed)

        # He-style initialization scaled for stability with a from-scratch net
        self.W1 = rng.normal(0, np.sqrt(2.0 / input_size), size=(input_size, hidden_size))
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = rng.normal(0, np.sqrt(2.0 / hidden_size), size=(hidden_size, output_size))
        self.b2 = np.zeros((1, output_size))

        # Adam optimizer state (only used if optimizer="adam")
        self._init_adam_state()

    def _init_adam_state(self):
        self.mW1 = np.zeros_like(self.W1); self.vW1 = np.zeros_like(self.W1)
        self.mb1 = np.zeros_like(self.b1); self.vb1 = np.zeros_like(self.b1)
        self.mW2 = np.zeros_like(self.W2); self.vW2 = np.zeros_like(self.W2)
        self.mb2 = np.zeros_like(self.b2); self.vb2 = np.zeros_like(self.b2)
        self.t = 0  # Adam time step

    # ----------------------------------------------------------------
    # FORWARD PROPAGATION
    # ----------------------------------------------------------------
    def forward(self, X):
        """
        X: (batch_size, 3072)
        Returns the output probability and caches intermediate values
        needed for backpropagation.
        """
        self.Z1 = np.dot(X, self.W1) + self.b1     # Dense: 3072 -> 32
        self.A1 = relu(self.Z1)                    # ReLU

        self.Z2 = np.dot(self.A1, self.W2) + self.b2  # Dense: 32 -> 1
        self.A2 = sigmoid(self.Z2)                     # Sigmoid -> probability

        self.X_cache = X
        return self.A2

    # ----------------------------------------------------------------
    # BACKPROPAGATION
    # ----------------------------------------------------------------
    def backward(self, y_true, loss_type="bce"):
        """
        Computes gradients of the loss w.r.t. all weights and biases,
        using the values cached during forward().
        y_true: (batch_size, 1)
        loss_type: "bce" (default, used for the main model) or "mse"
                   (used only for the MSE-vs-BCE comparison experiment).
        """
        m = y_true.shape[0]

        if loss_type == "bce":
            # dL/dZ2 for BCE + sigmoid combines to a very clean form:
            dZ2 = self.A2 - y_true                       # (m, 1)
        elif loss_type == "mse":
            # dL/dA2 = (2/m)(A2 - y_true); dA2/dZ2 = sigmoid'(Z2)
            dA2 = mse_loss_derivative(y_true, self.A2)
            dZ2 = dA2 * sigmoid_derivative(self.Z2)
        else:
            raise ValueError("loss_type must be 'bce' or 'mse'")
        dW2 = np.dot(self.A1.T, dZ2) / m              # (32, 1)
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m  # (1, 1)

        dA1 = np.dot(dZ2, self.W2.T)                  # (m, 32)
        dZ1 = dA1 * relu_derivative(self.Z1)           # (m, 32)
        dW1 = np.dot(self.X_cache.T, dZ1) / m          # (3072, 32)
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m   # (1, 32)

        return dW1, db1, dW2, db2

    # ----------------------------------------------------------------
    # OPTIMIZERS (Topic 4)
    # ----------------------------------------------------------------
    def update_sgd(self, grads, lr):
        dW1, db1, dW2, db2 = grads
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

    def update_adam(self, grads, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        dW1, db1, dW2, db2 = grads
        self.t += 1

        # --- W1 ---
        self.mW1 = beta1 * self.mW1 + (1 - beta1) * dW1
        self.vW1 = beta2 * self.vW1 + (1 - beta2) * (dW1 ** 2)
        mW1_hat = self.mW1 / (1 - beta1 ** self.t)
        vW1_hat = self.vW1 / (1 - beta2 ** self.t)
        self.W1 -= lr * mW1_hat / (np.sqrt(vW1_hat) + eps)

        # --- b1 ---
        self.mb1 = beta1 * self.mb1 + (1 - beta1) * db1
        self.vb1 = beta2 * self.vb1 + (1 - beta2) * (db1 ** 2)
        mb1_hat = self.mb1 / (1 - beta1 ** self.t)
        vb1_hat = self.vb1 / (1 - beta2 ** self.t)
        self.b1 -= lr * mb1_hat / (np.sqrt(vb1_hat) + eps)

        # --- W2 ---
        self.mW2 = beta1 * self.mW2 + (1 - beta1) * dW2
        self.vW2 = beta2 * self.vW2 + (1 - beta2) * (dW2 ** 2)
        mW2_hat = self.mW2 / (1 - beta1 ** self.t)
        vW2_hat = self.vW2 / (1 - beta2 ** self.t)
        self.W2 -= lr * mW2_hat / (np.sqrt(vW2_hat) + eps)

        # --- b2 ---
        self.mb2 = beta1 * self.mb2 + (1 - beta1) * db2
        self.vb2 = beta2 * self.vb2 + (1 - beta2) * (db2 ** 2)
        mb2_hat = self.mb2 / (1 - beta1 ** self.t)
        vb2_hat = self.vb2 / (1 - beta2 ** self.t)
        self.b2 -= lr * mb2_hat / (np.sqrt(vb2_hat) + eps)

    def step(self, X, y_true, lr, optimizer="adam", loss_type="bce"):
        """One full training step: forward -> loss -> backward -> update."""
        y_pred = self.forward(X)
        loss = binary_cross_entropy(y_true, y_pred) if loss_type == "bce" \
            else mse_loss(y_true, y_pred)
        grads = self.backward(y_true, loss_type=loss_type)

        if optimizer == "sgd":
            self.update_sgd(grads, lr)
        elif optimizer == "adam":
            self.update_adam(grads, lr)
        else:
            raise ValueError("optimizer must be 'sgd' or 'adam'")

        return loss

    # ----------------------------------------------------------------
    # SAVE / LOAD
    # ----------------------------------------------------------------
    def save(self, path):
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path):
        data = np.load(path)
        model = cls(input_size=data["W1"].shape[0], hidden_size=data["W1"].shape[1])
        model.W1 = data["W1"]
        model.b1 = data["b1"]
        model.W2 = data["W2"]
        model.b2 = data["b2"]
        return model


if __name__ == "__main__":
    # Quick sanity check with random data (NOT real training).
    net = NeuralNetwork()
    X_dummy = np.random.rand(4, INPUT_SIZE)
    y_dummy = np.array([[0], [1], [1], [0]], dtype=float)
    loss = net.step(X_dummy, y_dummy, lr=0.01, optimizer="adam")
    print("Sanity check forward+backward+update ran fine. Loss:", loss)
