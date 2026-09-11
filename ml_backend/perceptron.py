"""
==================================================
SYLLABUS TOPIC 2: PERCEPTRON AND ACTIVATION FUNCTIONS (part 1)
==================================================
A from-scratch single-layer Perceptron implemented with NumPy only.
This is a standalone educational demo showing the building block that the
full neural_network.py later generalizes into a 2-layer network.

No sklearn.linear_model.Perceptron is used anywhere.
"""

import numpy as np


class Perceptron:
    def __init__(self, n_features, lr=0.01, seed=42):
        rng = np.random.default_rng(seed)
        # Small random weights, one per input feature
        self.weights = rng.normal(0, 0.01, size=n_features)
        self.bias = 0.0
        self.lr = lr

    def step_activation(self, z):
        """Classic perceptron step function -> 0 or 1."""
        return np.where(z >= 0, 1, 0)

    def predict(self, X):
        """
        X: (n_samples, n_features)
        Returns predicted class (0 or 1) for each sample.
        """
        z = np.dot(X, self.weights) + self.bias
        return self.step_activation(z)

    def train(self, X, y, epochs=20, verbose=True):
        """
        Classic perceptron learning rule:
            weight_update = lr * (target - prediction) * input
            bias_update   = lr * (target - prediction)
        """
        n_samples = X.shape[0]
        history = []

        for epoch in range(epochs):
            total_error = 0
            for i in range(n_samples):
                xi = X[i]
                target = y[i]

                z = np.dot(xi, self.weights) + self.bias
                prediction = self.step_activation(z)

                error = target - prediction
                total_error += abs(error)

                # Weight update rule
                self.weights += self.lr * error * xi
                self.bias += self.lr * error

            history.append(total_error)
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - total misclassifications: {total_error}")

        return history


def _toy_demo():
    """
    Tiny synthetic demo (NOT the real road dataset) that shows the
    perceptron learning a simple linearly separable pattern, purely to
    prove the training loop works correctly. The real image-based
    classification happens in neural_network.py / train.py using the
    actual Kaggle dataset.
    """
    rng = np.random.default_rng(0)
    # Two clusters of 2D points
    class0 = rng.normal(loc=[2, 2], scale=0.5, size=(20, 2))
    class1 = rng.normal(loc=[6, 6], scale=0.5, size=(20, 2))
    X = np.vstack([class0, class1])
    y = np.array([0] * 20 + [1] * 20)

    model = Perceptron(n_features=2, lr=0.1)
    print("Training toy perceptron on a synthetic linearly-separable dataset...")
    model.train(X, y, epochs=10)

    preds = model.predict(X)
    accuracy = (preds == y).mean() * 100
    print(f"Toy demo training accuracy: {accuracy:.2f}%")
    print("Final weights:", model.weights, "bias:", model.bias)


if __name__ == "__main__":
    _toy_demo()
