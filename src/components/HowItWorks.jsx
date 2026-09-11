const steps = [
  "Road Image (upload or camera)",
  "OpenCV Preprocessing (RGB, resize 32×32, normalize)",
  "Flatten → 3072 NumPy Features",
  "Dense Layer (3072 → 32)",
  "ReLU Activation",
  "Dense Layer (32 → 1)",
  "Sigmoid Activation",
  "Probability",
  "NORMAL ROAD / DAMAGED ROAD",
];

const topics = [
  {
    title: "Topic 1 — NumPy for Deep Learning",
    what: "Array creation, reshaping, slicing, matrix operations and dot products.",
    where: "The weighted sum z = dot(X, weights) + bias inside every Dense layer.",
    why: "It's the core arithmetic that turns image features into a prediction.",
    file: "practical_numpy.py, neural_network.py",
  },
  {
    title: "Topic 2 — Perceptron & Activation Functions",
    what: "A single-layer perceptron with a manual weight-update rule, plus Sigmoid, Tanh and ReLU implemented by hand.",
    where: "perceptron.py demonstrates the rule standalone; ReLU and Sigmoid are used inside the real network.",
    why: "Activations introduce non-linearity so the network can separate normal vs. damaged roads.",
    file: "perceptron.py, activation_visualization.py",
  },
  {
    title: "Topic 3 — Forward & Backpropagation",
    what: "A full forward pass through two Dense layers, and a manual backward pass computing gradients with the chain rule.",
    where: "NeuralNetwork.forward() and NeuralNetwork.backward()",
    why: "Backpropagation is how the network learns from its mistakes each epoch.",
    file: "neural_network.py",
  },
  {
    title: "Topic 4 — Loss Functions & Optimizers",
    what: "Mean Squared Error and Binary Cross-Entropy losses; SGD and Adam optimizers, all implemented from scratch.",
    where: "Training loop in train.py, gradient updates in neural_network.py",
    why: "The loss tells the network how wrong it is; the optimizer decides how to update the weights.",
    file: "neural_network.py, train.py",
  },
];

export default function HowItWorks() {
  return (
    <div>
      <h1 className="page-title">How It Works</h1>
      <p className="page-lead">
        RoadGuard classifies a road image using a 2-layer neural network
        written from scratch in NumPy — no TensorFlow, PyTorch, or CNNs.
      </p>

      <div className="panel">
        <div className="flow-diagram">
          {steps.map((s, i) => (
            <div key={i} style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
              <div className="flow-step">{s}</div>
              {i < steps.length - 1 && <div className="flow-arrow">↓</div>}
            </div>
          ))}
        </div>
      </div>

      <div className="section-title">Viva: how the four practical topics are used</div>
      {topics.map((t, i) => (
        <div className="topic-block" key={i}>
          <h3>{t.title}</h3>
          <p><strong>What:</strong> {t.what}</p>
          <p><strong>Where:</strong> {t.where}</p>
          <p><strong>Why:</strong> {t.why}</p>
          <p><strong>File:</strong> <code>{t.file}</code></p>
        </div>
      ))}
    </div>
  );
}
