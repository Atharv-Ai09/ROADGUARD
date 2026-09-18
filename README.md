# RoadGuard — AI/ML-Based Road Condition Detection System

A Diploma final-year AI/ML mini project. RoadGuard accepts a road image
(via upload or webcam) and predicts whether the road is **NORMAL** or
**DAMAGED**, using a small neural network **implemented entirely from
scratch with NumPy** — no TensorFlow, PyTorch, YOLO, or scikit-learn model
training anywhere in the pipeline.

---

## 1. Problem Statement

Manually inspecting roads for damage (potholes, cracks, surface decay) is
slow and inconsistent. RoadGuard demonstrates a lightweight, explainable
machine-learning pipeline — built without deep-learning frameworks — that
can classify a road photo as normal or damaged in real time through a web
interface.

## 2. Objectives

- Build an image classifier using a hand-written NumPy neural network
  (no CNN, no framework).
- Demonstrate four core ML/DL practical topics end-to-end in one project.
- Serve predictions through a Flask API.
- Provide a React web app supporting image upload and live camera capture.
- Report only real, measured performance — never fabricated numbers.

## 3. Dataset

**Source:** [Kaggle — Road Issues Detection Dataset](https://www.kaggle.com/datasets/programmerrdai/road-issues-detection-dataset)
(~9,660 RGB JPG images across categories: Broken Road Sign Issues, Damaged
Road Issues, Pothole Issues, Illegal Parking Issues, Mixed Issues,
Littering/Garbage Issues.)

**The dataset is NOT bundled with this project.** Download it yourself:

```python
import kagglehub
path = kagglehub.dataset_download("programmerrdai/road-issues-detection-dataset")
print("Path to dataset files:", path)
```

### How the two RoadGuard classes were constructed

- **DAMAGED ROAD** — built only from categories that represent genuine
  road-surface damage: *Damaged Road Issues* and *Pothole Issues* (see
  `DAMAGED_CATEGORY_KEYWORDS` in `prepare_dataset.py`). Unrelated
  categories (signs, litter, parking) are explicitly excluded.
- **NORMAL ROAD** — this Kaggle dataset is built around *issues*, so it
  may not contain a clean "normal road" category. `prepare_dataset.py`
  checks for one; if none exists, it **stops and reports this clearly**
  instead of inventing labels, and tells you to add your own verified
  normal-road photos to `manual_normal_images/`.

No fake images or fabricated labels are ever created.

## 4. Installation

### Backend

```bash
cd ml_backend
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

### Frontend

```bash
npm install
```

## 5. Dataset Preparation

1. Download the dataset with `kagglehub` (see above) and note the path.
2. Edit `RAW_DATASET_PATH` at the top of `ml_backend/prepare_dataset.py`
   (or pass it as an argument).
3. Run:
   ```bash
   python prepare_dataset.py --raw_path /path/to/kaggle/dataset
   ```
   This prints the number of normal/damaged images found, warns about
   class imbalance, and splits data 70/15/15 into
   `dataset/train`, `dataset/validation`, `dataset/test`.

## 6. Practical Topics

### Topic 1 — NumPy for Deep Learning
`practical_numpy.py` demonstrates array creation, reshaping, slicing,
matrix operations, and the `dot(X, weights) + bias` computation that
powers every Dense layer in `neural_network.py`. Generates
`results/numpy_matrix.png`.

### Topic 2 — Perceptron and Activation Functions
`perceptron.py` implements a single-layer perceptron from scratch
(weights, bias, weighted sum, prediction, error, weight update — no
sklearn). `activation_visualization.py` implements Sigmoid, Tanh and
ReLU manually and plots them from real calculated values into
`results/activation_functions.png`.

### Topic 3 — Forward and Backpropagation
`neural_network.py` implements a 2-layer network:

```
Input (3072) → Dense → ReLU → Dense (32→1) → Sigmoid → Probability
```

Forward propagation, the backpropagation chain rule, gradient
calculation, and weight updates are all hand-written with NumPy.

### Topic 4 — Loss Functions and Optimizers
Manual implementations of **Mean Squared Error** and **Binary
Cross-Entropy** (`neural_network.py`), and manual **SGD** and **Adam**
optimizers. `train.py` lets you switch `optimizer = "sgd"` or `"adam"`.

## 7. Image Preprocessing

`preprocessing.py` defines the **one** pipeline used identically for
training, validation, testing and live web prediction:

1. Read image (OpenCV)
2. Convert BGR → RGB
3. Resize to 32×32
4. Normalize pixels to [0, 1]
5. Flatten to 3,072 features (32×32×3)

Grayscale conversion, Gaussian blur, Canny edge detection, and histogram
equalization are also implemented in this file as standalone
demonstrations, but are **not** stacked onto the production pipeline.

## 8. Training

```bash
cd ml_backend
python train.py
```

Trains with mini-batches (images are loaded from disk on demand, not all
held in RAM at once — see `data_loader.py`), using Binary Cross-Entropy
loss. Saves:

- `model/model.npz` (W1, b1, W2, b2)
- `results/loss_curve.png`
- `results/optimizer_comparison.png` (SGD vs Adam, real training runs)
- `results/loss_comparison.png` (MSE vs BCE, real training runs)

## 9. Evaluation

```bash
python evaluation.py
```

Computes Accuracy, Precision, Recall and F1 Score manually with NumPy
(no `sklearn.metrics`), on the real test set, and generates
`results/confusion_matrix.png` and `results/metrics.json`.

## 10. Flask API

```bash
python app.py
```

- `GET /health` → `{"status": "ok"}`
- `POST /predict` (multipart form field `image`) → `{"prediction": "...", "probability": 0.87}`
- `GET /metrics` → real evaluation metrics, or a "not trained" message

If `model.npz` doesn't exist yet, `/predict` returns a clear error instead
of a random or demo prediction.

## 11. React Frontend

```bash
npm run dev
```

Sections: **Dashboard, Detect Road Damage, Camera, Prediction History,
Model Performance, How It Works**. Image upload accepts JPG/PNG with a
preview. Camera capture uses `navigator.mediaDevices.getUserMedia()` and
only sends the frame after you press **Predict**. Prediction history is
stored in the browser's `localStorage`. The Model Performance page shows
"Train the model first to view actual model performance." until a real
model and `metrics.json` exist — never fake numbers.

> To see the results charts in the Performance page, copy the generated
> PNGs into `public/ml_backend_results/` after running `train.py` /
> `evaluation.py`:
> ```bash
> cp ml_backend/results/*.png public/ml_backend_results/
> ```

Build for production:

```bash
npm run build
```

## 12. Project Structure

```
RoadGuard/




├── dataset/{train,validation,test}/{normal,damaged}/
├── ml_backend/
│   ├── model/model.npz
│   ├── results/*.png, metrics.json
│   ├── prepare_dataset.py
│   ├── practical_numpy.py
│   ├── preprocessing.py
│   ├── perceptron.py
│   ├── activation_visualization.py
│   ├── neural_network.py
│   ├── data_loader.py
│   ├── train.py
│   ├── evaluation.py
│   ├── predict.py
│   ├── app.py
│   └── requirements.txt
├── src/
│   ├── components/ (Dashboard, Detect, Camera, History, Performance, HowItWorks, ResultCard)
│   ├── services/api.js
│   ├── App.jsx, main.jsx, index.css
├── public/
├── package.json
└── vite.config.js
```

## 13. Future Improvements

- Replace the flattened-pixel NumPy network with a from-scratch
  convolutional layer for better spatial feature extraction.
- Expand the normal-road dataset with a dedicated, verified source.
- Add data augmentation (rotation, brightness) implemented manually in
  NumPy/OpenCV.
- Add batch normalization and dropout, implemented from scratch.

