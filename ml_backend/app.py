"""
Flask API for RoadGuard.

Endpoints:
    GET  /health   -> {"status": "ok"}
    POST /predict  -> {"prediction": "...", "probability": 0.87}
    GET  /metrics  -> real evaluation metrics (or a "not trained" message)

Run:
    python app.py
"""

import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS

from predict import predict_image_bytes, MODEL_PATH

app = Flask(__name__)
CORS(app)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
METRICS_PATH = os.path.join(RESULTS_DIR, "metrics.json")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No 'image' field found in form-data."}), 400

    if not os.path.exists(MODEL_PATH):
        return jsonify({
            "error": "Model not trained yet. Run train.py before using /predict."
        }), 503

    file = request.files["image"]
    image_bytes = file.read()

    try:
        result = predict_image_bytes(image_bytes)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    return jsonify(result)


@app.route("/metrics", methods=["GET"])
def metrics():
    if not os.path.exists(METRICS_PATH):
        return jsonify({"trained": False, "message": "Train the model first to view actual model performance."})
    with open(METRICS_PATH) as f:
        data = json.load(f)
    data["trained"] = True
    return jsonify(data)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
