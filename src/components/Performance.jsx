import { useEffect, useState } from "react";
import { fetchMetrics } from "../services/api.js";

export default function Performance() {
  const [metrics, setMetrics] = useState(null);
  const [loadingErr, setLoadingErr] = useState(null);

  useEffect(() => {
    fetchMetrics()
      .then(setMetrics)
      .catch(() => setLoadingErr("Could not reach the RoadGuard API. Is app.py running?"));
  }, []);

  return (
    <div>
      <h1 className="page-title">Model Performance</h1>
      <p className="page-lead">
        Real evaluation results computed by evaluation.py from the trained
        model on the held-out test set. Nothing here is fabricated.
      </p>

      {loadingErr && <div className="metric-empty">{loadingErr}</div>}

      {!loadingErr && !metrics && <div className="metric-empty">Loading...</div>}

      {metrics && !metrics.trained && (
        <div className="metric-empty">Train the model first to view actual model performance.</div>
      )}

      {metrics && metrics.trained && (
        <>
          <div className="card-grid">
            <div className="stat-card">
              <div className="label">Accuracy</div>
              <div className="value">{(metrics.accuracy * 100).toFixed(2)}%</div>
            </div>
            <div className="stat-card">
              <div className="label">Precision</div>
              <div className="value">{(metrics.precision * 100).toFixed(2)}%</div>
            </div>
            <div className="stat-card">
              <div className="label">Recall</div>
              <div className="value">{(metrics.recall * 100).toFixed(2)}%</div>
            </div>
            <div className="stat-card">
              <div className="label">F1 Score</div>
              <div className="value">{(metrics.f1_score * 100).toFixed(2)}%</div>
            </div>
          </div>

          <div className="section-title">Confusion Matrix</div>
          <div className="panel">
            <img
              src="/ml_backend_results/confusion_matrix.png"
              alt="Confusion matrix"
              style={{ maxWidth: "100%", borderRadius: 6 }}
              onError={(e) => (e.target.style.display = "none")}
            />
            <p className="page-lead" style={{ marginTop: 12, marginBottom: 0 }}>
              Generated at ml_backend/results/confusion_matrix.png
            </p>
          </div>

          <div className="section-title">Loss Curve</div>
          <div className="panel">
            <img
              src="/ml_backend_results/loss_curve.png"
              alt="Loss curve"
              style={{ maxWidth: "100%", borderRadius: 6 }}
              onError={(e) => (e.target.style.display = "none")}
            />
          </div>

          <div className="section-title">SGD vs Adam</div>
          <div className="panel">
            <img
              src="/ml_backend_results/optimizer_comparison.png"
              alt="Optimizer comparison"
              style={{ maxWidth: "100%", borderRadius: 6 }}
              onError={(e) => (e.target.style.display = "none")}
            />
          </div>

          <div className="section-title">MSE vs Cross-Entropy</div>
          <div className="panel">
            <img
              src="/ml_backend_results/loss_comparison.png"
              alt="Loss comparison"
              style={{ maxWidth: "100%", borderRadius: 6 }}
              onError={(e) => (e.target.style.display = "none")}
            />
          </div>
        </>
      )}
    </div>
  );
}
