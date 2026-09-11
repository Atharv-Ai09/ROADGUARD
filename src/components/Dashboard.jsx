export default function Dashboard({ onNavigate }) {
  return (
    <div>
      <h1 className="page-title">ROADGUARD</h1>
      <p className="page-lead">
        Simple Road Condition Detection System. Upload or capture a road
        image to detect whether the road appears normal or damaged, using a
        neural network built from scratch with NumPy.
      </p>

      <div className="card-grid">
        <div className="stat-card">
          <div className="label">Model</div>
          <div className="value">NumPy Neural Network</div>
        </div>
        <div className="stat-card">
          <div className="label">Input</div>
          <div className="value">32 × 32 RGB Image</div>
        </div>
        <div className="stat-card">
          <div className="label">Architecture</div>
          <div className="value">3072 → 32 → 1</div>
        </div>
        <div className="stat-card">
          <div className="label">Classes</div>
          <div className="value">Normal / Damaged</div>
        </div>
      </div>

      <div className="panel">
        <div className="section-title" style={{ marginTop: 0 }}>Get started</div>
        <p className="page-lead" style={{ marginBottom: 20 }}>
          Analyze a road image by uploading a file or using your camera.
        </p>
        <div className="controls-row" style={{ justifyContent: "flex-start" }}>
          <button className="btn-primary" onClick={() => onNavigate("detect")}>
            Upload an image
          </button>
          <button className="btn-secondary" onClick={() => onNavigate("camera")}>
            Use camera
          </button>
        </div>
      </div>
    </div>
  );
}
