import { useState, useRef } from "react";
import { predictImage } from "../services/api.js";
import ResultCard from "./ResultCard.jsx";

export default function Detect({ onSaveHistory }) {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const inputRef = useRef(null);

  function handleFile(f) {
    if (!f) return;
    const validTypes = ["image/jpeg", "image/jpg", "image/png"];
    if (!validTypes.includes(f.type)) {
      setError("Please choose a JPG or PNG image.");
      return;
    }
    setError(null);
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
  }

  async function handleDetect() {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const res = await predictImage(file);
      setResult(res);
      onSaveHistory(res);
    } catch (e) {
      setError(e.message || "Something went wrong while predicting.");
    } finally {
      setLoading(false);
    }
  }

  function reset() {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  }

  return (
    <div>
      <h1 className="page-title">Detect Road Damage</h1>
      <p className="page-lead">
        Upload a JPG or PNG photo of a road. RoadGuard preprocesses the image
        with OpenCV and runs it through the trained NumPy neural network.
      </p>

      <div className="panel">
        {result ? (
          <ResultCard result={result} imageSrc={preview} onReset={reset} />
        ) : (
          <>
            {preview && <img src={preview} alt="Preview" className="preview-image" />}

            {!preview && (
              <div
                className={`dropzone ${dragActive ? "active" : ""}`}
                onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
                onDragLeave={() => setDragActive(false)}
                onDrop={(e) => {
                  e.preventDefault();
                  setDragActive(false);
                  handleFile(e.dataTransfer.files[0]);
                }}
                onClick={() => inputRef.current.click()}
                style={{ cursor: "pointer" }}
              >
                <p style={{ margin: 0 }}>Drag & drop a road image here, or click to browse</p>
                <p style={{ fontSize: 12, marginTop: 8 }}>JPG or PNG</p>
              </div>
            )}

            <input
              ref={inputRef}
              type="file"
              accept=".jpg,.jpeg,.png"
              style={{ display: "none" }}
              onChange={(e) => handleFile(e.target.files[0])}
            />

            <div className="controls-row" style={{ marginTop: 20 }}>
              {preview && (
                <button className="btn-secondary" onClick={reset}>Choose different image</button>
              )}
              <button className="btn-primary" onClick={handleDetect} disabled={!file || loading}>
                {loading ? "Analyzing..." : "Detect Road Condition"}
              </button>
            </div>

            {error && <p className="error-text" style={{ textAlign: "center" }}>{error}</p>}
          </>
        )}
      </div>
    </div>
  );
}
