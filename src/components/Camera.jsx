import { useRef, useState } from "react";
import { predictImage } from "../services/api.js";
import ResultCard from "./ResultCard.jsx";

export default function Camera({ onSaveHistory }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);

  const [streaming, setStreaming] = useState(false);
  const [capturedBlob, setCapturedBlob] = useState(null);
  const [capturedUrl, setCapturedUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function startCamera() {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      streamRef.current = stream;
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      setStreaming(true);
    } catch (e) {
      setError("Could not access camera. Check browser permissions.");
    }
  }

  function stopCamera() {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
    }
    setStreaming(false);
  }

  function captureImage() {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob((blob) => {
      setCapturedBlob(blob);
      setCapturedUrl(URL.createObjectURL(blob));
    }, "image/jpeg", 0.92);

    stopCamera();
  }

  async function handlePredict() {
    if (!capturedBlob) return;
    setLoading(true);
    setError(null);
    try {
      const res = await predictImage(capturedBlob);
      setResult(res);
      onSaveHistory(res);
    } catch (e) {
      setError(e.message || "Something went wrong while predicting.");
    } finally {
      setLoading(false);
    }
  }

  function reset() {
    setCapturedBlob(null);
    setCapturedUrl(null);
    setResult(null);
    setError(null);
  }

  return (
    <div>
      <h1 className="page-title">Camera</h1>
      <p className="page-lead">
        Capture a live photo of a road with your webcam. The image is only
        sent for prediction after you press Predict.
      </p>

      <div className="panel">
        {result ? (
          <ResultCard result={result} imageSrc={capturedUrl} onReset={reset} />
        ) : (
          <>
            <video ref={videoRef} style={{ display: streaming ? "block" : "none" }} muted playsInline />
            <canvas ref={canvasRef} className="camera-canvas" style={{ display: "none" }} />

            {capturedUrl && !streaming && (
              <img src={capturedUrl} alt="Captured" className="preview-image" />
            )}

            {!streaming && !capturedUrl && (
              <div className="dropzone">
                <p style={{ margin: 0 }}>Camera is off. Start the camera to capture a road image.</p>
              </div>
            )}

            <div className="controls-row" style={{ marginTop: 20 }}>
              {!streaming && !capturedUrl && (
                <button className="btn-primary" onClick={startCamera}>Start Camera</button>
              )}
              {streaming && (
                <button className="btn-primary" onClick={captureImage}>Capture Image</button>
              )}
              {capturedUrl && !streaming && (
                <>
                  <button className="btn-secondary" onClick={reset}>Retake</button>
                  <button className="btn-primary" onClick={handlePredict} disabled={loading}>
                    {loading ? "Predicting..." : "Predict"}
                  </button>
                </>
              )}
            </div>

            {error && <p className="error-text" style={{ textAlign: "center" }}>{error}</p>}
          </>
        )}
      </div>
    </div>
  );
}
