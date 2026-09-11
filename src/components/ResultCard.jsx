export default function ResultCard({ result, imageSrc, onReset }) {
  const isDamaged = result.prediction === "DAMAGED ROAD";
  const pct = Math.round(result.probability * 100);

  return (
    <div className={`result-box ${isDamaged ? "damaged" : "normal"}`}>
      {imageSrc && <img src={imageSrc} alt="Analyzed road" className="result-image" style={{ marginBottom: 20 }} />}
      <div style={{ fontSize: 28 }}>{isDamaged ? "⚠" : "✓"}</div>
      <div className={`result-label ${isDamaged ? "damaged" : "normal"}`}>
        {result.prediction}
      </div>
      <div className="result-prob">Probability: {pct}%</div>
      <button className="btn-primary" onClick={onReset}>
        Analyze Another Image
      </button>
    </div>
  );
}
