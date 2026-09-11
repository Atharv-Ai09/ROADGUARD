export default function History({ history, onClear }) {
  return (
    <div>
      <h1 className="page-title">Prediction History</h1>
      <p className="page-lead">
        A local log of predictions made in this browser, stored in
        localStorage.
      </p>

      <div className="panel">
        {history.length === 0 ? (
          <div className="metric-empty">No predictions yet. Analyze an image to see it appear here.</div>
        ) : (
          <>
            <table className="history-table">
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Time</th>
                  <th>Prediction</th>
                  <th>Probability</th>
                </tr>
              </thead>
              <tbody>
                {history.map((h, i) => (
                  <tr key={i}>
                    <td>{h.date}</td>
                    <td>{h.time}</td>
                    <td>
                      <span className={`tag ${h.prediction === "DAMAGED ROAD" ? "damaged" : "normal"}`}>
                        {h.prediction}
                      </span>
                    </td>
                    <td>{Math.round(h.probability * 100)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div style={{ marginTop: 20 }}>
              <button className="btn-secondary" onClick={onClear}>Clear History</button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
