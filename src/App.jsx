import { useEffect, useState } from "react";
import Dashboard from "./components/Dashboard.jsx";
import Detect from "./components/Detect.jsx";
import Camera from "./components/Camera.jsx";
import History from "./components/History.jsx";
import Performance from "./components/Performance.jsx";
import HowItWorks from "./components/HowItWorks.jsx";

const NAV_ITEMS = [
  { id: "dashboard", label: "Dashboard" },
  { id: "detect", label: "Detect Road Damage" },
  { id: "camera", label: "Camera" },
  { id: "history", label: "Prediction History" },
  { id: "performance", label: "Model Performance" },
  { id: "how", label: "How It Works" },
];

const HISTORY_KEY = "roadguard_history";

export default function App() {
  const [page, setPage] = useState("dashboard");
  const [history, setHistory] = useState([]);

  useEffect(() => {
    const stored = localStorage.getItem(HISTORY_KEY);
    if (stored) {
      try { setHistory(JSON.parse(stored)); } catch { /* ignore corrupt data */ }
    }
  }, []);

  function saveHistory(result) {
    const now = new Date();
    const entry = {
      date: now.toISOString().slice(0, 10),
      time: now.toTimeString().slice(0, 5),
      prediction: result.prediction,
      probability: result.probability,
    };
    const updated = [entry, ...history];
    setHistory(updated);
    localStorage.setItem(HISTORY_KEY, JSON.stringify(updated));
  }

  function clearHistory() {
    setHistory([]);
    localStorage.removeItem(HISTORY_KEY);
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">Road<span>Guard</span></div>
        <div className="brand-sub">AI/ML Road Condition Detection</div>
        {NAV_ITEMS.map((item) => (
          <button
            key={item.id}
            className={`nav-item ${page === item.id ? "active" : ""}`}
            onClick={() => setPage(item.id)}
          >
            {item.label}
          </button>
        ))}
      </aside>

      <main className="main-content">
        {page === "dashboard" && <Dashboard onNavigate={setPage} />}
        {page === "detect" && <Detect onSaveHistory={saveHistory} />}
        {page === "camera" && <Camera onSaveHistory={saveHistory} />}
        {page === "history" && <History history={history} onClear={clearHistory} />}
        {page === "performance" && <Performance />}
        {page === "how" && <HowItWorks />}
      </main>
    </div>
  );
}
