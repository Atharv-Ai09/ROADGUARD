const API_BASE = "http://localhost:5000";

export async function checkHealth() {
  const res = await fetch(`${API_BASE}/health`);
  return res.json();
}

export async function predictImage(blob) {
  const formData = new FormData();
  formData.append("image", blob, "road.jpg");
  const res = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: "Prediction failed." }));
    throw new Error(err.error || "Prediction failed.");
  }
  return res.json();
}

export async function fetchMetrics() {
  const res = await fetch(`${API_BASE}/metrics`);
  return res.json();
}
