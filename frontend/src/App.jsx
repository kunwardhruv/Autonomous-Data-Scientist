// App.jsx
//
// Structure:
//   App
//   ├── Header
//   ├── UploadZone        ← CSV drag-drop + analyze button
//   ├── PipelineProgress  ← Animated step tracker during loading
//   └── ResultsDashboard  ← 5 tabs: Overview / EDA / Models / Results / Explain

import React, { useState, useRef, useEffect } from "react";
import Header from "./components/Header";
import UploadZone from "./components/UploadZone";
import PipelineProgress from "./components/PipelineProgress";
import ResultsDashboard from "./components/ResultsDashboard";

// WHY no hardcoded URL?
// With Vite proxy (vite.config.js), /analyze automatically goes to FastAPI.
// In production, just change the proxy target or set VITE_API_BASE env var.
const API_BASE = import.meta.env.VITE_API_BASE || "";

export default function App() {
  const [file, setFile]       = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError]     = useState(null);

  const handleFileSelect = (f) => {
    if (!f) return;
    if (!f.name.endsWith(".csv")) { setError("Only .csv files are supported."); return; }
    setFile(f);
    setError(null);
    setResults(null);
  };

  const handleAnalyze = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResults(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch(`${API_BASE}/analyze`, {
        method: "POST",
        body: formData,
      });

      const data = await res.json();

      if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
      if (data.status === "error") throw new Error(data.detail);

      setResults(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setResults(null);
    setError(null);
  };

  return (
    <div style={{ maxWidth: 1100, margin: "0 auto", padding: "24px 20px" }}>
      <Header />

      {/* Show upload only when no results */}
      {!results && (
        <UploadZone
          file={file}
          onFileSelect={handleFileSelect}
          onAnalyze={handleAnalyze}
          onClear={handleReset}
          loading={loading}
        />
      )}

      {/* Error message */}
      {error && (
        <div style={{
          background: "rgba(248,113,113,0.08)",
          border: "1px solid rgba(248,113,113,0.25)",
          borderRadius: 10, padding: 16,
          color: "var(--red)", fontSize: 13, marginTop: 16,
        }}>
          ⚠ {error}
        </div>
      )}

      {/* Pipeline progress animation */}
      {loading && <PipelineProgress />}

      {/* Final results */}
      {results && (
        <ResultsDashboard
          results={results}
          onReset={handleReset}
          apiBase={API_BASE}
        />
      )}
    </div>
  );
}
