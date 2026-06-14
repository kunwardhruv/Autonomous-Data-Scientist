// components/UploadZone.jsx

import { useRef, useState } from "react";

const S = {
  zone: (drag, hasFile) => ({
    border: `1.5px dashed ${hasFile ? "var(--green)" : drag ? "var(--amber)" : "var(--border2)"}`,
    borderStyle: hasFile ? "solid" : "dashed",
    borderRadius: 16, padding: "48px 32px", textAlign: "center",
    cursor: "pointer", transition: "all 0.2s",
    background: hasFile
      ? "rgba(74,222,128,0.04)"
      : drag ? "rgba(245,166,35,0.04)" : "var(--surface)",
    position: "relative",
  }),
  fileChip: {
    display: "inline-flex", alignItems: "center", gap: 8,
    background: "rgba(74,222,128,0.1)",
    border: "1px solid rgba(74,222,128,0.3)",
    color: "var(--green)", borderRadius: 20,
    padding: "4px 12px 4px 8px",
    fontSize: 12, fontFamily: "var(--mono)", marginTop: 12,
  },
};

export default function UploadZone({ file, onFileSelect, onAnalyze, onClear, loading }) {
  const [drag, setDrag] = useState(false);
  const inputRef = useRef();

  const handleDrop = (e) => {
    e.preventDefault(); setDrag(false);
    onFileSelect(e.dataTransfer.files[0]);
  };

  return (
    <>
      <div
        style={S.zone(drag, !!file)}
        onDragOver={e => { e.preventDefault(); setDrag(true); }}
        onDragLeave={() => setDrag(false)}
        onDrop={handleDrop}
        onClick={() => !file && inputRef.current.click()}
      >
        {/* Hidden file input */}
        <input
          ref={inputRef} type="file" accept=".csv"
          style={{ position: "absolute", inset: 0, opacity: 0, cursor: "pointer" }}
          onChange={e => onFileSelect(e.target.files[0])}
        />

        <div style={{ fontSize: 40, marginBottom: 16, opacity: 0.6 }}>
          {file ? "📊" : "📂"}
        </div>

        {file ? (
          <>
            <h2 style={{ fontSize: 15, fontWeight: 500 }}>File ready to analyze</h2>
            <div style={S.fileChip}>
              <div style={{ width: 6, height: 6, background: "var(--green)", borderRadius: "50%" }} />
              {file.name} · {(file.size / 1024).toFixed(1)} KB
            </div>
          </>
        ) : (
          <>
            <h2 style={{ fontSize: 15, fontWeight: 500, marginBottom: 6 }}>
              Drop your CSV here
            </h2>
            <p style={{ fontSize: 12, color: "var(--muted)" }}>
              or click to browse · Titanic, Iris, Housing — any dataset works
            </p>
          </>
        )}
      </div>

      {/* Action buttons */}
      <div style={{ display: "flex", gap: 10, marginTop: 20, justifyContent: "center" }}>
        {file && (
          <button
            onClick={onAnalyze}
            disabled={loading}
            style={{
              padding: "10px 22px", borderRadius: 8, fontSize: 13,
              fontWeight: 500, cursor: loading ? "not-allowed" : "pointer",
              border: "none", background: "var(--amber)", color: "#0c0e14",
              opacity: loading ? 0.4 : 1, transition: "all 0.15s",
              display: "inline-flex", alignItems: "center", gap: 8,
              fontFamily: "var(--sans)",
            }}
          >
            {loading ? "⏳ Analyzing..." : "▶ Run Analysis"}
          </button>
        )}
        {file && (
          <button
            onClick={onClear}
            style={{
              padding: "10px 22px", borderRadius: 8, fontSize: 13,
              fontWeight: 500, cursor: "pointer", fontFamily: "var(--sans)",
              background: "transparent", color: "var(--muted)",
              border: "1px solid var(--border2)", transition: "all 0.15s",
            }}
          >
            ✕ Clear
          </button>
        )}
      </div>
    </>
  );
}
