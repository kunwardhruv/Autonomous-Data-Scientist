// components/Header.jsx

export default function Header() {
  return (
    <div style={{
      display: "flex", alignItems: "center", gap: 16,
      marginBottom: 40, paddingBottom: 20,
      borderBottom: "1px solid var(--border)",
    }}>
      <div style={{
        width: 36, height: 36,
        background: "linear-gradient(135deg, var(--amber), #e58c00)",
        borderRadius: 8, display: "flex", alignItems: "center",
        justifyContent: "center", fontSize: 18, flexShrink: 0,
      }}>⚗️</div>
      <div>
        <h1 style={{ fontSize: 18, fontWeight: 600, letterSpacing: "-0.3px" }}>
          Autonomous Data Scientist
        </h1>
        <p style={{ fontSize: 12, color: "var(--muted)", marginTop: 2 }}>
          Upload CSV → Full ML pipeline, automatically
        </p>
      </div>
      <span style={{
        marginLeft: "auto", fontSize: 10, fontFamily: "var(--mono)",
        background: "rgba(245,166,35,0.12)", color: "var(--amber)",
        border: "1px solid rgba(245,166,35,0.25)",
        padding: "2px 8px", borderRadius: 20, letterSpacing: "0.5px",
      }}>
        LangGraph + FastAPI
      </span>
    </div>
  );
}
