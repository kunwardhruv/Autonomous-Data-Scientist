// components/Card.jsx — Reusable dark card with accent title bar

export default function Card({ title, accentColor = "var(--amber)", children, style = {} }) {
  return (
    <div style={{
      background: "var(--surface)", border: "1px solid var(--border)",
      borderRadius: 12, padding: 20, marginBottom: 16, ...style,
    }}>
      {title && (
        <div style={{
          fontSize: 11, fontWeight: 600, letterSpacing: 1,
          textTransform: "uppercase", color: "var(--muted)",
          marginBottom: 16, display: "flex", alignItems: "center", gap: 8,
        }}>
          <div style={{ width: 16, height: 2, borderRadius: 1, background: accentColor, flexShrink: 0 }} />
          {title}
        </div>
      )}
      {children}
    </div>
  );
}
