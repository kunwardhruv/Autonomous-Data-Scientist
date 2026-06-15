// components/tabs/EDATab.jsx

import Card from "../Card";

export default function EDATab({ results }) {
  const { eda } = results;
  const missing = eda?.missing_values || {};
  const hasMissing = Object.keys(missing).length > 0;
  const charts = eda?.charts || [];  // [{name, data}]

  return (
    <>
      {/* Missing Values */}
      {hasMissing ? (
        <Card title="Missing values" accentColor="var(--red)">
          {Object.entries(missing).map(([col, info]) => (
            <div key={col} style={{
              display: "flex", alignItems: "center",
              justifyContent: "space-between",
              padding: "9px 0", borderBottom: "1px solid var(--border)",
              fontSize: 12,
            }}>
              <span style={{ fontFamily: "var(--mono)" }}>{col}</span>
              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <div style={{ width: 80, height: 4, background: "var(--surface3)", borderRadius: 2, overflow: "hidden" }}>
                  <div style={{ width: `${Math.min(info.percent, 100)}%`, height: "100%", background: "var(--red)", borderRadius: 2 }} />
                </div>
                <span style={{ fontFamily: "var(--mono)", fontSize: 11, color: info.percent > 30 ? "var(--red)" : "var(--muted)" }}>
                  {info.count} rows · {info.percent}%
                </span>
              </div>
            </div>
          ))}
        </Card>
      ) : (
        <Card title="Missing values" accentColor="var(--green)">
          <div style={{ fontSize: 13, color: "var(--green)" }}>✓ No missing values — clean dataset!</div>
        </Card>
      )}

      {/* Column Types */}
      <Card title="Column types" accentColor="var(--blue)">
        <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
          <div>
            <div style={{ fontSize: 11, color: "var(--muted)", marginBottom: 8, textTransform: "uppercase", letterSpacing: "0.5px" }}>
              Numeric ({eda?.numeric_columns?.length})
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 5 }}>
              {eda?.numeric_columns?.map(c => (
                <span key={c} style={{ background: "rgba(96,165,250,0.1)", border: "1px solid rgba(96,165,250,0.25)", color: "var(--blue)", borderRadius: 4, padding: "2px 7px", fontSize: 11, fontFamily: "var(--mono)" }}>{c}</span>
              ))}
            </div>
          </div>
          <div>
            <div style={{ fontSize: 11, color: "var(--muted)", marginBottom: 8, textTransform: "uppercase", letterSpacing: "0.5px" }}>
              Categorical ({eda?.categorical_columns?.length})
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 5 }}>
              {eda?.categorical_columns?.map(c => (
                <span key={c} style={{ background: "rgba(167,139,250,0.1)", border: "1px solid rgba(167,139,250,0.25)", color: "var(--purple)", borderRadius: 4, padding: "2px 7px", fontSize: 11, fontFamily: "var(--mono)" }}>{c}</span>
              ))}
            </div>
          </div>
        </div>
      </Card>

      {/* Charts — base64 directly in src */}
      {charts.length > 0 && (
        <Card title="Charts" accentColor="var(--blue)">
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: 14 }}>
            {charts.map((chart, i) => (
              <div key={i}>
                <div style={{ fontSize: 11, color: "var(--muted)", marginBottom: 6, fontFamily: "var(--mono)", textTransform: "capitalize" }}>
                  {chart.name}
                </div>
                <img
                  src={`data:image/png;base64,${chart.data}`}
                  alt={chart.name}
                  style={{ width: "100%", borderRadius: 8, background: "var(--surface2)", border: "1px solid var(--border)", display: "block" }}
                />
              </div>
            ))}
          </div>
        </Card>
      )}
    </>
  );
}