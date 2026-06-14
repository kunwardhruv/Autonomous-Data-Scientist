// components/tabs/OverviewTab.jsx

import Card from "../Card";

// ── Stat box ─────────────────────────────────────────────────────────────────
function StatBox({ value, label, color = "var(--blue)" }) {
  return (
    <div style={{
      background: "var(--surface2)", borderRadius: 8,
      padding: 14, textAlign: "center",
    }}>
      <div style={{
        fontSize: 22, fontWeight: 600, fontFamily: "var(--mono)", color,
      }}>{value}</div>
      <div style={{
        fontSize: 10, color: "var(--muted)", marginTop: 4,
        letterSpacing: "0.5px", textTransform: "uppercase",
      }}>{label}</div>
    </div>
  );
}

// ── Problem type badge ────────────────────────────────────────────────────────
const PROBLEM_STYLES = {
  classification: { bg: "rgba(167,139,250,0.12)", color: "var(--purple)", border: "rgba(167,139,250,0.3)", icon: "🔵" },
  regression:     { bg: "rgba(96,165,250,0.12)",  color: "var(--blue)",   border: "rgba(96,165,250,0.3)",  icon: "📈" },
  clustering:     { bg: "rgba(245,166,35,0.12)",  color: "var(--amber)",  border: "rgba(245,166,35,0.3)",  icon: "🔮" },
};

export default function OverviewTab({ results }) {
  const { eda, problem, model_selection, tuning } = results;
  const ps = PROBLEM_STYLES[problem?.type] || PROBLEM_STYLES.classification;
  const metricLabel = problem?.type === "regression" ? "R² Score" : "F1 Score";
  const missingCount = Object.keys(eda?.missing_values || {}).length;

  return (
    <>
      {/* Dataset stats */}
      <Card title="Dataset" accentColor="var(--amber)">
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(120px, 1fr))", gap: 10 }}>
          <StatBox value={eda?.shape?.rows?.toLocaleString()} label="Rows"       color="var(--blue)"   />
          <StatBox value={eda?.shape?.columns}                label="Columns"    color="var(--purple)" />
          <StatBox value={eda?.numeric_columns?.length}       label="Numeric"    color="var(--amber)"  />
          <StatBox value={eda?.categorical_columns?.length}   label="Categorical" color="var(--green)" />
          <StatBox
            value={missingCount}
            label="Cols w/ NaN"
            color={missingCount > 0 ? "var(--red)" : "var(--green)"}
          />
        </div>
      </Card>

      {/* Problem type */}
      <Card title="Problem detected" accentColor="var(--purple)">
        <div style={{ display: "flex", alignItems: "center", gap: 16, flexWrap: "wrap" }}>
          <span style={{
            display: "inline-flex", alignItems: "center", gap: 8,
            padding: "8px 16px", borderRadius: 20,
            fontSize: 12, fontWeight: 500, fontFamily: "var(--mono)",
            background: ps.bg, color: ps.color,
            border: `1px solid ${ps.border}`,
          }}>
            {ps.icon}&nbsp;{problem?.type?.toUpperCase()}
          </span>
          <div style={{ fontSize: 12, color: "var(--muted)" }}>
            Target column:{" "}
            <span style={{ fontFamily: "var(--mono)", color: "var(--text)" }}>
              {problem?.target}
            </span>
          </div>
        </div>

        {/* Feature chips */}
        <div style={{ marginTop: 16 }}>
          <div style={{
            fontSize: 11, color: "var(--muted)", marginBottom: 8,
            textTransform: "uppercase", letterSpacing: "0.5px",
          }}>
            Features ({problem?.features?.length})
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
            {problem?.features?.map(f => (
              <span key={f} style={{
                background: "var(--surface2)", border: "1px solid var(--border)",
                borderRadius: 4, padding: "3px 8px",
                fontSize: 11, fontFamily: "var(--mono)", color: "var(--muted)",
              }}>{f}</span>
            ))}
          </div>
        </div>
      </Card>

      {/* Winner at a glance */}
      <Card title="Best model" accentColor="var(--green)">
        <div style={{ display: "flex", alignItems: "flex-end", gap: 32, flexWrap: "wrap" }}>
          {/* Big score */}
          <div>
            <div style={{
              fontSize: 52, fontWeight: 600, fontFamily: "var(--mono)",
              color: "var(--green)", lineHeight: 1,
            }}>
              {(tuning?.final_score * 100).toFixed(1)}
              <span style={{ fontSize: 22 }}>%</span>
            </div>
            <div style={{
              fontSize: 11, color: "var(--muted)", marginTop: 6,
              textTransform: "uppercase", letterSpacing: "0.5px",
            }}>{metricLabel} (tuned)</div>
          </div>

          {/* Model info */}
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: 20, fontWeight: 600, marginBottom: 6 }}>
              {model_selection?.winner}
            </div>
            <div style={{ fontSize: 12, color: "var(--muted)", lineHeight: 1.7 }}>
              Selected from {Object.keys(model_selection?.all_models || {}).length} candidates
              <br />
              Tuned with GridSearchCV · 5-fold cross validation
              <br />
              Evaluated on held-out 20% test split
            </div>
          </div>
        </div>
      </Card>
    </>
  );
}
