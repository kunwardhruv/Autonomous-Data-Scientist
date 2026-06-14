// components/tabs/ModelsTab.jsx

import Card from "../Card";

export default function ModelsTab({ results }) {
  const { model_selection, problem } = results;
  const allModels = model_selection?.all_models || {};
  const winner    = model_selection?.winner;

  // Regression = rank by r2, classification = rank by f1
  const rankKey = problem?.type === "regression" ? "r2" : "f1";

  const sortedModels = Object.entries(allModels).sort(
    (a, b) => (b[1][rankKey] || 0) - (a[1][rankKey] || 0)
  );

  // Get all metric keys from first model
  const metricKeys = Object.keys(Object.values(allModels)[0] || {});

  // Bar color: winner = green, rest = purple
  const barColor = (name) => name === winner ? "var(--green)" : "var(--purple)";

  return (
    <>
      <Card title="Model comparison" accentColor="var(--purple)">
        <p style={{ fontSize: 12, color: "var(--muted)", marginBottom: 16 }}>
          All models trained on 80% data, evaluated on held-out 20% test split.
          {problem?.type === "classification"
            ? " F1 used for ranking (handles class imbalance better than accuracy)."
            : " R² used for ranking (1.0 = perfect fit, 0 = just predicting mean)."}
        </p>

        <div style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
            <thead>
              <tr style={{ borderBottom: "1px solid var(--border2)" }}>
                <th style={thStyle}>Model</th>
                {metricKeys.map(k => (
                  <th key={k} style={thStyle}>{k.toUpperCase()}</th>
                ))}
                <th style={{ ...thStyle, minWidth: 160 }}>Score bar</th>
              </tr>
            </thead>
            <tbody>
              {sortedModels.map(([name, metrics], idx) => {
                const isWinner  = name === winner;
                const rankScore = metrics[rankKey] || 0;

                return (
                  <tr
                    key={name}
                    style={{
                      background: isWinner ? "rgba(74,222,128,0.04)" : "transparent",
                      borderBottom: "1px solid var(--border)",
                    }}
                  >
                    {/* Model name */}
                    <td style={{
                      padding: "11px 12px", fontFamily: "var(--mono)",
                      color: isWinner ? "var(--green)" : "var(--text)",
                      whiteSpace: "nowrap",
                    }}>
                      {isWinner ? "★ " : `${idx + 1}. `}{name}
                    </td>

                    {/* Metric values */}
                    {metricKeys.map(k => (
                      <td key={k} style={{
                        padding: "11px 12px", fontFamily: "var(--mono)",
                        color: k === rankKey
                          ? (isWinner ? "var(--green)" : "var(--text)")
                          : "var(--muted)",
                      }}>
                        {metrics[k]?.toFixed(4)}
                      </td>
                    ))}

                    {/* Bar */}
                    <td style={{ padding: "11px 12px" }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                        <div style={{
                          flex: 1, height: 6, background: "var(--surface3)",
                          borderRadius: 3, overflow: "hidden",
                        }}>
                          <div style={{
                            width: `${Math.max(0, rankScore) * 100}%`,
                            height: "100%",
                            background: barColor(name),
                            borderRadius: 3,
                            transition: "width 0.6s ease",
                          }} />
                        </div>
                        <span style={{
                          fontSize: 10, color: "var(--muted)",
                          minWidth: 36, fontFamily: "var(--mono)",
                        }}>
                          {(rankScore * 100).toFixed(1)}%
                        </span>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Why winner won */}
      <Card title="Why this model?" accentColor="var(--amber)">
        <div style={{ fontSize: 13, color: "var(--muted)", lineHeight: 1.8 }}>
          <span style={{ color: "var(--text)", fontWeight: 500 }}>{winner}</span> was selected
          because it achieved the highest{" "}
          <span style={{ fontFamily: "var(--mono)", color: "var(--amber)" }}>{rankKey}</span>{" "}
          score on the test set among all {sortedModels.length} candidates.
          <br /><br />
          The LLM explanation in the <em>Explain</em> tab provides a deeper analysis of why
          this model likely performed well for this particular dataset.
        </div>
      </Card>
    </>
  );
}

const thStyle = {
  padding: "8px 12px", textAlign: "left", color: "var(--muted)",
  fontWeight: 500, fontSize: 11, letterSpacing: "0.5px",
  textTransform: "uppercase", whiteSpace: "nowrap",
};
