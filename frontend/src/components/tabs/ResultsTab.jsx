// components/tabs/ResultsTab.jsx

import Card from "../Card";

export default function ResultsTab({ results }) {
  const { tuning, explanation, problem, model_selection } = results;
  const metricLabel = problem?.type === "regression" ? "R² Score" : "F1 Score";

  return (
    <>
      {/* Tuned Hyperparameters */}
      <Card title="Best hyperparameters" accentColor="var(--amber)">
        <p style={{ fontSize: 12, color: "var(--muted)", marginBottom: 16 }}>
          Found via GridSearchCV with 5-fold cross validation on training data.
        </p>

        {Object.keys(tuning?.best_params || {}).length > 0 ? (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: 8 }}>
            {Object.entries(tuning.best_params).map(([key, val]) => (
              <div key={key} style={{ background: "var(--surface2)", borderRadius: 8, padding: 14 }}>
                <div style={{ fontSize: 10, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: 6 }}>
                  {key.replace(/_/g, " ")}
                </div>
                <div style={{ fontSize: 15, fontFamily: "var(--mono)", color: "var(--amber)" }}>
                  {String(val)}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div style={{ fontSize: 12, color: "var(--muted)" }}>No grid search params available.</div>
        )}

        <div style={{ marginTop: 20, paddingTop: 18, borderTop: "1px solid var(--border)", display: "flex", alignItems: "center", gap: 24, flexWrap: "wrap" }}>
          <div>
            <div style={{ fontSize: 40, fontWeight: 600, fontFamily: "var(--mono)", color: "var(--green)", lineHeight: 1 }}>
              {(tuning?.final_score * 100).toFixed(2)}<span style={{ fontSize: 18 }}>%</span>
            </div>
            <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 5, textTransform: "uppercase", letterSpacing: "0.5px" }}>
              {metricLabel} on test set
            </div>
          </div>
          <div style={{ fontSize: 12, color: "var(--muted)", lineHeight: 1.7 }}>
            Model: <span style={{ color: "var(--text)" }}>{model_selection?.winner}</span><br />
            Tuned then retrained on full training split (80%)<br />
            Evaluated on never-seen test split (20%)
          </div>
        </div>
      </Card>

      {/* SHAP chart — base64 directly */}
      {explanation?.shap_chart ? (
        <Card title="SHAP feature importance" accentColor="var(--green)">
          <p style={{ fontSize: 12, color: "var(--muted)", marginBottom: 12 }}>
            SHapley Additive exPlanations — longer bar = more influence on predictions.
          </p>
          <img
            src={`data:image/png;base64,${explanation.shap_chart}`}
            alt="SHAP Feature Importance"
            style={{ width: "100%", borderRadius: 8, background: "var(--surface2)", border: "1px solid var(--border)", display: "block" }}
          />
        </Card>
      ) : (
        <Card title="SHAP feature importance" accentColor="var(--green)">
          <div style={{ fontSize: 13, color: "var(--muted)" }}>SHAP analysis not available for this run.</div>
        </Card>
      )}
    </>
  );
}