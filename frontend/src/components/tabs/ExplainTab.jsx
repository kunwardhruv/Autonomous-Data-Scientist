// components/tabs/ExplainTab.jsx

import Card from "../Card";

export default function ExplainTab({ results }) {
  const { explanation, model_selection, problem, tuning } = results;
  const text = explanation?.llm_text;
  const hasText = text && !text.startsWith("⚠️") && !text.startsWith("Could not");

  return (
    <>
      <Card title="AI explanation" accentColor="var(--blue)">
        {/* Header row */}
        <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 20 }}>
          <span style={{
            background: "rgba(96,165,250,0.1)",
            border: "1px solid rgba(96,165,250,0.25)",
            color: "var(--blue)", fontSize: 10,
            fontFamily: "var(--mono)", padding: "3px 8px",
            borderRadius: 4, letterSpacing: "0.5px",
          }}>
            Groq · Llama 3.3 70B
          </span>
          <span style={{
            background: "rgba(74,222,128,0.08)",
            border: "1px solid rgba(74,222,128,0.2)",
            color: "var(--green)", fontSize: 10,
            fontFamily: "var(--mono)", padding: "3px 8px",
            borderRadius: 4, letterSpacing: "0.5px",
          }}>
            {model_selection?.winner} · {(tuning?.final_score * 100).toFixed(1)}%
          </span>
        </div>

        {/* Explanation text */}
        {hasText ? (
          <div style={{
            fontSize: 13.5, lineHeight: 1.85, color: "#c8cad8",
            whiteSpace: "pre-wrap", fontWeight: 300,
            borderLeft: "2px solid var(--border2)",
            paddingLeft: 16,
          }}>
            {text}
          </div>
        ) : (
          <div style={{
            background: "rgba(245,166,35,0.06)",
            border: "1px solid rgba(245,166,35,0.2)",
            borderRadius: 8, padding: 16,
            fontSize: 13, color: "var(--amber)",
          }}>
            <strong>GROQ_API_KEY not set.</strong>
            <br /><br />
            <span style={{ color: "var(--muted)", fontSize: 12 }}>
              1. Get a free key from{" "}
              <a
                href="https://console.groq.com"
                target="_blank"
                rel="noreferrer"
                style={{ color: "var(--blue)" }}
              >
                console.groq.com
              </a>
              <br />
              2. Add to your <code style={{ fontFamily: "var(--mono)", background: "var(--surface3)", padding: "1px 4px", borderRadius: 3 }}>.env</code> file:{" "}
              <code style={{ fontFamily: "var(--mono)", background: "var(--surface3)", padding: "1px 4px", borderRadius: 3 }}>
                GROQ_API_KEY=gsk_...
              </code>
              <br />
              3. Restart the FastAPI server
            </span>
          </div>
        )}
      </Card>

      {/* What each section means */}
      <Card title="How to read this" accentColor="var(--muted)">
        <div style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
          gap: 12, fontSize: 12, color: "var(--muted)",
        }}>
          {[
            { label: "Dataset overview", desc: "Shape, column types, any data quality issues noticed by the LLM" },
            { label: "Why this model", desc: `Why ${model_selection?.winner || "the winner"} likely outperformed others on this data` },
            { label: "Feature drivers", desc: "Which columns drove predictions most (backed by SHAP values)" },
            { label: "Actionable insight", desc: "One concrete thing to do with this analysis" },
          ].map(({ label, desc }) => (
            <div key={label} style={{
              background: "var(--surface2)", borderRadius: 8, padding: 12,
            }}>
              <div style={{
                fontSize: 11, fontWeight: 600, color: "var(--text)",
                marginBottom: 5, textTransform: "uppercase", letterSpacing: "0.5px",
              }}>
                {label}
              </div>
              <div>{desc}</div>
            </div>
          ))}
        </div>
      </Card>
    </>
  );
}
