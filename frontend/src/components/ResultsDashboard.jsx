// components/ResultsDashboard.jsx
//
// Yeh component results receive karta hai aur 5 tabs manage karta hai.
// Har tab ek alag component hai — clean separation of concerns.

import { useState } from "react";
import OverviewTab from "./tabs/OverviewTab";
import EDATab      from "./tabs/EDATab";
import ModelsTab   from "./tabs/ModelsTab";
import ResultsTab  from "./tabs/ResultsTab";
import ExplainTab  from "./tabs/ExplainTab";

const TABS = [
  { id: "overview", label: "Overview", dot: "var(--amber)"  },
  { id: "eda",      label: "EDA",      dot: "var(--blue)"   },
  { id: "models",   label: "Models",   dot: "var(--purple)" },
  { id: "results",  label: "Results",  dot: "var(--green)"  },
  { id: "explain",  label: "Explain",  dot: "var(--amber2)" },
];

export default function ResultsDashboard({ results, onReset, apiBase }) {
  const [activeTab, setActiveTab] = useState("overview");

  return (
    <>
      {/* Top bar: success message + re-analyze button */}
      <div style={{
        display: "flex", alignItems: "center",
        justifyContent: "space-between", marginBottom: 4,
      }}>
        <div style={{ fontSize: 13, color: "var(--green)" }}>
          ✓ Analysis complete — {results.eda?.shape?.rows?.toLocaleString()} rows processed
        </div>
        <button
          onClick={onReset}
          style={{
            padding: "7px 14px", borderRadius: 8, fontSize: 12,
            cursor: "pointer", background: "transparent",
            color: "var(--muted)", border: "1px solid var(--border2)",
            fontFamily: "var(--sans)", transition: "all 0.15s",
          }}
          onMouseEnter={e => e.target.style.color = "var(--text)"}
          onMouseLeave={e => e.target.style.color = "var(--muted)"}
        >
          ↑ New CSV
        </button>
      </div>

      {/* Tab bar */}
      <div style={{
        display: "flex", gap: 2,
        background: "var(--surface)",
        border: "1px solid var(--border)",
        borderRadius: 10, padding: 4,
        margin: "20px 0 16px",
        overflowX: "auto",
      }}>
        {TABS.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            style={{
              padding: "7px 16px", borderRadius: 7,
              fontSize: 12, fontWeight: 500, cursor: "pointer",
              border: "none", fontFamily: "var(--sans)",
              transition: "all 0.15s",
              background: activeTab === tab.id ? "var(--surface3)" : "transparent",
              color: activeTab === tab.id ? "var(--text)" : "var(--muted)",
              display: "flex", alignItems: "center", gap: 6,
              whiteSpace: "nowrap",
            }}
          >
            <div style={{
              width: 6, height: 6, borderRadius: "50%",
              background: activeTab === tab.id ? tab.dot : "var(--dim)",
              flexShrink: 0,
            }} />
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      {activeTab === "overview" && <OverviewTab results={results} />}
      {activeTab === "eda"      && <EDATab      results={results} apiBase={apiBase} />}
      {activeTab === "models"   && <ModelsTab   results={results} />}
      {activeTab === "results"  && <ResultsTab  results={results} apiBase={apiBase} />}
      {activeTab === "explain"  && <ExplainTab  results={results} />}
    </>
  );
}
