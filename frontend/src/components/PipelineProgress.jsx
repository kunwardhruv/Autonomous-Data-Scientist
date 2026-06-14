// components/PipelineProgress.jsx
//
// Loading ke time yeh component dikhata hai — animated pipeline steps.
// Fake progress hai (timer based), real progress ke liye SSE/WebSocket chahiye.
// But visually user ko feel hota hai kuch ho raha hai.

import { useState, useEffect } from "react";

const STEPS = [
  { label: "Node 1 — Loading & validating CSV",        icon: "📂" },
  { label: "Node 2 — Running EDA & generating charts",  icon: "📊" },
  { label: "Node 3 — Detecting problem type",           icon: "🔍" },
  { label: "Node 4 — Preparing features",               icon: "⚙️" },
  { label: "Node 5 — Selecting best model",             icon: "🤖" },
  { label: "Node 6 — Hyperparameter tuning",            icon: "🎯" },
  { label: "Node 7 — SHAP analysis & LLM explanation", icon: "✨" },
];

export default function PipelineProgress() {
  const [activeStep, setActiveStep] = useState(0);

  useEffect(() => {
    // Har 1.4s mein next step
    const iv = setInterval(() => {
      setActiveStep(prev => (prev < STEPS.length - 1 ? prev + 1 : prev));
    }, 1400);
    return () => clearInterval(iv);
  }, []);

  return (
    <div style={{
      background: "var(--surface)", border: "1px solid var(--border)",
      borderRadius: 12, padding: 20, marginTop: 24,
    }}>
      {/* Title */}
      <div style={{
        fontSize: 11, fontWeight: 600, letterSpacing: 1,
        textTransform: "uppercase", color: "var(--muted)",
        display: "flex", alignItems: "center", gap: 8, marginBottom: 16,
      }}>
        <div style={{ width: 16, height: 2, borderRadius: 1, background: "var(--amber)" }} />
        Pipeline running
      </div>

      {/* Indeterminate progress bar */}
      <div style={{
        height: 3, background: "var(--border)", borderRadius: 2,
        overflow: "hidden", marginBottom: 20,
      }}>
        <div style={{
          height: "100%", borderRadius: 2,
          background: "linear-gradient(90deg, var(--amber), var(--amber2))",
          animation: "slide 1.5s ease-in-out infinite",
        }} />
      </div>

      {/* Step list */}
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        {STEPS.map((step, i) => {
          const isDone   = i < activeStep;
          const isActive = i === activeStep;
          const isWait   = i > activeStep;

          return (
            <div
              key={i}
              style={{
                display: "flex", alignItems: "center", gap: 10,
                padding: "8px 12px", borderRadius: 8,
                fontSize: 12, fontFamily: "var(--mono)",
                transition: "all 0.3s",
                background: isActive
                  ? "rgba(245,166,35,0.08)"
                  : isDone ? "rgba(74,222,128,0.04)" : "var(--surface2)",
                border: `1px solid ${isActive ? "rgba(245,166,35,0.2)" : "transparent"}`,
                color: isDone ? "var(--green)" : isActive ? "var(--amber)" : "var(--dim)",
              }}
            >
              {/* Dot */}
              <div style={{
                width: 8, height: 8, borderRadius: "50%", flexShrink: 0,
                background: isDone ? "var(--green)" : isActive ? "var(--amber)" : "var(--dim)",
                animation: isActive ? "pulse 1s infinite" : "none",
              }} />
              <span>{isDone ? "✓ " : ""}{step.icon} {step.label}</span>
            </div>
          );
        })}
      </div>

      {/* CSS animations (injected once) */}
      <style>{`
        @keyframes slide {
          0%   { width: 0%;   margin-left: 0; }
          50%  { width: 60%;  margin-left: 20%; }
          100% { width: 0%;   margin-left: 100%; }
        }
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50%       { opacity: 0.3; }
        }
      `}</style>
    </div>
  );
}
