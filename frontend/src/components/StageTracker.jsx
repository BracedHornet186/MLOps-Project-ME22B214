/**
 * StageTracker.jsx
 * ─────────────────────────────────────────────────────────────────
 * Vertical pipeline stage tracker for the left sidebar.
 * Maps status.stage + status.message to visual stage indicators.
 */

const STAGES = [
  { key: "extracting",    label: "Global Descriptor Extraction", icon: "📂" },
  { key: "matching",      label: "Local Feature Extraction & Point Tracking", icon: "🔗" },
  { key: "triangulating", label: "Retriangulation & Bundle Adjustment", icon: "📐" },
  { key: "decimating",    label: "Point Cloud Optimization", icon: "⚡" },
  { key: "success",       label: "3D Reconstruction Complete", icon: "✅" },
];

const STAGE_ORDER = STAGES.map((s) => s.key);

export default function StageTracker({ status }) {
  if (!status) return null;

  const currentIdx = STAGE_ORDER.indexOf(status.stage);

  return (
    <div className="panel stage-panel" id="stage-tracker">
      <h3 className="panel-title">Pipeline Stages</h3>

      <div className="stage-list">
        {STAGES.map((stage, i) => {
          let stateClass = "";
          if (status.stage === "failed" && i === currentIdx)
            stateClass = "stage-failed";
          else if (i < currentIdx) stateClass = "stage-done";
          else if (i === currentIdx) stateClass = "stage-current";
          else stateClass = "stage-pending";

          return (
            <div key={stage.key} className={`stage-item ${stateClass}`}>
              <div className="stage-indicator">
                {stateClass === "stage-done" ? (
                  <span className="check">✓</span>
                ) : stateClass === "stage-current" ? (
                  <span className="arrow">→</span>
                ) : stateClass === "stage-failed" ? (
                  <span className="cross">✕</span>
                ) : (
                  <span className="dot" />
                )}
              </div>
              <span className="stage-text">{stage.label}</span>
            </div>
          );
        })}
      </div>

      {/* Progress bar */}
      <div className="stage-progress-bar">
        <div
          className="stage-progress-fill"
          style={{ width: `${status.progress || 0}%` }}
        />
      </div>
      <p className="stage-message">
        <strong>{status.progress}%</strong> — {status.message}
      </p>

      {status.stage === "failed" && status.error && (
        <div className="panel-error">{status.error}</div>
      )}
    </div>
  );
}
