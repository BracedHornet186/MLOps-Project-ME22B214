/**
 * ProgressTracker.jsx
 * ─────────────────────────────────────────────────────────────────
 * Polls the backend every 2 s for job status and displays an
 * animated multi-stage progress bar.
 */
import { useEffect, useRef, useState } from "react";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8002";

const STAGES = [
  { key: "extracting", label: "Extract", icon: "📂" },
  { key: "matching", label: "Match", icon: "🔗" },
  { key: "triangulating", label: "Triangulate", icon: "📐" },
  { key: "decimating", label: "Optimise", icon: "⚡" },
  { key: "done", label: "Done", icon: "✅" },
];

const STAGE_ORDER = STAGES.map((s) => s.key);

export default function ProgressTracker({ jobId, onComplete, onFailed }) {
  const [status, setStatus] = useState(null);
  const timerRef = useRef(null);

  useEffect(() => {
    if (!jobId) return;

    const poll = async () => {
      try {
        const res = await fetch(`${API_BASE}/status/${jobId}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        setStatus(data);

        if (data.stage === "done") {
          clearInterval(timerRef.current);
          onComplete?.(data);
        } else if (data.stage === "failed") {
          clearInterval(timerRef.current);
          onFailed?.(data);
        }
      } catch (err) {
        console.error("Polling error:", err);
        setStatus((prev) => 
          prev ? { ...prev, message: "Network error, retrying...", error: "Connection lost. Retrying..." } 
               : { stage: "queued", progress: 0, message: "Network error, retrying...", error: "Connection lost." }
        );
      }
    };

    poll(); // immediate first call
    timerRef.current = setInterval(poll, 2000);

    return () => clearInterval(timerRef.current);
  }, [jobId, onComplete, onFailed]);

  if (!status) return null;

  const currentIdx = STAGE_ORDER.indexOf(status.stage);

  return (
    <div className="glass-card progress-card fade-in">
      {/* Stage dots */}
      <div className="progress-stages">
        {STAGES.map((stage, i) => {
          let cls = "stage";
          if (status.stage === "failed" && i === currentIdx) cls += " failed";
          else if (i < currentIdx) cls += " completed";
          else if (i === currentIdx) cls += " active";
          return (
            <div key={stage.key} className={cls}>
              <div className="stage-dot">{stage.icon}</div>
              <span className="stage-label">{stage.label}</span>
            </div>
          );
        })}
      </div>

      {/* Progress bar */}
      <div className="progress-bar-track">
        <div
          className="progress-bar-fill"
          style={{ width: `${status.progress}%` }}
        />
      </div>

      {/* Status message */}
      <p className="progress-message">
        <span className="percentage">{status.progress}%</span>
        {" — "}
        {status.message}
      </p>

      {/* Metadata */}
      {status.n_images > 0 && (
        <p className="progress-message" style={{ marginTop: 4, fontSize: "0.82rem" }}>
          {status.n_images} images
          {status.n_points > 0 && ` · ${status.n_points.toLocaleString()} points`}
        </p>
      )}

      {/* Error */}
      {status.stage === "failed" && status.error && (
        <div className="error-message">{status.error}</div>
      )}
    </div>
  );
}
