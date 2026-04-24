/**
 * Header.jsx
 * ─────────────────────────────────────────────────────────────────
 * Fixed header bar with app title, job ID, status, and GPU info.
 */
import useGpuStatus from "../hooks/useGpuStatus";

export default function Header({ jobId, stage }) {
  const { gpu } = useGpuStatus();

  const stageLabel = stage
    ? stage.charAt(0).toUpperCase() + stage.slice(1)
    : "Idle";

  return (
    <header className="dashboard-header">
      <div className="header-left">
        <h1 className="header-title">3D Reconstruction Dashboard</h1>
      </div>

      <div className="header-meta">
        {jobId && (
          <span className="header-chip" id="header-job-id">
            Job: <code>{jobId.slice(0, 8)}</code>
          </span>
        )}

        <span
          className={`header-chip ${
            stage === "success"
              ? "chip-success"
              : stage === "failed"
              ? "chip-error"
              : stage
              ? "chip-active"
              : ""
          }`}
          id="header-status"
        >
          {stageLabel}
        </span>

        <span
          className={`header-chip ${gpu.ready ? "chip-success" : "chip-error"}`}
          id="header-gpu"
        >
          GPU: {gpu.ready ? gpu.device : "NOT READY"}
        </span>
      </div>
    </header>
  );
}
