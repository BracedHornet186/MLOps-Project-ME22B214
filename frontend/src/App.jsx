/**
 * App.jsx
 * ──────────────────────────────────────────────────────────────────────────
 * Main application shell for the 3D Scene Reconstruction Web UI.
 *
 * State machine:
 *   IDLE  →  upload completes  →  PROCESSING  →  done  →  VIEWING
 *                                               ↘ failed → PROCESSING (with error)
 *   VIEWING  →  "New Job" button  →  IDLE
 */
import { useCallback, useState } from "react";
import UploadZone from "./components/UploadZone";
import ProgressTracker from "./components/ProgressTracker";
import PointCloudViewer from "./components/PointCloudViewer";
import "./index.css";

const PHASE = { IDLE: "idle", PROCESSING: "processing", VIEWING: "viewing" };

export default function App() {
  const [phase, setPhase] = useState(PHASE.IDLE);
  const [jobId, setJobId] = useState(null);
  const [nPoints, setNPoints] = useState(0);

  /* ── Callbacks ──────────────────────────────────────────────── */
  const onJobCreated = useCallback((id) => {
    setJobId(id);
    setPhase(PHASE.PROCESSING);
  }, []);

  const onComplete = useCallback((data) => {
    setNPoints(data.n_points || 0);
    setPhase(PHASE.VIEWING);
  }, []);

  const onFailed = useCallback(() => {
    /* stay on PROCESSING so the error message is visible */
  }, []);

  const reset = () => {
    setPhase(PHASE.IDLE);
    setJobId(null);
    setNPoints(0);
  };

  return (
    <>
      {/* Animated gradient background */}
      <div className="app-bg" />

      {/* Header */}
      <header className="app-header">
        <h1>Scene Reconstruct 3D</h1>
        <p>Upload overlapping 2D images → explore the 3D point cloud</p>
      </header>

      <main className="app-main">
        {/* ── Phase: IDLE ──────────────────────────────────────── */}
        {phase === PHASE.IDLE && <UploadZone onJobCreated={onJobCreated} />}

        {/* ── Phase: PROCESSING ────────────────────────────────── */}
        {phase === PHASE.PROCESSING && (
          <ProgressTracker
            jobId={jobId}
            onComplete={onComplete}
            onFailed={onFailed}
          />
        )}

        {/* ── Phase: VIEWING ───────────────────────────────────── */}
        {phase === PHASE.VIEWING && (
          <>
            <PointCloudViewer jobId={jobId} nPoints={nPoints} />
            <button className="upload-btn" onClick={reset}>
              🔄 Start New Reconstruction
            </button>
          </>
        )}
      </main>
    </>
  );
}
