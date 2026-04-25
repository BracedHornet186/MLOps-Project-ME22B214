/**
 * App.jsx
 * ─────────────────────────────────────────────────────────────────
 * Dashboard layout for 3D Reconstruction system.
 *
 *  ┌──────────────── Header ─────────────────┐
 *  │ Left Panel          │  Right Panel       │
 *  │  Upload + Stages    │  3D Viewer         │
 *  ├─────────────────────┴────────────────────┤
 *  │          Stats Table (bottom)            │
 *  └──────────────────────────────────────────┘
 */
import { useState, useEffect } from "react";
import apiClient from "./api";
import Header from "./components/Header";
import UploadPanel from "./components/UploadPanel";
import StageTracker from "./components/StageTracker";
import StatsTable from "./components/StatsTable";
import ModelViewer from "./components/ModelViewer";
import Login from "./components/Login";
import useJobStatus from "./hooks/useJobStatus";
import "./index.css";

export default function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(
    !!localStorage.getItem("access_token")
  );
  const [jobId, setJobId] = useState(null);
  const [clusters, setClusters] = useState([]);
  const { status } = useJobStatus(jobId);

  const stage = status?.stage || null;
  const isDone = stage === "success";
  const isRunning = jobId && !isDone && stage !== "failed";

  // Listen for unauthorized events to clear auth state
  useEffect(() => {
    const handleUnauthorized = () => setIsAuthenticated(false);
    window.addEventListener("auth:unauthorized", handleUnauthorized);
    return () =>
      window.removeEventListener("auth:unauthorized", handleUnauthorized);
  }, []);

  // Fetch clusters when job completes
  useEffect(() => {
    if (!isDone || !jobId) return;
    apiClient
      .get(`/clusters/${jobId}`)
      .then(({ data }) => setClusters(data.clusters || []))
      .catch(() => setClusters([]));
  }, [isDone, jobId]);

  const handleNewJob = (id) => {
    setJobId(id);
    setClusters([]);
  };

  const handleReset = () => {
    setJobId(null);
    setClusters([]);
  };

  if (!isAuthenticated) {
    return <Login onLoginSuccess={() => setIsAuthenticated(true)} />;
  }

  return (
    <div className="dashboard">
      <Header jobId={jobId} stage={stage} />

      <div className="dashboard-body">
        {/* ── Left Panel ──────────────────────────────────────── */}
        <aside className="dashboard-left">
          <UploadPanel onJobCreated={handleNewJob} disabled={isRunning} />

          {status && <StageTracker status={status} />}

          {isDone && (
            <button className="btn-secondary" onClick={handleReset}>
              New Reconstruction
            </button>
          )}
        </aside>

        {/* ── Right Panel ─────────────────────────────────────── */}
        <main className="dashboard-right">
          <ModelViewer jobId={jobId} clusters={clusters} />
        </main>
      </div>

      {/* ── Bottom Stats ──────────────────────────────────────── */}
      {status && <StatsTable jobId={jobId} status={status} clusters={clusters} />}
    </div>
  );
}
