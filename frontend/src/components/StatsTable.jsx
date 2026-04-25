/**
 * StatsTable.jsx
 * ─────────────────────────────────────────────────────────────────
 * Reconstruction statistics table showing per-cluster stats.
 */
import { useEffect, useState } from "react";
import axios from "axios";

const API = import.meta.env.VITE_API_URL || "/api";

export default function StatsTable({ jobId, status }) {
  const [clusters, setClusters] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!jobId || status?.stage !== "success") return;
    setLoading(true);
    axios
      .get(`${API}/clusters/${jobId}`)
      .then(({ data }) => setClusters(data.clusters || []))
      .catch(() => setClusters([]))
      .finally(() => setLoading(false));
  }, [jobId, status?.stage]);

  if (!status) return null;

  const isDone = status.stage === "success";
  const hasCluster = isDone && clusters.length > 0;

  return (
    <div className="panel stats-panel" id="stats-table">
      <h3 className="panel-title">Reconstruction Statistics</h3>

      {loading && <p className="stats-loading"><span className="spinner-sm" /> Loading cluster data…</p>}

      {!loading && hasCluster && (
        <div className="table-wrap">
          <table className="stats-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Points3D</th>
                <th>File</th>
              </tr>
            </thead>
            <tbody>
              {clusters.map((c) => (
                <tr key={c.id}>
                  <td>{c.name}</td>
                  <td className="num">{c.num_points3D?.toLocaleString() ?? "—"}</td>
                  <td className="mono" style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                    <span>{c.filename}</span>
                    <a
                      href={`${API}/download/${jobId}/${c.filename}`}
                      download
                      className="btn-download"
                      title="Download PLY"
                    >
                      ⬇️
                    </a>
                  </td>
                </tr>
              ))}
              <tr>
                <td>Submission CSV</td>
                <td className="num">—</td>
                <td className="mono" style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                  <span>submission_{jobId.slice(0, 8)}.csv</span>
                  <a
                    href={`${API}/jobs/${jobId}/download`}
                    download
                    className="btn-download"
                    title="Download CSV"
                  >
                    ⬇️
                  </a>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      )}

      {!loading && isDone && clusters.length === 0 && (
        <p className="stats-loading">No clusters found for this reconstruction.</p>
      )}

      {/* Job info row */}
      {isDone && (
        <div className="stats-summary" style={{ marginTop: 12 }}>
          <div className="stat-card">
            <span className="stat-value">{status.n_images || "—"}</span>
            <span className="stat-label">Total Images</span>
          </div>
          <div className="stat-card">
            <span className="stat-value">
              {status.registration_rate != null
                ? `${(status.registration_rate * 100).toFixed(1)}%`
                : "—"}
            </span>
            <span className="stat-label">Registration</span>
          </div>
          {status.finished_at && status.started_at && (
            <div className="stat-card">
              <span className="stat-value">
                {(status.finished_at - status.started_at).toFixed(1)}s
              </span>
              <span className="stat-label">Latency</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
