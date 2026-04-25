/**
 * StatsTable.jsx
 * ─────────────────────────────────────────────────────────────────
 * Reconstruction statistics table showing per-cluster stats.
 */
import { useEffect, useState } from "react";
import apiClient from "../api";

export default function StatsTable({ jobId, status, clusters = [] }) {
  if (!status) return null;

  const isDone = status.stage === "success";
  const hasCluster = isDone && clusters.length > 0;

  return (
    <div className="panel stats-panel" id="stats-table">
      <h3 className="panel-title">Reconstruction Statistics</h3>

      {hasCluster && (
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
                    <button
                      onClick={async () => {
                        try {
                          const response = await apiClient.get(`/download/jobs/${jobId}/${c.filename}`, { responseType: 'blob' });
                          const url = window.URL.createObjectURL(new Blob([response.data]));
                          const link = document.createElement('a');
                          link.href = url;
                          link.setAttribute('download', c.filename);
                          document.body.appendChild(link);
                          link.click();
                          link.remove();
                        } catch (err) {
                          console.error("Download failed:", err);
                        }
                      }}
                      className="btn-download"
                      title="Download PLY"
                      style={{ background: 'none', border: 'none', cursor: 'pointer' }}
                    >
                      ⬇️
                    </button>
                  </td>
                </tr>
              ))}
              <tr>
                <td>Submission CSV</td>
                <td className="num">—</td>
                <td className="mono" style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                  <span>submission_{jobId.slice(0, 8)}.csv</span>
                  <button
                    onClick={async () => {
                      try {
                        const response = await apiClient.get(`/download/jobs/${jobId}/csv`, { responseType: 'blob' });
                        const url = window.URL.createObjectURL(new Blob([response.data]));
                        const link = document.createElement('a');
                        link.href = url;
                        link.setAttribute('download', `submission_${jobId.slice(0, 8)}.csv`);
                        document.body.appendChild(link);
                        link.click();
                        link.remove();
                      } catch (err) {
                        console.error("Download failed:", err);
                      }
                    }}
                    className="btn-download"
                    title="Download CSV"
                    style={{ background: 'none', border: 'none', cursor: 'pointer' }}
                  >
                    ⬇️
                  </button>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      )}

      {isDone && clusters.length === 0 && (
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
