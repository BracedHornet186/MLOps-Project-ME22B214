/**
 * UploadZone.jsx
 * ─────────────────────────────────────────────────────────────────
 * Drag-and-drop (or click) upload area for ZIP files.
 * Validates the file is a ZIP before enabling the upload button.
 */
import { useCallback, useRef, useState } from "react";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8002";

export default function UploadZone({ onJobCreated }) {
  const [file, setFile] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);
  const inputRef = useRef(null);

  /* ── Validate that the selected file is a ZIP ───────────────── */
  const handleFile = useCallback((f) => {
    setError(null);
    if (!f) return;
    const isZip =
      f.type === "application/zip" ||
      f.type === "application/x-zip-compressed" ||
      f.name.toLowerCase().endsWith(".zip");
    if (!isZip) {
      setError("Please select a valid .zip file");
      setFile(null);
      return;
    }
    setFile(f);
  }, []);

  /* ── Drag-and-drop handlers ─────────────────────────────────── */
  const onDragOver = (e) => {
    e.preventDefault();
    setDragOver(true);
  };
  const onDragLeave = () => setDragOver(false);
  const onDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    handleFile(e.dataTransfer.files[0]);
  };

  /* ── Upload to backend ──────────────────────────────────────── */
  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${API_BASE}/upload`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || `HTTP ${res.status}`);
      }
      const data = await res.json();
      onJobCreated(data.job_id);
    } catch (err) {
      setError(err.message || "Upload failed");
    } finally {
      setUploading(false);
    }
  };

  /* ── Format file size ───────────────────────────────────────── */
  const formatSize = (bytes) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <div
      className={`glass-card upload-zone fade-in ${dragOver ? "drag-over" : ""}`}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
      onClick={() => inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".zip"
        hidden
        onChange={(e) => handleFile(e.target.files[0])}
      />

      <span className="upload-icon">📦</span>

      <h3>{file ? file.name : "Drop your ZIP file here"}</h3>
      <p className="hint">
        Upload a .zip archive containing overlapping 2D images
      </p>

      {file && (
        <p className="file-info">
          {formatSize(file.size)} — ready to upload
        </p>
      )}

      {error && <p className="error-message">{error}</p>}

      {file && (
        <button
          className="upload-btn"
          disabled={uploading}
          onClick={(e) => {
            e.stopPropagation();
            handleUpload();
          }}
        >
          {uploading ? (
            <>
              <span className="spinner" style={{ width: 18, height: 18, borderWidth: 2 }} />
              Uploading…
            </>
          ) : (
            <>🚀 Start Reconstruction</>
          )}
        </button>
      )}
    </div>
  );
}
