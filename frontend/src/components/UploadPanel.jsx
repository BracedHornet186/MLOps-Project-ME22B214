/**
 * UploadPanel.jsx
 * ─────────────────────────────────────────────────────────────────
 * Compact upload panel for the left sidebar.
 * Validates ZIP files. Disables while a job is running.
 */
import { useCallback, useRef, useState } from "react";
import apiClient from "../api";

export default function UploadPanel({ onJobCreated, disabled }) {
  const [file, setFile] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);
  const inputRef = useRef(null);

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

  const handleUpload = async () => {
    if (!file || disabled) return;
    setUploading(true);
    setError(null);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const { data } = await apiClient.post("/upload", formData);
      onJobCreated(data.job_id);
      setFile(null);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || "Upload failed");
    } finally {
      setUploading(false);
    }
  };

  const formatSize = (bytes) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1048576).toFixed(1)} MB`;
  };

  const isDisabled = disabled || uploading;

  return (
    <div className="panel upload-panel" id="upload-panel">
      <h3 className="panel-title">Upload Images</h3>

      <div
        className={`upload-dropzone ${dragOver ? "drag-active" : ""} ${
          isDisabled ? "disabled" : ""
        }`}
        onDragOver={(e) => {
          e.preventDefault();
          if (!isDisabled) setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          if (!isDisabled) handleFile(e.dataTransfer.files[0]);
        }}
        onClick={() => !isDisabled && inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".zip"
          hidden
          onChange={(e) => handleFile(e.target.files[0])}
        />
        <span className="dropzone-icon">📦</span>
        <p className="dropzone-text">
          {file ? file.name : "Drop ZIP here or click"}
        </p>
        {file && (
          <p className="dropzone-size">{formatSize(file.size)}</p>
        )}
      </div>

      {error && <p className="panel-error">{error}</p>}

      <button
        className="btn-primary"
        disabled={!file || isDisabled}
        onClick={handleUpload}
        id="upload-btn"
      >
        {uploading ? (
          <>
            <span className="spinner-sm" /> Uploading…
          </>
        ) : (
          "Start Reconstruction"
        )}
      </button>
    </div>
  );
}
