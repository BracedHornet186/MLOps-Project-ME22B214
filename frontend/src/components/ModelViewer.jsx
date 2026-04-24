/**
 * ModelViewer.jsx
 * ─────────────────────────────────────────────────────────────────
 * Three.js point cloud viewer with multi-cluster dropdown.
 * Loads individual PLY files via /download/{jobId}/{filename}.
 */
import { Canvas, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader.js";
import axios from "axios";

const API = import.meta.env.VITE_API_URL || "/api";

/* ── Inner scene components ─────────────────────────────────────── */

function PointCloud({ geometry, pointSize }) {
  const ref = useRef();

  useEffect(() => {
    if (!geometry) return;
    geometry.computeBoundingBox();
    const center = new THREE.Vector3();
    geometry.boundingBox.getCenter(center);
    geometry.translate(-center.x, -center.y, -center.z);
  }, [geometry]);

  if (!geometry) return null;

  return (
    <points ref={ref}>
      <bufferGeometry attach="geometry" {...geometry} />
      <pointsMaterial
        attach="material"
        size={pointSize}
        vertexColors
        sizeAttenuation
        transparent
        opacity={0.9}
        depthWrite={false}
      />
    </points>
  );
}

function SceneSetup({ autoRotate }) {
  const { scene } = useThree();
  useEffect(() => {
    scene.background = new THREE.Color("#0f1117");
  }, [scene]);

  return (
    <>
      <ambientLight intensity={0.6} />
      <OrbitControls
        autoRotate={autoRotate}
        autoRotateSpeed={1.2}
        enableDamping
        dampingFactor={0.08}
        minDistance={0.2}
        maxDistance={50}
      />
    </>
  );
}

/* ── Main viewer component ──────────────────────────────────────── */

export default function ModelViewer({ jobId, clusters }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [geometry, setGeometry] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [pointSize, setPointSize] = useState(0.015);
  const [autoRotate, setAutoRotate] = useState(false);

  // Auto-select first cluster when clusters arrive
  useEffect(() => {
    if (clusters?.length > 0 && !selectedFile) {
      setSelectedFile(clusters[0].filename);
    }
  }, [clusters, selectedFile]);

  // Load PLY when selection changes
  useEffect(() => {
    if (!jobId || !selectedFile) return;
    setLoading(true);
    setError(null);
    setGeometry(null);

    const url = `${API}/download/${jobId}/${selectedFile}`;
    const loader = new PLYLoader();

    fetch(url)
      .then((res) => {
        if (!res.ok) throw new Error(`Download failed (HTTP ${res.status})`);
        return res.arrayBuffer();
      })
      .then((buffer) => {
        const geo = loader.parse(buffer);
        geo.computeVertexNormals();

        // Generate gradient colors if no vertex colors
        if (!geo.attributes.color) {
          const count = geo.attributes.position.count;
          const colors = new Float32Array(count * 3);
          const positions = geo.attributes.position.array;
          let yMin = Infinity,
            yMax = -Infinity;
          for (let i = 0; i < count; i++) {
            const y = positions[i * 3 + 1];
            yMin = Math.min(yMin, y);
            yMax = Math.max(yMax, y);
          }
          const range = yMax - yMin || 1;
          const c = new THREE.Color();
          for (let i = 0; i < count; i++) {
            const t = (positions[i * 3 + 1] - yMin) / range;
            c.setHSL(0.55 + t * 0.3, 0.8, 0.45 + t * 0.25);
            colors[i * 3] = c.r;
            colors[i * 3 + 1] = c.g;
            colors[i * 3 + 2] = c.b;
          }
          geo.setAttribute(
            "color",
            new THREE.Float32BufferAttribute(colors, 3)
          );
        }

        setGeometry(geo);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, [jobId, selectedFile]);

  const noClusters = !clusters || clusters.length === 0;

  return (
    <div className="panel viewer-panel" id="model-viewer">
      {/* Toolbar */}
      <div className="viewer-bar">
        <h3 className="panel-title" style={{ margin: 0 }}>
          3D Point Cloud
        </h3>

        <div className="viewer-controls">
          {!noClusters && clusters.length > 1 && (
            <select
              className="viewer-select"
              value={selectedFile || ""}
              onChange={(e) => setSelectedFile(e.target.value)}
              id="cluster-select"
            >
              {clusters.map((c) => (
                <option key={c.id} value={c.filename}>
                  {c.name} ({c.num_points3D?.toLocaleString()} pts)
                </option>
              ))}
            </select>
          )}

          <label className="viewer-ctrl-label">
            Size
            <input
              type="range"
              min="0.002"
              max="0.08"
              step="0.002"
              value={pointSize}
              onChange={(e) => setPointSize(parseFloat(e.target.value))}
            />
          </label>

          <label className="viewer-ctrl-label">
            <input
              type="checkbox"
              checked={autoRotate}
              onChange={(e) => setAutoRotate(e.target.checked)}
            />
            Rotate
          </label>
        </div>
      </div>

      {/* Canvas */}
      <div className="viewer-canvas">
        {loading && (
          <div className="viewer-overlay">
            <span className="spinner-sm" />
            <span>Loading point cloud…</span>
          </div>
        )}
        {error && (
          <div className="viewer-overlay">
            <span>⚠️ {error}</span>
          </div>
        )}
        {noClusters && !loading && (
          <div className="viewer-overlay">
            <span>No reconstruction available yet</span>
          </div>
        )}
        {!error && (
          <Canvas
            camera={{ position: [2, 1.5, 2], fov: 50, near: 0.01, far: 100 }}
            gl={{ antialias: true, alpha: false }}
            dpr={[1, 2]}
          >
            <SceneSetup autoRotate={autoRotate} />
            {geometry && (
              <PointCloud geometry={geometry} pointSize={pointSize} />
            )}
          </Canvas>
        )}
      </div>
    </div>
  );
}
