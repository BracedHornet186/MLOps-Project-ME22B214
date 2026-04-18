/**
 * PointCloudViewer.jsx
 * ─────────────────────────────────────────────────────────────────
 * Three.js-powered 3D point cloud viewer using @react-three/fiber.
 * Loads a .ply file from the backend, renders it with orbit controls,
 * and provides UI knobs for point size, auto-rotate, and background.
 */
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader.js";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8002";

/* ══════════════════════════════════════════════════════════════════
   Inner Three.js scene (rendered inside <Canvas>)
   ══════════════════════════════════════════════════════════════════ */
function PointCloud({ geometry, pointSize }) {
  const ref = useRef();

  /* Centre the cloud on the scene origin */
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

function SceneSetup({ autoRotate, bgColor }) {
  const { scene } = useThree();
  useEffect(() => {
    scene.background = new THREE.Color(bgColor);
  }, [bgColor, scene]);

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

/* ══════════════════════════════════════════════════════════════════
   Main exported component
   ══════════════════════════════════════════════════════════════════ */
export default function PointCloudViewer({ jobId, nPoints }) {
  const [geometry, setGeometry] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [pointSize, setPointSize] = useState(0.015);
  const [autoRotate, setAutoRotate] = useState(true);
  const [darkBg, setDarkBg] = useState(true);

  const bgColor = darkBg ? "#060a14" : "#e2e8f0";

  /* ── Load PLY from backend ──────────────────────────────────── */
  useEffect(() => {
    if (!jobId) return;
    setLoading(true);
    setError(null);

    const loader = new PLYLoader();
    const url = `${API_BASE}/download/${jobId}`;

    fetch(url)
      .then((res) => {
        if (!res.ok) throw new Error(`Download failed (HTTP ${res.status})`);
        return res.arrayBuffer();
      })
      .then((buffer) => {
        const geo = loader.parse(buffer);
        geo.computeVertexNormals();

        // If no vertex colors, generate a gradient palette
        if (!geo.attributes.color) {
          const count = geo.attributes.position.count;
          const colors = new Float32Array(count * 3);
          const positions = geo.attributes.position.array;
          let yMin = Infinity, yMax = -Infinity;
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
          geo.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
        }

        setGeometry(geo);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, [jobId]);

  return (
    <div className="glass-card viewer-card fade-in">
      {/* Toolbar */}
      <div className="viewer-toolbar">
        <h3>
          🌐 3D Point Cloud
          {nPoints > 0 && (
            <span className="point-count-badge">
              {nPoints.toLocaleString()} pts
            </span>
          )}
        </h3>

        <div className="viewer-controls">
          <label>
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
          <label>
            <input
              type="checkbox"
              checked={autoRotate}
              onChange={(e) => setAutoRotate(e.target.checked)}
            />
            Auto-rotate
          </label>
          <label>
            <input
              type="checkbox"
              checked={darkBg}
              onChange={(e) => setDarkBg(e.target.checked)}
            />
            Dark
          </label>
        </div>
      </div>

      {/* Canvas */}
      <div className="viewer-canvas-wrapper">
        {loading && (
          <div className="viewer-loading">
            <div className="spinner" />
            Loading point cloud…
          </div>
        )}
        {error && (
          <div className="viewer-loading">
            <span style={{ fontSize: "2rem" }}>⚠️</span>
            <span style={{ color: "#fca5a5" }}>{error}</span>
          </div>
        )}
        {!error && (
          <Canvas
            camera={{ position: [2, 1.5, 2], fov: 50, near: 0.01, far: 100 }}
            gl={{ antialias: true, alpha: false }}
            dpr={[1, 2]}
          >
            <SceneSetup autoRotate={autoRotate} bgColor={bgColor} />
            {geometry && <PointCloud geometry={geometry} pointSize={pointSize} />}
          </Canvas>
        )}
      </div>
    </div>
  );
}
