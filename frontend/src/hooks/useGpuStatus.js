/**
 * useGpuStatus.js
 * ─────────────────────────────────────────────────────────────────
 * Fetches GPU readiness from /ready on mount.
 */
import { useEffect, useState } from "react";
import apiClient from "../api";

export default function useGpuStatus() {
  const [gpu, setGpu] = useState({ ready: false, device: "—" });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const check = async () => {
      try {
        const { data } = await apiClient.get(`/ready`);
        setGpu({ ready: data.status === "ready", device: data.device || "—" });
      } catch {
        setGpu({ ready: false, device: "unavailable" });
      } finally {
        setLoading(false);
      }
    };
    check();
    // Re-check every 30s
    const id = setInterval(check, 30000);
    return () => clearInterval(id);
  }, []);

  return { gpu, loading };
}
