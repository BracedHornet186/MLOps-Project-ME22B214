/**
 * useJobStatus.js
 * ─────────────────────────────────────────────────────────────────
 * Custom hook that polls /status/{jobId} every 2 seconds.
 * Returns { status, error, isPolling }.
 */
import { useEffect, useRef, useState, useCallback } from "react";
import apiClient from "../api";
const POLL_MS = 2000;

export default function useJobStatus(jobId) {
  const [status, setStatus] = useState(null);
  const [error, setError] = useState(null);
  const [isPolling, setIsPolling] = useState(false);
  const timerRef = useRef(null);

  const stop = useCallback(() => {
    clearInterval(timerRef.current);
    timerRef.current = null;
    setIsPolling(false);
  }, []);

  useEffect(() => {
    if (!jobId) return;
    setIsPolling(true);
    setError(null);

    const poll = async () => {
      try {
        const { data } = await apiClient.get(`/status/${jobId}`);
        setStatus(data);

        if (data.stage === "success" || data.stage === "failed") {
          stop();
        }
      } catch (err) {
        setError(err.message || "Polling failed");
      }
    };

    poll();
    timerRef.current = setInterval(poll, POLL_MS);

    return () => {
      clearInterval(timerRef.current);
    };
  }, [jobId, stop]);

  return { status, error, isPolling, stop };
}
