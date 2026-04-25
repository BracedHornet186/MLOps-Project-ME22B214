"""
api/middleware.py
─────────────────────────────────────────────────────────────────────────────
Request access-logging middleware.

Logs every HTTP request as structured JSON with:
  - request_id   (UUID4, also returned in X-Request-ID header)
  - timestamp     (ISO 8601 UTC)
  - method + path
  - status code
  - duration_ms
  - client_ip     (anonymized — last octet replaced with 'xxx')
  - user_agent    (truncated to 120 chars)
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

access_logger = logging.getLogger("access_log")
access_logger.setLevel(logging.INFO)


def _anonymize_ip(ip: str) -> str:
    """Replace the last octet of an IPv4 address with 'xxx'."""
    parts = ip.split(".")
    if len(parts) == 4:
        parts[-1] = "xxx"
        return ".".join(parts)
    # IPv6 or unknown — hash it
    return ip[:ip.rfind(":")] + ":xxx" if ":" in ip else ip


class AccessLogMiddleware(BaseHTTPMiddleware):
    """Structured JSON access logger for every API request."""

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())
        start = time.monotonic()

        # Attach request_id to request state for downstream use
        request.state.request_id = request_id

        try:
            response = await call_next(request)
        except Exception:
            # Log the error and re-raise
            duration_ms = (time.monotonic() - start) * 1000
            self._log(request, request_id, 500, duration_ms)
            raise

        duration_ms = (time.monotonic() - start) * 1000
        self._log(request, request_id, response.status_code, duration_ms)

        # Add request ID to response headers for traceability
        response.headers["X-Request-ID"] = request_id
        return response

    @staticmethod
    def _log(request: Request, request_id: str, status: int, duration_ms: float) -> None:
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "")[:120]

        entry = {
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "method": request.method,
            "path": str(request.url.path),
            "query": str(request.url.query) if request.url.query else None,
            "status": status,
            "duration_ms": round(duration_ms, 2),
            "client_ip": _anonymize_ip(client_ip),
            "user_agent": user_agent or None,
        }
        # Remove None values for cleaner logs
        entry = {k: v for k, v in entry.items() if v is not None}
        access_logger.info(json.dumps(entry, separators=(",", ":")))
