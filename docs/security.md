# Security & Compliance — Implementation Walkthrough

## Changes Made

### 1. Docker Secrets Management (no plaintext passwords)

**Files created:**
- [generate_secrets.sh](file:///home/abhiyaan-cu/Yash/MLOps-Project-ME22B214/scripts/generate_secrets.sh) — generates `secrets/jwt_secret`, `secrets/postgres_password`, `secrets/grafana_admin_password`

**Files modified:**
- [docker-compose.yaml](file:///home/abhiyaan-cu/Yash/MLOps-Project-ME22B214/docker-compose.yaml):
  - Added top-level `secrets:` block with 3 file-based secrets
  - Postgres: `POSTGRES_PASSWORD` → `POSTGRES_PASSWORD_FILE: /run/secrets/postgres_password`
  - Grafana: `GF_SECURITY_ADMIN_PASSWORD` → `GF_SECURITY_ADMIN_PASSWORD__FILE`
  - Airflow: `SQL_ALCHEMY_CONN` → `SQL_ALCHEMY_CONN_CMD` (reads secret at runtime)
  - Ray-serve: mounts `jwt_secret`
- [.gitignore](file:///home/abhiyaan-cu/Yash/MLOps-Project-ME22B214/.gitignore): added `secrets/`, `certs/`

---

### 2. JWT Bearer Token Authentication (15-min expiry)

**Files created:**
- [auth.py](file:///home/abhiyaan-cu/Yash/MLOps-Project-ME22B214/api/auth.py):
  - Zero-dependency HS256 JWT (stdlib `hmac`/`hashlib` — no PyJWT needed)
  - `POST /auth/token` — login endpoint, returns `{"access_token": "...", "expires_in": 900}`
  - `get_current_user` — FastAPI `Depends()` for protected endpoints
  - `AUTH_ENABLED` env toggle (set to `true` by default, set `false` for dev)
  - Secret loaded from `/run/secrets/jwt_secret` with env fallback

**Files modified:**
- [serve_app.py](file:///home/abhiyaan-cu/Yash/MLOps-Project-ME22B214/api/serve_app.py):
  - Mounted `auth_router` on the FastAPI app
  - Added `Depends(get_current_user)` to 9 protected endpoints
  - Left `/health`, `/ready`, `/metrics` unauthenticated (infrastructure)

**Protected endpoints:** `/upload`, `/drift`, `/drift/trigger-retrain`, `/status/{job_id}`, `/jobs/{job_id}`, `/download/*`, `/clusters/*`, `/jobs/{job_id}/insights`

---

### 3. Access Logging (structured JSON, anonymized IPs)

**Files created:**
- [middleware.py](file:///home/abhiyaan-cu/Yash/MLOps-Project-ME22B214/api/middleware.py):
  - Logs every request as JSON: `request_id`, `timestamp`, `method`, `path`, `status`, `duration_ms`, `client_ip` (anonymized), `user_agent`
  - Adds `X-Request-ID` header to all responses for traceability

---

### 4. TLS 1.3 (External nginx termination)

**Files modified:**
- [nginx.conf](file:///home/abhiyaan-cu/Yash/MLOps-Project-ME22B214/frontend/nginx.conf):
  - Port 443 with `ssl_protocols TLSv1.3`
  - Port 80 redirects to HTTPS (except `/health` for Docker healthcheck)
- [docker-compose.yaml](file:///home/abhiyaan-cu/Yash/MLOps-Project-ME22B214/docker-compose.yaml):
  - Exposes port `443:443`
  - Mounts `./certs:/etc/nginx/certs:ro`

**Files created:**
- `certs/server.crt` + `certs/server.key` — self-signed cert (gitignored)

---

### 5. AES-256 Encrypted Volumes

**Files created:**
- [setup_encrypted_volumes.sh](file:///home/abhiyaan-cu/Yash/MLOps-Project-ME22B214/scripts/setup_encrypted_volumes.sh):
  - LUKS2 encrypted loop devices (AES-256-XTS) when run as root
  - Automatic bind-mount fallback for non-root development

---

### 6. CI — Trivy Image Scanning + pip-audit

**Files created:**
- [security.yml](file:///home/abhiyaan-cu/Yash/MLOps-Project-ME22B214/.github/workflows/security.yml):
  - Trivy scans `ray-serve`, `scene3d-ui`, `ml-pipeline-builder` images
  - Fails on CRITICAL/HIGH vulnerabilities
  - Uploads SARIF reports to GitHub Security tab
  - pip-audit job audits Python dependencies

---

### 7. Dockerfile Updates

**Files modified:**
- [Dockerfile](file:///home/abhiyaan-cu/Yash/MLOps-Project-ME22B214/api/Dockerfile):
  - Copies `api/auth.py` and `api/middleware.py`
  - Creates `api/__init__.py` for package import

---

## Verification

### Docker Compose config validates ✓
```
docker compose config --quiet  # exits 0, no errors
```

### All 3 secrets mounted correctly ✓
- `jwt_secret` → `ray-serve` at `/run/secrets/jwt_secret`
- `postgres_password` → `postgres` + all Airflow services
- `grafana_admin_password` → `grafana`

### JWT Auth Flow
```bash
# 1. Get token
curl -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin"}'

# 2. Use token on protected endpoint
curl http://localhost:8000/jobs/<job_id> \
  -H "Authorization: Bearer <token>"

# 3. Infra endpoints remain open
curl http://localhost:8000/health   # 200 OK (no auth needed)
curl http://localhost:8000/metrics  # 200 OK (no auth needed)
```

### To disable auth for development
Set `AUTH_ENABLED: "false"` in the `ray-serve` environment in `docker-compose.yaml`.

## Non-Breaking Guarantees
- `/health`, `/ready`, `/metrics` — unauthenticated, healthchecks + Prometheus keep working
- DVC pipeline — runs in Airflow containers, doesn't touch the API
- MLflow — service config unchanged, only password source moved to Docker Secret
- Grafana/Prometheus — monitoring stack fully functional
