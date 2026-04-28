Security & Compliance
=====================

This page documents the security controls implemented in the system, including
secrets management, API authentication, access logging, TLS configuration, and
CI/CD security scanning.

----

Overview of Security Controls
-------------------------------

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Control
     - Implementation
   * - **Secrets management**
     - Docker Secrets (file-based, mounted at ``/run/secrets/``)
   * - **API authentication**
     - JWT Bearer tokens (HS256, 15-minute expiry)
   * - **Access logging**
     - Structured JSON logs with anonymised client IPs
   * - **Transport security**
     - TLS 1.3 via nginx (external termination)
   * - **Container scanning**
     - Trivy on every push (CRITICAL/HIGH fails CI)
   * - **Dependency auditing**
     - ``pip-audit`` on every push

----

1. Docker Secrets Management
------------------------------

Plaintext passwords are never stored in environment variables or committed to
version control. Two secrets are provisioned via Docker Secrets:

.. list-table::
   :widths: 30 35 35
   :header-rows: 1

   * - Secret Name
     - File Path (container)
     - Used By
   * - ``jwt_secret``
     - ``/run/secrets/jwt_secret``
     - ``ray-serve`` (API token signing)
   * - ``grafana_admin_password``
     - ``/run/secrets/grafana_admin_password``
     - ``grafana`` (admin login)

**Generating secrets**

.. code-block:: bash

   ./generate_secrets.sh
   chmod 644 ./secrets/*

The ``secrets/`` directory is listed in ``.gitignore``.

**How secrets are loaded in code**

The ``api/auth.py`` module reads secrets with a Docker Secrets → environment
variable → hardcoded-default cascade:

.. code-block:: python

   def _load_secret(name: str, env_fallback: str, default: str = "") -> str:
       secret_path = Path(f"/run/secrets/{name}")
       if secret_path.exists():
           return secret_path.read_text().strip()
       return os.environ.get(env_fallback, default)

In production (Docker), ``/run/secrets/jwt_secret`` is always present.
In development (local), set the ``JWT_SECRET`` environment variable.

.. warning::
   The default development fallback key (``"dev-only-insecure-key"``) must never
   be used in production. Always run ``generate_secrets.sh`` before deploying.

----

2. JWT Bearer Token Authentication
------------------------------------

All API endpoints except ``/health``, ``/ready``, and ``/metrics`` are protected
by JWT Bearer token authentication.

**Token lifecycle**

1. Client calls ``POST /auth/token`` with username and password.
2. The API validates credentials against the configured hash.
3. A signed HS256 JWT is returned with a 15-minute expiry (``exp`` claim).
4. The client includes the token in ``Authorization: Bearer <token>`` headers.
5. The ``get_current_user`` FastAPI dependency validates the signature and expiry
   on every protected request.

**Token structure**

Tokens are standard JWT with these payload fields:

- ``sub`` — username
- ``iat`` — issued-at timestamp
- ``exp`` — expiry timestamp (``iat + 900`` seconds by default)

**Implementation details**

The JWT implementation uses only Python standard library (``hmac``, ``hashlib``) —
no external JWT library is required. Signature verification uses
``hmac.compare_digest`` to prevent timing attacks.

**Disabling authentication for development**

Set ``AUTH_ENABLED=false`` in the ``ray-serve`` environment:

.. code-block:: yaml

   environment:
     AUTH_ENABLED: "false"

.. warning::
   This disables all authentication checks and returns ``"anonymous"`` for all
   requests. Never use this in production.

----

3. Access Logging
------------------

Every HTTP request is logged as a structured JSON record by the
``AccessLogMiddleware`` (``api/middleware.py``).

**Log fields**

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Field
     - Description
   * - ``request_id``
     - UUID4, unique per request; also returned in ``X-Request-ID`` header
   * - ``timestamp``
     - ISO 8601 UTC timestamp
   * - ``method``
     - HTTP method (GET, POST, etc.)
   * - ``path``
     - URL path
   * - ``query``
     - Query string (omitted if empty)
   * - ``status``
     - HTTP response status code
   * - ``duration_ms``
     - Request processing time in milliseconds
   * - ``client_ip``
     - Anonymised IPv4 address (last octet replaced with ``xxx``)
   * - ``user_agent``
     - Truncated to 120 characters

**Example log entry**

.. code-block:: json

   {
     "request_id": "a1b2c3d4-...",
     "timestamp": "2024-04-28T10:00:00.123+00:00",
     "method": "POST",
     "path": "/upload",
     "status": 202,
     "duration_ms": 45.2,
     "client_ip": "192.168.1.xxx",
     "user_agent": "Mozilla/5.0 ..."
   }

**IP anonymisation**

For IPv4, the last octet is replaced with ``xxx``. For IPv6, the final segment
is replaced. This provides sufficient traceability for security audits while
protecting individual user privacy.

----

4. TLS 1.3 (Transport Security)
---------------------------------

External traffic is protected by TLS 1.3 via nginx:

- Port 80 redirects to HTTPS (except ``/health`` for Docker healthchecks).
- Port 443 serves the frontend with ``ssl_protocols TLSv1.3``.
- Certificates are mounted from ``./certs/`` as a read-only Docker volume.

**Generating self-signed certificates (development)**

.. code-block:: bash

   ./generate-certs.sh

For production, replace ``certs/server.crt`` and ``certs/server.key`` with
certificates from a trusted Certificate Authority (e.g., Let's Encrypt).

The ``certs/`` directory is listed in ``.gitignore``.

----

5. CI/CD Security Scanning
----------------------------

The GitHub Actions workflow ``.github/workflows/security.yml`` runs on every
push to ``main`` or ``develop`` and every pull request.

**Trivy container image scanning**

Three Docker images are scanned:

- ``ray-serve`` (``api/Dockerfile``)
- ``scene3d-ui`` (``frontend/Dockerfile``)
- ``ml-pipeline-builder`` (``Dockerfile.pipeline``)

The scan fails the CI pipeline on any ``CRITICAL`` or ``HIGH`` severity
vulnerability that has a fix available. SARIF reports are uploaded to the
GitHub Security tab.

**pip-audit dependency auditing**

Python dependencies from both ``pyproject.toml`` (root pipeline) and
``api/pyproject.toml`` (API container) are audited for known CVEs. Reports
are uploaded as workflow artifacts.

----

Protected API Endpoints
------------------------

The following endpoints require a valid JWT token:

``/upload``, ``/drift``, ``/drift/trigger-retrain``,
``/status/{job_id}``, ``/jobs/{job_id}``, ``/download/*``,
``/clusters/*``, ``/jobs/{job_id}/insights``

The following endpoints are **unauthenticated** (infrastructure use only):

``/health``, ``/ready``, ``/metrics``, ``/auth/token``

----

Alertmanager Authentication
-----------------------------

The Alertmanager webhook that triggers Airflow retraining DAGs uses HTTP Basic
Auth within the ``mlops_net`` Docker network:

.. code-block:: yaml

   basic_auth:
     username: 'airflow'
     password: 'airflow'

For external access, this should be replaced with a scoped service account
token and the Airflow API should not be exposed outside the Docker network.

----

Environment Variables
----------------------

The following environment variables affect security behaviour:

.. list-table::
   :widths: 30 20 50
   :header-rows: 1

   * - Variable
     - Default
     - Description
   * - ``AUTH_ENABLED``
     - ``true``
     - Set to ``false`` to disable JWT auth (dev only)
   * - ``JWT_EXPIRY_SECONDS``
     - ``900``
     - Token lifetime in seconds
   * - ``JWT_SECRET``
     - (from Docker Secret)
     - HMAC signing key fallback for local dev
   * - ``API_USERNAME``
     - ``admin``
     - API login username
   * - ``API_PASSWORD_HASH``
     - SHA-256 of "admin"
     - SHA-256 hex digest of the API password

----

Security Recommendations for Production
-----------------------------------------

- Replace the default ``admin/admin`` credentials with a strong password and
  store its SHA-256 hash in a Docker Secret (``api_password_hash``).
- Use certificates from a trusted CA instead of self-signed certificates.
- Restrict ``mlops_net`` to internal-only traffic; never expose Airflow, MLflow,
  or Prometheus ports to the public internet.
- Enable Grafana authentication (set ``GF_AUTH_ANONYMOUS_ENABLED=false``).
- Rotate ``jwt_secret`` and ``grafana_admin_password`` regularly.
- Review Trivy and pip-audit reports in the GitHub Security tab after every push.
