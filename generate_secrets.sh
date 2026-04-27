#!/bin/bash
# scripts/generate_secrets.sh
# ─────────────────────────────────────────────────────────────────────────────
# Generates Docker Secret files for the MLOps stack.
# Run once before first `docker compose up`.
# All secrets are gitignored and never committed.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SECRETS_DIR="$(pwd)/secrets"

mkdir -p "$SECRETS_DIR"

generate_if_missing() {
    local name="$1"
    local length="${2:-32}"
    local path="$SECRETS_DIR/$name"

    if [ -f "$path" ]; then
        echo "  ✓ $name already exists — skipping"
    else
        openssl rand -hex "$length" | tr -d '\n' > "$path"
        chmod 600 "$path"
        echo "  ✓ $name generated"
    fi
}

echo "Generating Docker Secrets in $SECRETS_DIR …"
echo ""

generate_if_missing "jwt_secret" 32
generate_if_missing "grafana_admin_password" 16

echo ""
echo "Done. All secrets stored in $SECRETS_DIR/"
echo "These files are gitignored and must never be committed."
