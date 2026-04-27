#!/usr/bin/env bash

set -e

echo "Setting up .env file..."

# ─────────────────────────────────────────────
# Detect system values
# ─────────────────────────────────────────────
HOST_PROJECT_ROOT=$(pwd)
AIRFLOW_UID=$(id -u)

# Get Docker group ID safely
if getent group docker > /dev/null 2>&1; then
    DOCKER_GID=$(getent group docker | cut -d: -f3)
else
    echo "Docker group not found. Using default GID=1000"
    DOCKER_GID=1000
fi

# Generate Fernet key
FERNET_KEY=$(python3 - <<EOF
from cryptography.fernet import Fernet
print(Fernet.generate_key().decode())
EOF
)

# ─────────────────────────────────────────────
# Prompt user for optional inputs
# ─────────────────────────────────────────────
read -p "Enter SMTP email (e.g. your_email@gmail.com): " SMTP_USER
read -s -p "Enter SMTP app password: " SMTP_PASSWORD
echo ""

# ─────────────────────────────────────────────
# Create .env file
# ─────────────────────────────────────────────
cat <<EOL > .env
# Auto-generated .env file

# Airflow Configuration
AIRFLOW_UID=${AIRFLOW_UID}
DOCKER_GID=${DOCKER_GID}
FERNET_KEY=${FERNET_KEY}

# Airflow Web UI Login
_AIRFLOW_WWW_USER_USERNAME=airflow
_AIRFLOW_WWW_USER_PASSWORD=airflow

# API Auth
AIRFLOW__API_AUTH__JWT_SECRET=me22b214

# Project Paths
HOST_PROJECT_ROOT=${HOST_PROJECT_ROOT}

# Email (SMTP) Configuration
SMTP_MAIL_FROM=mlops-me22b214@alert.com
SMTP_USER=${SMTP_USER}
SMTP_PASSWORD=${SMTP_PASSWORD}
EOL

echo ".env file created successfully!"
echo "Location: $(pwd)/.env"