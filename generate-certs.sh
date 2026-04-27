#!/usr/bin/env bash

set -e

echo "Generating self-signed TLS certs..."

mkdir -p certs

if [ ! -f certs/server.crt ] || [ ! -f certs/server.key ]; then
  echo "Generating self-signed TLS certs..."
  openssl req -x509 -nodes -days 365 \
    -newkey rsa:2048 \
    -keyout certs/server.key \
    -out certs/server.crt \
    -subj "/CN=localhost"
fi

echo "Certificates installed successfully!"