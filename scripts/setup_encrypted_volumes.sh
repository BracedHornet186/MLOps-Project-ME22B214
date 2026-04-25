#!/bin/bash
# scripts/setup_encrypted_volumes.sh
# ─────────────────────────────────────────────────────────────────────────────
# AES-256 encrypted volumes for model weights and artifacts.
#
# This script creates LUKS-encrypted loop devices for sensitive data storage.
# Requires: sudo, cryptsetup, at least 2GB free disk space.
#
# For development/demo environments, the bind-mount fallback (below) is
# sufficient. The LUKS approach is for production deployments where data-at-rest
# encryption is a compliance requirement.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENCRYPTED_DIR="$PROJECT_ROOT/encrypted_volumes"
VOLUME_SIZE_MB="${1:-1024}"  # Default 1GB per volume

echo "═══════════════════════════════════════════════════════════════"
echo "  AES-256 Encrypted Volumes Setup"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# ── Check prerequisites ──────────────────────────────────────────────────────

if ! command -v cryptsetup &>/dev/null; then
    echo "ERROR: cryptsetup is not installed."
    echo "Install with: sudo apt-get install cryptsetup"
    exit 1
fi

if [ "$EUID" -ne 0 ]; then
    echo "ERROR: This script must be run as root (sudo)."
    echo ""
    echo "  Usage: sudo bash scripts/setup_encrypted_volumes.sh [size_mb]"
    echo ""
    echo "  ──── ALTERNATIVE: Bind-Mount Fallback (no sudo needed) ────"
    echo "  For development, just create the directories:"
    echo ""
    echo "    mkdir -p encrypted_volumes/mlflow-artifacts"
    echo "    mkdir -p encrypted_volumes/reconstruction-results"
    echo ""
    echo "  Then update docker-compose.yaml volumes to use bind mounts."
    echo "═══════════════════════════════════════════════════════════════"
    
    # Create the bind-mount fallback automatically
    echo ""
    echo "Creating bind-mount fallback directories..."
    mkdir -p "$ENCRYPTED_DIR/mlflow-artifacts"
    mkdir -p "$ENCRYPTED_DIR/reconstruction-results"
    echo "  ✓ $ENCRYPTED_DIR/mlflow-artifacts"
    echo "  ✓ $ENCRYPTED_DIR/reconstruction-results"
    echo ""
    echo "Done. These directories are ready for use as Docker bind mounts."
    exit 0
fi

# ── Create LUKS-encrypted loop volumes ────────────────────────────────────────

mkdir -p "$ENCRYPTED_DIR"

setup_encrypted_volume() {
    local name="$1"
    local img_path="$ENCRYPTED_DIR/${name}.img"
    local mapper_name="mlops_${name}"
    local mount_path="$ENCRYPTED_DIR/$name"

    if [ -f "$img_path" ]; then
        echo "  ⚠ $img_path already exists — skipping creation"
        return
    fi

    echo "  Creating ${VOLUME_SIZE_MB}MB encrypted volume: $name"

    # Create a sparse file
    dd if=/dev/zero of="$img_path" bs=1M count="$VOLUME_SIZE_MB" status=progress

    # Set up LUKS (AES-256-XTS)
    echo "  Setting up LUKS encryption (AES-256-XTS)..."
    cryptsetup luksFormat --type luks2 --cipher aes-xts-plain64 \
        --key-size 512 --hash sha256 "$img_path"

    # Open the encrypted volume
    cryptsetup open "$img_path" "$mapper_name"

    # Format with ext4
    mkfs.ext4 "/dev/mapper/$mapper_name"

    # Mount
    mkdir -p "$mount_path"
    mount "/dev/mapper/$mapper_name" "$mount_path"
    chmod 777 "$mount_path"

    echo "  ✓ $name encrypted volume mounted at $mount_path"
}

setup_encrypted_volume "mlflow-artifacts"
setup_encrypted_volume "reconstruction-results"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Encrypted volumes ready!"
echo ""
echo "  To close volumes after stopping Docker:"
echo "    sudo umount $ENCRYPTED_DIR/mlflow-artifacts"
echo "    sudo cryptsetup close mlops_mlflow-artifacts"
echo "    sudo umount $ENCRYPTED_DIR/reconstruction-results"
echo "    sudo cryptsetup close mlops_reconstruction-results"
echo "═══════════════════════════════════════════════════════════════"
