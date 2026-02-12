#!/bin/bash
# 02-packages.sh - Install system-level dependencies
# Installs all required OS packages for the Rails application stack.
set -euo pipefail

SCRIPT_NAME="$(basename "$0")"
echo "[$SCRIPT_NAME] Starting package installation..."

# ---------- Configuration ----------
REQUIRED_PACKAGES=(
  git
  gcc
  gcc-c++
  make
  automake
  autoconf
  libtool
  bison
  curl
  wget
  tar
  gzip
  zlib-devel
  readline-devel
  libffi-devel
  libyaml-devel
  openssl-devel
  postgresql-devel
  ImageMagick
  ImageMagick-devel
  jq
  docker
)

# ---------- Idempotency Check ----------
MISSING=()
for pkg in "${REQUIRED_PACKAGES[@]}"; do
  if ! rpm -q "$pkg" &>/dev/null; then
    MISSING+=("$pkg")
  fi
done

if [[ ${#MISSING[@]} -eq 0 ]]; then
  echo "[$SCRIPT_NAME] All required packages are already installed. Skipping."
  exit 0
fi

# ---------- Install ----------
echo "[$SCRIPT_NAME] Installing ${#MISSING[@]} missing packages..."
sudo dnf update -y --quiet
sudo dnf install -y "${MISSING[@]}"

# ---------- Enable Docker ----------
sudo systemctl enable docker
sudo systemctl start docker
sudo usermod -aG docker ec2-user || true

# ---------- Verify ----------
FAILED=()
for pkg in "${REQUIRED_PACKAGES[@]}"; do
  if ! rpm -q "$pkg" &>/dev/null; then
    FAILED+=("$pkg")
  fi
done

if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "[$SCRIPT_NAME] ERROR: Failed to install: ${FAILED[*]}"
  exit 1
fi

echo "[$SCRIPT_NAME] All packages installed successfully."
echo "[$SCRIPT_NAME] Done."
